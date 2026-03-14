#!/usr/bin/env python3
"""Precompute wine embeddings and taste levels for lightweight serving.

Run once offline, then upload the resulting precomputed.zip to GDrive.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uain.config import DATA_DIR, WINE_WEIGHTS
from uain.scraper.parsing import get_flavour, get_grapes, load_wines

METADATA_COLS = [
    "id",
    "wine_id",
    "wine_seo_name",
    "winery_seo_name",
    "wine_type",
    "year",
    "region_seo_name",
    "ratings_average",
    "structure_sweetness",
    "structure_acidity",
    "structure_tannin",
    "style_body",
    "structure_intensity",
    "structure_fizziness",
]

# Wine structure features that describe taste/mouthfeel (used for embeddings)
EMBEDDING_FEATURES = [
    "structure_sweetness",
    "structure_acidity",
    "structure_tannin",
    "style_body",
    "structure_intensity",
    "structure_fizziness",
    "ratings_average",
    "is_natural",
]

STRUCTURE_TO_TASTE = {
    "weight": "style_body",
    "sweet": "structure_sweetness",
    "acid": "structure_acidity",
    "bitter": "structure_tannin",
    "salt": "structure_intensity",
    "piquant": "structure_fizziness",
    "fat": "style_body",
    "tannin": "structure_tannin",
}

# Number of embedding dimensions stored for similarity search
SEARCH_DIMS = 8
# Number of embedding dimensions stored for visualization
VIZ_DIMS = 2


def _score_to_level(value: float, weight_map: dict[int, tuple[float, float]]) -> int:
    for level in sorted(weight_map.keys()):
        lo, hi = weight_map[level]
        if lo <= value <= hi:
            return level
    return max(weight_map.keys())


def _build_feature_matrix(wines: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix for embedding: structure + flavors + grapes."""
    df = wines[["id"]].copy()

    # 1. Structure / numeric features
    for col in EMBEDDING_FEATURES:
        if col in wines.columns:
            df[col] = pd.to_numeric(wines[col], errors="coerce").fillna(0)

    # 2. Flavor scores (pivoted: one column per flavor group)
    if "flavor" in wines.columns:
        try:
            flavors = get_flavour(wines)
            # flavors has 'id' + one column per flavor name
            flavor_cols = [c for c in flavors.columns if c != "id"]
            df = df.merge(flavors, on="id", how="left")
            df[flavor_cols] = df[flavor_cols].fillna(0)
        except (ValueError, KeyError):
            pass  # no flavor data available

    # 3. Grape indicators (one-hot: one column per grape variety)
    if "style_grapes" in wines.columns:
        try:
            grapes = get_grapes(wines)
            grape_cols = [c for c in grapes.columns if c != "id"]
            # Keep only grapes that appear in at least 1% of wines to avoid noise
            min_count = max(1, int(0.01 * len(wines)))
            frequent = [c for c in grape_cols if grapes[c].sum() >= min_count]
            if frequent:
                df = df.merge(grapes[["id"] + frequent], on="id", how="left")
                df[frequent] = df[frequent].fillna(0)
        except (ValueError, KeyError):
            pass  # no grape data available

    feature_cols = [c for c in df.columns if c != "id"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


def _compute_embeddings(
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute search embeddings (higher-dim) and viz embeddings (2D).

    Returns (search_embeddings, viz_embeddings, explained_variance_ratio).
    """
    feature_cols = [c for c in feature_df.columns if c != "id"]
    X = feature_df[feature_cols].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Search embeddings: PCA retaining 99% variance, capped at SEARCH_DIMS
    pca_search = PCA(n_components=min(SEARCH_DIMS, len(feature_cols)), random_state=0)
    search_emb = pca_search.fit_transform(X_scaled)
    explained = pca_search.explained_variance_ratio_.sum()

    # Try UMAP for 2D visualization (better neighborhood preservation)
    viz_emb: np.ndarray
    try:
        from umap import UMAP

        umap = UMAP(n_components=VIZ_DIMS, random_state=0, n_neighbors=15, min_dist=0.1)
        viz_emb = umap.fit_transform(X_scaled)
        print(f"  UMAP 2D visualization embeddings computed")
    except ImportError:
        pca_viz = PCA(n_components=VIZ_DIMS, random_state=0)
        viz_emb = pca_viz.fit_transform(X_scaled)
        print(f"  PCA 2D fallback (umap-learn not installed)")

    return search_emb, viz_emb, explained


def main() -> None:
    print("Loading wines from parquet...")
    wines = load_wines()
    print(f"Loaded {len(wines)} wines")

    # Build feature matrix (structure + flavors + grapes)
    print("Building feature matrix...")
    feature_df = _build_feature_matrix(wines)
    feature_cols = [c for c in feature_df.columns if c != "id"]
    print(f"  {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    # Compute embeddings
    print("Computing embeddings...")
    search_emb, viz_emb, explained = _compute_embeddings(feature_df)
    print(f"  Search: {search_emb.shape[1]}D, explained variance: {explained:.1%}")
    print(f"  Viz: {viz_emb.shape[1]}D")

    # Drop raw nested columns before building output
    wines = wines.drop(
        columns=[c for c in ("flavor", "style_food", "style_grapes") if c in wines.columns],
    )

    # Taste levels
    for taste, source in STRUCTURE_TO_TASTE.items():
        if source in wines.columns:
            raw = wines[source].fillna(0).astype(float)
            wmap = WINE_WEIGHTS.get(taste, WINE_WEIGHTS["weight"])
            wines[taste] = raw.apply(lambda v, wm=wmap: _score_to_level(v, wm))
        else:
            wines[taste] = 2

    # Build output
    meta_cols = [c for c in METADATA_COLS if c in wines.columns]
    taste_cols = list(STRUCTURE_TO_TASTE.keys())

    out = wines[meta_cols + taste_cols].copy()

    # Store search embeddings (higher-dim for distance computation)
    for i in range(search_emb.shape[1]):
        out[f"emb_{i}"] = search_emb[:, i]

    # Store 2D viz embeddings (backward-compatible column names)
    out["pca_0"] = viz_emb[:, 0]
    out["pca_1"] = viz_emb[:, 1]

    parquet_path = DATA_DIR / "wines_precomputed.parquet"
    out.to_parquet(parquet_path, index=False)
    print(f"Saved {len(out)} wines to {parquet_path} ({parquet_path.stat().st_size / 1024:.0f} KB)")

    # Bundle into a zip for GDrive upload
    zip_path = DATA_DIR / "precomputed.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(parquet_path, parquet_path.name)
        # Include raw wines.parquet for /pair-to-food endpoint
        wines_path = DATA_DIR / "wines.parquet"
        if wines_path.exists():
            zf.write(wines_path, wines_path.name)
        for name in ("list_of_foods.csv", "descriptor_mapping_tastes.csv"):
            p = DATA_DIR / name
            if p.exists():
                zf.write(p, name)

    print(f"Zip: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    print("\nUpload precomputed.zip to Google Drive and update GDRIVE_DATA_LINK.")


if __name__ == "__main__":
    main()
