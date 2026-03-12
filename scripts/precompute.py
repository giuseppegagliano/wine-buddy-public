#!/usr/bin/env python3
"""Precompute wine embeddings and taste levels for lightweight serving."""
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

from uain.config import COUNTRY_CODE, DATA_DIR, WINE_WEIGHTS
from uain.parsing import get_flavour

METADATA_COLS = [
    "id",
    "wine_seo_name",
    "winery_seo_name",
    "wine_type",
    "year",
    "region_seo_name",
    "ratings_average",
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


def _score_to_level(value: float, weight_map: dict[int, tuple[float, float]]) -> int:
    for level in sorted(weight_map.keys()):
        lo, hi = weight_map[level]
        if lo <= value <= hi:
            return level
    return max(weight_map.keys())


def main() -> None:
    frames = []
    for color in ("red", "white", "sparkling"):
        path = DATA_DIR / f"{COUNTRY_CODE.lower()}_{color}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        print(f"No wine data in {DATA_DIR}/", file=sys.stderr)
        sys.exit(1)

    wines = pd.concat(frames).reset_index(drop=True)
    flavours = get_flavour(wines)
    wines = wines.merge(flavours, on="id")
    wines = wines.drop(
        columns=[c for c in ("flavor", "style_food", "style_grapes") if c in wines.columns],
    )

    # PCA embeddings
    df = wines.copy()
    for col in ("winery_seo_name", "region_seo_name"):
        if col in df.columns:
            ohe = pd.get_dummies(df[col])
            df = df.drop(columns=col).join(ohe)

    id_cols = {"id", "wine_id", "wine_seo_name", "region_country_code", "wine_type"}
    features = [c for c in df.columns if c not in id_cols]
    df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0))
    embeddings = pca.fit_transform(df[features])

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
    out["pca_0"] = embeddings[:, 0]
    out["pca_1"] = embeddings[:, 1]

    parquet_path = DATA_DIR / "wines_precomputed.parquet"
    out.to_parquet(parquet_path, index=False)
    print(f"Saved {len(out)} wines to {parquet_path} ({parquet_path.stat().st_size / 1024:.0f} KB)")

    # Bundle into a zip for GDrive upload
    zip_path = DATA_DIR / "precomputed.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(parquet_path, parquet_path.name)
        for name in ("list_of_foods.csv", "descriptor_mapping_tastes.csv"):
            p = DATA_DIR / name
            if p.exists():
                zf.write(p, name)

    print(f"Zip: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")
    print("\nUpload precomputed.zip to Google Drive and update GDRIVE_DATA_LINK.")


if __name__ == "__main__":
    main()
