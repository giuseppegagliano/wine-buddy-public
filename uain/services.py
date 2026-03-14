"""Shared data loading — loads precomputed parquet, provides lookups."""
from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from uain.config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WineIndex:
    wines: pd.DataFrame
    embeddings: np.ndarray


@functools.lru_cache(maxsize=1)
def get_wine_index() -> WineIndex:
    path = DATA_DIR / "wines_precomputed.parquet"
    if path.exists():
        logger.info("Loading precomputed wine index from %s", path)
        t0 = time.perf_counter()
        wines = pd.read_parquet(path)
        embeddings = wines[["pca_0", "pca_1"]].to_numpy()
        logger.info("Wine index loaded: %d wines in %.2fs", len(wines), time.perf_counter() - t0)
        return WineIndex(wines=wines, embeddings=embeddings)

    # Fallback: compute embeddings on the fly from raw data (slower first call)
    logger.warning("Precomputed parquet not found at %s — computing embeddings on the fly (slow)", path)
    from uain.cli import _build_embedding, _load_wines

    t0 = time.perf_counter()
    wines = _load_wines()
    logger.info("Loaded %d wines in %.2fs, building embeddings...", len(wines), time.perf_counter() - t0)
    t1 = time.perf_counter()
    x_embedded, _, _ = _build_embedding(wines)
    logger.info("Embeddings built in %.2fs", time.perf_counter() - t1)
    return WineIndex(wines=wines, embeddings=x_embedded)


def find_similar(query: str, k: int = 5) -> dict:
    """Find wines similar to a search term."""
    idx = get_wine_index()
    mask = idx.wines["wine_seo_name"].str.contains(query, case=False, na=False)
    candidates = idx.wines.loc[mask]

    if candidates.empty:
        return {"query_wine": None, "matches": [], "candidates": 0}

    query_pos = candidates.index[0]
    query_point = idx.embeddings[query_pos]
    query_wine = idx.wines.iloc[query_pos]

    dists = np.linalg.norm(idx.embeddings - query_point, axis=1)
    dists[query_pos] = np.inf
    top_indices = np.argsort(dists)[:k]

    matches = []
    for i in top_indices:
        row = idx.wines.iloc[i]
        matches.append({
            "name": row.get("wine_seo_name", ""),
            "winery": row.get("winery_seo_name", ""),
            "wine_type": row.get("wine_type", ""),
            "year": row.get("year", ""),
            "region": row.get("region_seo_name", ""),
            "rating": round(float(row.get("ratings_average", 0) or 0), 2),
            "distance": round(float(dists[i]), 4),
            "url": "https://www.google.com/search?q=" + "+".join(
                f"{row['wine_seo_name']} {row['winery_seo_name']}".split()
            ),
        })

    return {
        "query_wine": {
            "name": query_wine.get("wine_seo_name", ""),
            "winery": query_wine.get("winery_seo_name", ""),
            "wine_type": query_wine.get("wine_type", ""),
        },
        "matches": matches,
        "candidates": len(candidates),
    }
