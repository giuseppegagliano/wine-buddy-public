"""Shared service layer — reuses CLI data loading, caches heavy objects."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from uain.cli import _build_embedding, _load_wines


@dataclass(frozen=True)
class WineIndex:
    wines: pd.DataFrame
    df: pd.DataFrame
    x_embedded: np.ndarray
    tree: KDTree


@functools.lru_cache(maxsize=1)
def get_wine_index() -> WineIndex:
    wines = _load_wines()
    x_embedded, df, _ = _build_embedding(wines)
    tree = KDTree(x_embedded)
    return WineIndex(wines=wines, df=df, x_embedded=x_embedded, tree=tree)


def find_similar_wines(query: str, k: int = 5) -> dict:
    """Return {'query_wine': dict|None, 'matches': list[dict], 'candidates': list[dict]}."""
    idx = get_wine_index()
    mask = idx.df["wine_seo_name"].str.contains(query, case=False, na=False)
    candidates = idx.wines.loc[mask]

    if candidates.empty:
        return {"query_wine": None, "matches": [], "candidates": []}

    query_idx = candidates.index[0]
    query_point = idx.x_embedded[query_idx].reshape(1, -1)
    query_wine = idx.wines.iloc[query_idx]

    dist, ind = idx.tree.query(query_point, k=k + 1)
    result = idx.wines.iloc[ind[0]].copy()
    result["distance"] = dist[0]
    result = result[result.index != query_idx].head(k)

    def _to_url(row: pd.Series) -> str:
        parts = f"{row['wine_seo_name']} {row['winery_seo_name']}".split()
        return "https://www.google.com/search?q=" + "+".join(parts)

    matches = []
    for _, row in result.iterrows():
        matches.append(
            {
                "name": row.get("wine_seo_name", ""),
                "winery": row.get("winery_seo_name", ""),
                "wine_type": row.get("wine_type", ""),
                "year": row.get("year", ""),
                "region": row.get("region_seo_name", ""),
                "rating": round(float(row.get("ratings_average", 0) or 0), 2),
                "distance": round(float(row.get("distance", 0)), 4),
                "url": _to_url(row),
            }
        )

    return {
        "query_wine": {
            "name": query_wine.get("wine_seo_name", ""),
            "winery": query_wine.get("winery_seo_name", ""),
            "wine_type": query_wine.get("wine_type", ""),
        },
        "matches": matches,
        "candidates": len(candidates),
    }
