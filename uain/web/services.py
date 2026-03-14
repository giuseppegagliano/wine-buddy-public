"""Shared service layer — reuses CLI data loading, caches heavy objects."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from uain.cli import (
    _build_embedding,
    _load_descriptor_tastes,
    _load_food_list,
    _load_wines,
    _score_to_level,
)
from uain.config import FOOD_WEIGHTS, WINE_WEIGHTS
from uain.pairing.rules import nonaroma_rules


@dataclass(frozen=True)
class WineIndex:
    wines: pd.DataFrame
    df: pd.DataFrame
    x_embedded: np.ndarray
    tree: KDTree


@functools.lru_cache(maxsize=1)
def _get_wines() -> pd.DataFrame:
    """Cached wine DataFrame — full load including flavors (for find-similar)."""
    return _load_wines()


@functools.lru_cache(maxsize=1)
def _get_wines_light() -> pd.DataFrame:
    """Cached wine DataFrame — skips slow flavor parsing (for food pairing)."""
    from uain.scraper.parsing import load_wines

    wines = load_wines()
    wines = wines.drop(
        columns=[c for c in ("flavor", "style_food", "style_grapes") if c in wines.columns],
    )
    return wines


@functools.lru_cache(maxsize=1)
def get_wine_index() -> WineIndex:
    wines = _get_wines()
    x_embedded, df, _ = _build_embedding(wines)
    tree = KDTree(x_embedded)
    return WineIndex(wines=wines, df=df, x_embedded=x_embedded, tree=tree)


def _wine_to_url(row: pd.Series) -> str:
    return f"https://www.vivino.com/w/{int(row['wine_id'])}"


def _name_similarity(query: str, name: str) -> float:
    """Score how well *query* matches *name* (0-1).

    Combines substring presence with SequenceMatcher ratio so that
    exact substring hits rank highest, but close misspellings still
    surface.
    """
    q = query.lower().replace("-", " ")
    n = name.lower().replace("-", " ")
    # Exact substring → boost
    if q in n:
        return 1.0 + SequenceMatcher(None, q, n).ratio()
    return SequenceMatcher(None, q, n).ratio()


def search_wines_by_name(query: str, limit: int = 20) -> list[dict]:
    """Fuzzy-search wines by full name (wine + winery). Returns a ranked list of candidates."""
    wines = _get_wines_light()
    full_names = (
        wines["wine_seo_name"].fillna("")
        + " "
        + wines["winery_seo_name"].fillna("")
    )
    scores = full_names.apply(lambda n: _name_similarity(query, n))
    # Keep only wines with some minimal similarity
    mask = scores > 0.3
    if not mask.any():
        return []

    scored = wines.loc[mask].copy()
    scored["_score"] = scores.loc[mask]
    scored = scored.sort_values("_score", ascending=False).head(limit)

    candidates = []
    for pos_idx, (_, row) in enumerate(scored.iterrows()):
        year = row.get("year", "")
        region = row.get("region_seo_name", "")
        candidates.append(
            {
                "idx": int(row.name) if hasattr(row, "name") else pos_idx,
                "name": row.get("wine_seo_name", ""),
                "winery": row.get("winery_seo_name", ""),
                "wine_type": row.get("wine_type", ""),
                "year": "" if pd.isna(year) else year,
                "region": "" if pd.isna(region) else region,
                "rating": round(float(row.get("ratings_average", 0) or 0), 2),
                "score": round(float(row.get("_score", 0)), 3),
            }
        )
    return candidates


def find_similar_wines(wine_idx: int, k: int = 5) -> dict:
    """Find wines with similar flavour profiles to the wine at *wine_idx*."""
    idx = get_wine_index()

    if wine_idx < 0 or wine_idx >= len(idx.wines):
        return {"query_wine": None, "matches": []}

    query_point = idx.x_embedded[wine_idx].reshape(1, -1)
    query_wine = idx.wines.iloc[wine_idx]

    dist, ind = idx.tree.query(query_point, k=k + 1)
    result = idx.wines.iloc[ind[0]].copy()
    result["distance"] = dist[0]
    result = result[result.index != wine_idx].head(k)

    matches = []
    for _, row in result.iterrows():
        year = row.get("year", "")
        region = row.get("region_seo_name", "")
        matches.append(
            {
                "name": row.get("wine_seo_name", ""),
                "winery": row.get("winery_seo_name", ""),
                "wine_type": row.get("wine_type", ""),
                "year": "" if pd.isna(year) else year,
                "region": "" if pd.isna(region) else region,
                "rating": round(float(row.get("ratings_average", 0) or 0), 2),
                "distance": round(float(row.get("distance", 0)), 4),
                "url": _wine_to_url(row),
            }
        )

    return {
        "query_wine": {
            "name": query_wine.get("wine_seo_name", ""),
            "winery": query_wine.get("winery_seo_name", ""),
            "wine_type": query_wine.get("wine_type", ""),
        },
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Food pairing
# ---------------------------------------------------------------------------

# The taste dimensions we compare between wine and food
_TASTE_DIMENSIONS = ("sweet", "acid", "salt", "piquant", "fat", "bitter")
_TASTE_LABELS = {
    "sweet": "Sweetness / Dolcezza",
    "acid": "Acidity / Acidit\u00e0",
    "salt": "Saltiness / Intensit\u00e0",
    "piquant": "Spiciness / Effervescenza",
    "fat": "Fattiness / Corpo",
    "bitter": "Bitterness / Tannicit\u00e0",
}

# Wine structure features for the AIS-style wine radar (raw columns, ~1-5 scale)
_WINE_STRUCTURE_KEYS = (
    "structure_sweetness",
    "structure_acidity",
    "structure_tannin",
    "style_body",
    "structure_intensity",
    "structure_fizziness",
)
_WINE_STRUCTURE_LABELS = {
    "structure_sweetness": "Dolcezza",
    "structure_acidity": "Acidit\u00e0",
    "structure_tannin": "Tannicit\u00e0",
    "style_body": "Corpo",
    "structure_intensity": "Intensit\u00e0",
    "structure_fizziness": "Effervescenza",
}

# Wine structure column → taste dimension mapping
_WINE_COL_MAP = {
    "weight": "style_body",
    "sweet": "structure_sweetness",
    "acid": "structure_acidity",
    "bitter": "structure_tannin",
    "salt": "structure_intensity",
    "piquant": "structure_fizziness",
    "fat": "style_body",
    "tannin": "structure_tannin",
}

def search_foods(query: str, limit: int = 20) -> list[str]:
    """Fuzzy-search foods by name."""
    foods = _load_food_list()
    food_col = foods.columns[0]
    foods[food_col] = foods[food_col].str.strip()
    mask = foods[food_col].str.contains(query, case=False, na=False)
    return foods.loc[mask, food_col].head(limit).tolist()


@functools.lru_cache(maxsize=1)
def get_all_foods() -> list[str]:
    """Return all food names for the searchable dropdown."""
    foods = _load_food_list()
    food_col = foods.columns[0]
    return foods[food_col].str.strip().dropna().tolist()


def _build_food_profile(food_name: str) -> dict[str, tuple[float, int]]:
    """Build a taste profile for a food using known profiles or descriptor mappings."""
    food_lower = food_name.lower().strip()

    # Known food taste profiles (sommelier knowledge)
    # Format: {taste: level} where level is 1-4
    _KNOWN_PROFILES: dict[str, dict[str, int]] = {
        "steak": {"weight": 4, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 4, "bitter": 1},
        "beef": {"weight": 4, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "beef tenderloin": {"weight": 4, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "lamb": {"weight": 4, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "pork": {"weight": 3, "sweet": 2, "acid": 1, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "chicken": {"weight": 2, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 2, "bitter": 1},
        "turkey": {"weight": 2, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 2, "bitter": 1},
        "duck": {"weight": 3, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "salmon": {"weight": 3, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "tuna": {"weight": 3, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 2, "bitter": 1},
        "shrimp": {"weight": 2, "sweet": 2, "acid": 1, "salt": 3, "piquant": 1, "fat": 1, "bitter": 1},
        "lobster": {"weight": 3, "sweet": 2, "acid": 1, "salt": 3, "piquant": 1, "fat": 2, "bitter": 1},
        "oyster": {"weight": 2, "sweet": 1, "acid": 2, "salt": 4, "piquant": 1, "fat": 1, "bitter": 2},
        "crab": {"weight": 2, "sweet": 2, "acid": 1, "salt": 3, "piquant": 1, "fat": 1, "bitter": 1},
        "cheese": {"weight": 3, "sweet": 1, "acid": 2, "salt": 3, "piquant": 1, "fat": 4, "bitter": 1},
        "blue cheese": {"weight": 3, "sweet": 1, "acid": 2, "salt": 4, "piquant": 2, "fat": 4, "bitter": 3},
        "goats cheese": {"weight": 2, "sweet": 1, "acid": 3, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "chocolate": {"weight": 3, "sweet": 4, "acid": 1, "salt": 1, "piquant": 1, "fat": 3, "bitter": 3},
        "pasta": {"weight": 2, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 2, "bitter": 1},
        "pizza": {"weight": 3, "sweet": 2, "acid": 3, "salt": 3, "piquant": 1, "fat": 3, "bitter": 1},
        "salad": {"weight": 1, "sweet": 1, "acid": 3, "salt": 1, "piquant": 1, "fat": 1, "bitter": 2},
        "sushi": {"weight": 2, "sweet": 2, "acid": 3, "salt": 3, "piquant": 2, "fat": 2, "bitter": 1},
        "barbecue": {"weight": 4, "sweet": 3, "acid": 2, "salt": 2, "piquant": 3, "fat": 3, "bitter": 2},
        "curry": {"weight": 3, "sweet": 1, "acid": 2, "salt": 2, "piquant": 4, "fat": 2, "bitter": 2},
        "chili": {"weight": 3, "sweet": 1, "acid": 2, "salt": 2, "piquant": 4, "fat": 2, "bitter": 1},
        "mushroom": {"weight": 2, "sweet": 1, "acid": 1, "salt": 1, "piquant": 1, "fat": 1, "bitter": 2},
        "truffle": {"weight": 3, "sweet": 1, "acid": 1, "salt": 1, "piquant": 1, "fat": 2, "bitter": 1},
        "ceviche": {"weight": 2, "sweet": 1, "acid": 4, "salt": 2, "piquant": 2, "fat": 1, "bitter": 1},
        "ham": {"weight": 3, "sweet": 1, "acid": 1, "salt": 4, "piquant": 1, "fat": 3, "bitter": 1},
        "bacon": {"weight": 3, "sweet": 1, "acid": 1, "salt": 4, "piquant": 1, "fat": 4, "bitter": 1},
        "dessert": {"weight": 2, "sweet": 4, "acid": 1, "salt": 1, "piquant": 1, "fat": 2, "bitter": 1},
        "cake": {"weight": 2, "sweet": 4, "acid": 1, "salt": 1, "piquant": 1, "fat": 3, "bitter": 1},
        "fruit": {"weight": 1, "sweet": 3, "acid": 3, "salt": 1, "piquant": 1, "fat": 1, "bitter": 1},
        "fish": {"weight": 2, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 2, "bitter": 1},
        "cod": {"weight": 2, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 1, "bitter": 1},
        "venison": {"weight": 4, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 2, "bitter": 2},
        "risotto": {"weight": 3, "sweet": 1, "acid": 1, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "soup": {"weight": 2, "sweet": 1, "acid": 1, "salt": 3, "piquant": 1, "fat": 2, "bitter": 1},
        "asparagus": {"weight": 1, "sweet": 1, "acid": 1, "salt": 1, "piquant": 1, "fat": 1, "bitter": 3},
        "artichoke": {"weight": 2, "sweet": 1, "acid": 1, "salt": 1, "piquant": 1, "fat": 1, "bitter": 3},
        "lasagna": {"weight": 4, "sweet": 2, "acid": 3, "salt": 3, "piquant": 1, "fat": 4, "bitter": 1},
        "bolognese": {"weight": 3, "sweet": 2, "acid": 3, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
        "tacos": {"weight": 3, "sweet": 1, "acid": 2, "salt": 2, "piquant": 3, "fat": 2, "bitter": 1},
        "guacamole": {"weight": 2, "sweet": 1, "acid": 3, "salt": 2, "piquant": 2, "fat": 3, "bitter": 1},
        "hummus": {"weight": 2, "sweet": 1, "acid": 2, "salt": 2, "piquant": 1, "fat": 3, "bitter": 1},
    }

    # Try exact match first, then partial match
    known = _KNOWN_PROFILES.get(food_lower)
    if known is None:
        for key, profile in _KNOWN_PROFILES.items():
            if key in food_lower or food_lower in key:
                known = profile
                break

    if known is not None:
        return {taste: (0.5, known[taste]) for taste in known}

    # Fallback: try descriptor mapping
    desc = _load_descriptor_tastes()
    food_key = food_lower.replace(" ", "_")
    food_descriptors = desc[desc["raw descriptor"].str.contains(food_key, case=False, na=False)]

    taste_profile: dict[str, tuple[float, int]] = {}
    if not food_descriptors.empty:
        nonaroma = food_descriptors[food_descriptors["type"] == "nonaroma"]
        for _, row in nonaroma.iterrows():
            taste = row.get("primary taste", "")
            if taste and taste in FOOD_WEIGHTS:
                level_str = row.get("level_2", "")
                intensity = 3 if "high" in str(level_str) else 2
                taste_profile[taste] = (0.5, intensity)

    for taste in ("weight", "sweet", "acid", "salt", "piquant", "fat", "bitter"):
        if taste not in taste_profile:
            taste_profile[taste] = (0.3, 2)

    return taste_profile


def _prepare_wine_taste_columns(wine_df: pd.DataFrame) -> pd.DataFrame:
    """Add discrete taste level columns to wine dataframe."""
    df = wine_df.copy()
    for taste, source in _WINE_COL_MAP.items():
        if source in df.columns:
            raw = df[source].fillna(0).astype(float)
            normalized = ((raw - 1) / 4).clip(0, 1)
            wmap = WINE_WEIGHTS.get(taste, WINE_WEIGHTS["weight"])
            df[taste] = normalized.apply(lambda v, wm=wmap: _score_to_level(v, wm))
        else:
            df[taste] = 2
    return df


def _pairing_score(wine_tastes: dict[str, int], food_levels: dict[str, int]) -> int:
    """Compute a 0-100 pairing match score between a wine and a food.

    Components:
    - Proximity (60%): how close wine and food taste levels are (per dimension).
    - Rule harmony (40%): sommelier rule bonuses/penalties.
    """
    # --- Proximity: 1 - |diff|/3  per dimension, averaged ---
    prox_scores = []
    for taste in _TASTE_DIMENSIONS:
        w = wine_tastes.get(taste, 2)
        f = food_levels.get(taste, 2)
        prox_scores.append(1.0 - abs(w - f) / 3.0)
    proximity = sum(prox_scores) / len(prox_scores)

    # --- Rule harmony bonuses/penalties ---
    harmony = 0.0
    checks = 0

    # Acidity: wine >= food is good
    checks += 1
    if wine_tastes.get("acid", 2) >= food_levels.get("acid", 2):
        harmony += 1.0
    else:
        harmony += 0.3

    # Sweetness: wine >= food is good
    checks += 1
    if wine_tastes.get("sweet", 2) >= food_levels.get("sweet", 2):
        harmony += 1.0
    else:
        harmony += 0.2

    # Tannin + fat synergy: high fat food + high tannin wine = great
    checks += 1
    food_fat = food_levels.get("fat", 2)
    wine_bitter = wine_tastes.get("bitter", 2)
    if food_fat >= 3 and wine_bitter >= 3:
        harmony += 1.0
    elif food_fat >= 3 and wine_bitter >= 2:
        harmony += 0.7
    else:
        harmony += 0.5

    # Tannin vs spice: high spice + high tannin = bad
    checks += 1
    food_piquant = food_levels.get("piquant", 2)
    if food_piquant >= 3 and wine_bitter >= 3:
        harmony += 0.1
    elif food_piquant >= 3 and wine_bitter <= 1:
        harmony += 1.0
    else:
        harmony += 0.6

    # Tannin vs salt: high salt + high tannin = bad
    checks += 1
    food_salt = food_levels.get("salt", 2)
    if food_salt >= 3 and wine_bitter >= 3:
        harmony += 0.1
    elif food_salt >= 3 and wine_bitter <= 1:
        harmony += 1.0
    else:
        harmony += 0.6

    # Bitter clash: both high bitter = bad
    checks += 1
    food_bitter = food_levels.get("bitter", 2)
    if food_bitter >= 3 and wine_bitter >= 3:
        harmony += 0.1
    else:
        harmony += 0.8

    harmony_score = harmony / checks

    # Weighted combination
    score = proximity * 0.6 + harmony_score * 0.4
    return max(0, min(100, round(score * 100)))


def pair_wine_to_food(food_name: str, k: int = 10) -> dict:
    """Find wines that pair with a food and return comparison data for visualization."""
    food_profile = _build_food_profile(food_name)
    food_weight = food_profile.get("weight", (0.5, 2))

    wines = _get_wines_light()
    wine_df = _prepare_wine_taste_columns(wines)

    paired = nonaroma_rules(wine_df, food_profile, food_weight)

    if paired.empty:
        return {"food_name": food_name, "food_profile": {}, "wines": []}

    if "ratings_average" in paired.columns:
        paired = paired.sort_values("ratings_average", ascending=False)

    paired = paired.head(k)

    # Build food profile data for the chart
    food_chart = {
        taste: food_profile[taste][1]
        for taste in _TASTE_DIMENSIONS
    }

    # Build wine data with taste profiles for comparison
    wines = []
    for _, row in paired.iterrows():
        year = row.get("year", "")
        region = row.get("region_seo_name", "")

        # Use the already-discretized taste columns from _prepare_wine_taste_columns
        wine_tastes = {
            taste: int(row.get(taste, 2))
            for taste in _TASTE_DIMENSIONS
        }

        # Raw wine structure values (clamped to 0-5 for the radar)
        wine_structure = {}
        for col in _WINE_STRUCTURE_KEYS:
            val = row.get(col, 0)
            val = 0.0 if pd.isna(val) else float(val)
            wine_structure[col] = round(min(val, 5.0), 2)

        match_score = _pairing_score(wine_tastes, food_chart)

        wines.append({
            "name": row.get("wine_seo_name", ""),
            "winery": row.get("winery_seo_name", ""),
            "wine_type": row.get("wine_type", ""),
            "year": "" if pd.isna(year) else year,
            "region": "" if pd.isna(region) else region,
            "rating": round(float(row.get("ratings_average", 0) or 0), 2),
            "url": _wine_to_url(row),
            "tastes": wine_tastes,
            "structure": wine_structure,
            "match": match_score,
        })

    # Sort by match score descending
    wines.sort(key=lambda w: w["match"], reverse=True)

    return {
        "food_name": food_name,
        "food_profile": food_chart,
        "taste_labels": _TASTE_LABELS,
        "taste_keys": list(_TASTE_DIMENSIONS),
        "structure_labels": _WINE_STRUCTURE_LABELS,
        "structure_keys": list(_WINE_STRUCTURE_KEYS),
        "wines": wines,
    }
