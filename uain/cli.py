"""Wine Buddy CLI — find similar wines and pair wines to food."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from uain.config import COUNTRY_CODE, FOOD_WEIGHTS, ROOT_PATH, WINE_WEIGHTS
from uain.parsing import get_flavour
from uain.rules import nonaroma_rules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = ROOT_PATH / "data"
REF_DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_wines() -> pd.DataFrame:
    """Load and merge scraped wine CSVs from data/."""
    frames = []
    for color in ("red", "white", "sparkling"):
        path = DATA_DIR / f"{COUNTRY_CODE.lower()}_{color}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        print(
            f"No wine data found in {DATA_DIR}/. Run the scraper first (see the notebook).",
            file=sys.stderr,
        )
        sys.exit(1)
    wines = pd.concat(frames).reset_index(drop=True)
    flavours = get_flavour(wines)
    wines = wines.merge(flavours, on="id")
    wines = wines.drop(
        columns=[c for c in ("flavor", "style_food", "style_grapes") if c in wines.columns],
    )
    return wines


def _build_embedding(wines: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """One-hot encode categoricals, PCA-embed, return (x_embedded, df, features)."""
    df = wines.copy()
    for col in ("winery_seo_name", "region_seo_name"):
        if col in df.columns:
            ohe = pd.get_dummies(df[col])
            df = df.drop(columns=col).join(ohe)

    id_cols = {"id", "wine_id", "wine_seo_name", "region_country_code", "wine_type"}
    features = [c for c in df.columns if c not in id_cols]
    # coerce non-numeric values (e.g. 'N.V.' in year) and fill NaNs
    df[features] = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

    pca = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=0))
    x_embedded = pca.fit_transform(df[features])
    return x_embedded, df, features


def _load_food_list() -> pd.DataFrame:
    path = REF_DATA_DIR / "list_of_foods.csv"
    return pd.read_csv(path)


def _load_descriptor_tastes() -> pd.DataFrame:
    path = REF_DATA_DIR / "descriptor_mapping_tastes.csv"
    return pd.read_csv(path)


def _score_to_level(value: float, weight_map: dict[int, tuple[float, float]]) -> int:
    """Map a continuous 0-1 score to a discrete 1-4 level."""
    for level in sorted(weight_map.keys()):
        lo, hi = weight_map[level]
        if lo <= value <= hi:
            return level
    return max(weight_map.keys())


def _wine_taste_profile(wine_row: pd.Series) -> dict[str, tuple[float, int]]:
    """Build a discrete taste profile from a wine row's structure columns."""
    mapping = {
        "weight": "style_body",
        "sweet": "structure_sweetness",
        "acid": "structure_acidity",
        "bitter": "structure_tannin",  # tannin maps to bitterness perception
        "salt": "structure_intensity",  # intensity as a proxy
        "piquant": "structure_fizziness",  # fizziness as a proxy
        "fat": "style_body",  # body also correlates with mouthfeel/fat
        "tannin": "structure_tannin",
    }
    profile = {}
    for taste, col in mapping.items():
        raw = float(wine_row.get(col, 0) or 0)
        level = _score_to_level(raw, WINE_WEIGHTS.get(taste, WINE_WEIGHTS["weight"]))
        profile[taste] = (raw, level)
    return profile


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_find_wine_like(args: argparse.Namespace) -> None:
    """Find wines similar to a search term."""
    query = args.query
    k = args.top

    wines = _load_wines()
    x_embedded, df, _ = _build_embedding(wines)
    tree = KDTree(x_embedded)

    # search by name
    mask = df["wine_seo_name"].str.contains(query, case=False, na=False)
    matches = wines.loc[mask]

    if matches.empty:
        print(f'No wines found matching "{query}".')
        return

    if len(matches) > 1:
        print(f'Found {len(matches)} wines matching "{query}":\n')
        display_cols = ["id", "wine_seo_name", "winery_seo_name", "wine_type", "year", "ratings_average"]
        cols = [c for c in display_cols if c in matches.columns]
        print(matches[cols].to_string(index=False))
        print()

    # use the first match as the query wine
    query_idx = matches.index[0]
    query_point = x_embedded[query_idx].reshape(1, -1)
    query_wine = wines.iloc[query_idx]

    print(f'Wines similar to "{query_wine["wine_seo_name"]}" ({query_wine.get("winery_seo_name", "")}):\n')

    dist, ind = tree.query(query_point, k=k + 1)  # +1 because the wine itself is included
    result = wines.iloc[ind[0]].copy()
    result["distance"] = dist[0]
    result = result[result.index != query_idx].head(k)
    result["url"] = result.apply(
        lambda r: "https://www.google.com/search?q=" + "+".join(f"{r['wine_seo_name']} {r['winery_seo_name']}".split()),
        axis=1,
    )

    display_cols = [
        "wine_seo_name",
        "winery_seo_name",
        "wine_type",
        "year",
        "region_seo_name",
        "ratings_average",
        "distance",
        "url",
    ]
    cols = [c for c in display_cols if c in result.columns]
    print(result[cols].to_string(index=False))


def cmd_pair_wine_to(args: argparse.Namespace) -> None:
    """Find wines that pair well with a given food."""
    query = args.food
    k = args.top

    # load food list for fuzzy matching
    foods = _load_food_list()
    food_col = foods.columns[0]
    foods[food_col] = foods[food_col].str.strip()

    mask = foods[food_col].str.contains(query, case=False, na=False)
    matched_foods = foods.loc[mask, food_col].tolist()

    if not matched_foods:
        print(f'No food found matching "{query}". Try a different search term.')
        print(f"\nSome available foods: {', '.join(foods[food_col].head(20).tolist())}")
        return

    food_name = matched_foods[0]
    if len(matched_foods) > 1:
        print(f'Found {len(matched_foods)} foods matching "{query}":')
        for f in matched_foods[:15]:
            print(f"  - {f}")
        print(f'\nUsing: "{food_name}"\n')
    else:
        print(f'Pairing wines for: "{food_name}"\n')

    # build a food taste profile from descriptor mapping
    desc = _load_descriptor_tastes()
    food_lower = food_name.lower().replace(" ", "_")
    food_descriptors = desc[desc["raw descriptor"].str.contains(food_lower, case=False, na=False)]

    # derive taste profile from descriptors, or use defaults
    taste_profile: dict[str, tuple[float, int]] = {}
    if not food_descriptors.empty:
        nonaroma = food_descriptors[food_descriptors["type"] == "nonaroma"]
        for _, row in nonaroma.iterrows():
            taste = row.get("primary taste", "")
            if taste and taste in FOOD_WEIGHTS:
                level_str = row.get("level_2", "")
                intensity = 3 if "high" in str(level_str) else 2
                taste_profile[taste] = (0.5, intensity)

    # fill missing tastes with neutral defaults
    for taste in ("weight", "sweet", "acid", "salt", "piquant", "fat", "bitter"):
        if taste not in taste_profile:
            taste_profile[taste] = (0.3, 2)

    food_weight = taste_profile.get("weight", (0.5, 2))

    # load wines and build taste columns
    wines = _load_wines()
    wine_df = wines.copy()

    # map wine structure columns to the taste columns the rules expect
    col_map = {
        "weight": "style_body",
        "sweet": "structure_sweetness",
        "acid": "structure_acidity",
        "bitter": "structure_tannin",
        "salt": "structure_intensity",
        "piquant": "structure_fizziness",
        "fat": "style_body",
        "tannin": "structure_tannin",
    }
    for taste, source in col_map.items():
        if source in wine_df.columns:
            raw = wine_df[source].fillna(0).astype(float)
            wmap = WINE_WEIGHTS.get(taste, WINE_WEIGHTS["weight"])
            wine_df[taste] = raw.apply(lambda v, wm=wmap: _score_to_level(v, wm))
        else:
            wine_df[taste] = 2

    paired = nonaroma_rules(wine_df, taste_profile, food_weight)

    if paired.empty:
        print("No wines matched the pairing rules. Try a different food.")
        return

    # sort by rating
    if "ratings_average" in paired.columns:
        paired = paired.sort_values("ratings_average", ascending=False)

    display_cols = [
        "wine_seo_name",
        "winery_seo_name",
        "wine_type",
        "year",
        "region_seo_name",
        "ratings_average",
    ]
    cols = [c for c in display_cols if c in paired.columns]
    print(f'Top {k} wines that pair with "{food_name}":\n')
    print(paired[cols].head(k).to_string(index=False))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def main() -> None:
    logger.info("Data directory: %s", DATA_DIR)
    logger.info("Reference data directory: %s", REF_DATA_DIR)

    parser = argparse.ArgumentParser(
        prog="wine-buddy",
        description="Wine Buddy CLI — find similar wines and pair wines to food.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # find-wine-like
    p_find = sub.add_parser(
        "find-wine-like",
        help="Find wines similar to a given wine name",
    )
    p_find.add_argument("query", help="Wine name to search for (partial match)")
    p_find.add_argument("-n", "--top", type=int, default=5, help="Number of results (default: 5)")
    p_find.set_defaults(func=cmd_find_wine_like)

    # pair-wine-to
    p_pair = sub.add_parser(
        "pair-wine-to",
        help="Find wines that pair well with a given food",
    )
    p_pair.add_argument("food", help="Food name to search for (partial match)")
    p_pair.add_argument("-n", "--top", type=int, default=10, help="Number of results (default: 10)")
    p_pair.set_defaults(func=cmd_pair_wine_to)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
