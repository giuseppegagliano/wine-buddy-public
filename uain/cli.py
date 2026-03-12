"""Wine Buddy CLI — find similar wines and pair wines to food."""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from uain.config import DATA_DIR, FOOD_WEIGHTS
from uain.rules import nonaroma_rules
from uain.services import get_wine_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_find_wine_like(args: argparse.Namespace) -> None:
    """Find wines similar to a search term."""
    query = args.query
    k = args.top

    idx = get_wine_index()
    mask = idx.wines["wine_seo_name"].str.contains(query, case=False, na=False)
    matches = idx.wines.loc[mask]

    if matches.empty:
        print(f'No wines found matching "{query}".')
        return

    if len(matches) > 1:
        print(f'Found {len(matches)} wines matching "{query}":\n')
        display_cols = ["id", "wine_seo_name", "winery_seo_name", "wine_type", "year", "ratings_average"]
        cols = [c for c in display_cols if c in matches.columns]
        print(matches[cols].to_string(index=False))
        print()

    query_pos = matches.index[0]
    query_point = idx.embeddings[query_pos]
    query_wine = idx.wines.iloc[query_pos]

    print(f'Wines similar to "{query_wine["wine_seo_name"]}" ({query_wine.get("winery_seo_name", "")}):\n')

    dists = np.linalg.norm(idx.embeddings - query_point, axis=1)
    dists[query_pos] = np.inf
    top_indices = np.argsort(dists)[:k]

    result = idx.wines.iloc[top_indices].copy()
    result["distance"] = dists[top_indices]
    result["url"] = result.apply(
        lambda r: "https://www.google.com/search?q="
        + "+".join(f"{r['wine_seo_name']} {r['winery_seo_name']}".split()),
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

    foods = pd.read_csv(DATA_DIR / "list_of_foods.csv")
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

    # Build food taste profile from descriptor mapping
    desc = pd.read_csv(DATA_DIR / "descriptor_mapping_tastes.csv")
    food_lower = food_name.lower().replace(" ", "_")
    food_descriptors = desc[desc["raw descriptor"].str.contains(food_lower, case=False, na=False)]

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

    food_weight = taste_profile.get("weight", (0.5, 2))

    # Taste levels are already precomputed in the parquet
    idx = get_wine_index()
    wine_df = idx.wines.copy()

    paired = nonaroma_rules(wine_df, taste_profile, food_weight)

    if paired.empty:
        print("No wines matched the pairing rules. Try a different food.")
        return

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

    parser = argparse.ArgumentParser(
        prog="wine-buddy",
        description="Wine Buddy CLI — find similar wines and pair wines to food.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser(
        "find-wine-like",
        help="Find wines similar to a given wine name",
    )
    p_find.add_argument("query", help="Wine name to search for (partial match)")
    p_find.add_argument("-n", "--top", type=int, default=5, help="Number of results (default: 5)")
    p_find.set_defaults(func=cmd_find_wine_like)

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
