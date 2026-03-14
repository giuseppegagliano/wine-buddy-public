"""Convert raw JSON scraper dumps into a single unified Parquet file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from uain.config import DATA_DIR

logger = logging.getLogger(__name__)


def _safe(d: dict | None, *keys: str) -> Any:
    """Traverse nested dicts safely, returning None on missing keys."""
    obj: Any = d
    for k in keys:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(k)
    return obj


def _names_list(items: list[dict] | None) -> list[str]:
    """Extract 'name' from a list of dicts, filtering None."""
    if not items:
        return []
    return [x["name"] for x in items if isinstance(x, dict) and "name" in x]


def flatten_match(match: dict[str, Any], *, wine_type: str) -> dict[str, Any]:
    """Flatten one explore API match into a single row dict."""
    v = match.get("vintage", {}) or {}
    w = v.get("wine", {}) or {}
    taste = w.get("taste", {}) or {}
    structure = taste.get("structure", {}) or {}
    style = w.get("style", {}) or {}
    region = w.get("region", {}) or {}
    country = region.get("country", {}) or {}
    winery = w.get("winery", {}) or {}
    v_stats = v.get("statistics", {}) or {}
    price = match.get("price", {}) or {}

    # food/grapes/flavor as JSON-encoded lists of names
    food_items = style.get("food") or []
    grape_items = style.get("grapes") or []
    flavor_items = taste.get("flavor") or []

    food_seo = [x.get("seo_name") for x in food_items if isinstance(x, dict) and x.get("seo_name")] if food_items else []
    grape_seo = [x.get("seo_name") for x in grape_items if isinstance(x, dict) and x.get("seo_name")] if grape_items else []

    # flavor: keep group + stats for downstream scoring
    flavor_records = []
    for f in flavor_items:
        if not isinstance(f, dict):
            continue
        rec: dict[str, Any] = {"group": f.get("group")}
        stats = f.get("stats")
        if isinstance(stats, dict):
            rec["stats"] = {"count": stats.get("count"), "score": stats.get("score")}
        flavor_records.append(rec)

    return {
        # identifiers
        "vintage_id": v.get("id"),
        "vintage_name": v.get("name"),
        "vintage_seo_name": v.get("seo_name"),
        "year": v.get("year"),
        "wine_id": w.get("id"),
        "wine_name": w.get("name"),
        "wine_seo_name": w.get("seo_name"),
        "wine_type_id": w.get("type_id"),
        "wine_type": wine_type,
        "is_natural": w.get("is_natural"),
        # winery
        "winery_id": winery.get("id"),
        "winery_name": winery.get("name"),
        "winery_seo_name": winery.get("seo_name"),
        # region / country
        "region_id": region.get("id"),
        "region_name": region.get("name"),
        "region_name_en": region.get("name_en"),
        "region_seo_name": region.get("seo_name"),
        "country_code": country.get("code"),
        "country_name": country.get("native_name"),
        # vintage-level stats
        "ratings_count": v_stats.get("ratings_count"),
        "ratings_average": v_stats.get("ratings_average"),
        "wine_ratings_count": v_stats.get("wine_ratings_count"),
        "wine_ratings_average": v_stats.get("wine_ratings_average"),
        "labels_count": v_stats.get("labels_count"),
        # taste structure
        "structure_acidity": structure.get("acidity"),
        "structure_fizziness": structure.get("fizziness"),
        "structure_intensity": structure.get("intensity"),
        "structure_sweetness": structure.get("sweetness"),
        "structure_tannin": structure.get("tannin"),
        "structure_user_count": structure.get("user_structure_count"),
        "structure_calc_count": structure.get("calculated_structure_count"),
        # style
        "style_id": style.get("id"),
        "style_name": style.get("name"),
        "style_seo_name": style.get("seo_name"),
        "style_body": style.get("body"),
        "style_body_description": style.get("body_description"),
        "style_acidity": style.get("acidity"),
        "style_acidity_description": style.get("acidity_description"),
        # lists as JSON strings
        "style_food": json.dumps(food_seo, ensure_ascii=False) if food_seo else None,
        "style_grapes": json.dumps(grape_seo, ensure_ascii=False) if grape_seo else None,
        "flavor": json.dumps(flavor_records, ensure_ascii=False) if flavor_records else None,
        # price
        "price_amount": price.get("amount"),
        "price_discounted_from": price.get("discounted_from"),
        "price_discount_percent": price.get("discount_percent"),
        "price_currency": _safe(price, "currency", "code"),
        "price_bottle_volume_ml": _safe(price, "bottle_type", "volume_ml"),
    }


def load_all_raw(data_dir: Path = DATA_DIR / "raw") -> pd.DataFrame:
    """Load every *_raw.json in data_dir and return a single DataFrame."""
    from uain.config import WINE_TYPES

    type_id_to_name = {int(k): v for k, v in WINE_TYPES.items()}

    raw_files = sorted(data_dir.glob("*_raw.json"))
    if not raw_files:
        raise FileNotFoundError(f"No *_raw.json files found in {data_dir}")

    rows: list[dict[str, Any]] = []
    for path in raw_files:
        # parse wine_type from filename: e.g. fr_red_raw.json -> red
        stem = path.stem  # fr_red_raw
        parts = stem.rsplit("_raw", 1)[0].split("_", 1)  # ['fr', 'red']
        wine_type = parts[1] if len(parts) > 1 else "unknown"

        with open(path, encoding="utf-8") as f:
            matches = json.load(f)

        logger.info("Processing %s: %s matches", path.name, len(matches))
        for match in matches:
            rows.append(flatten_match(match, wine_type=wine_type))

    df = pd.DataFrame(rows)
    # coerce year to numeric (handles "N.V." etc.)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    # deduplicate by vintage_id, keep first occurrence
    df = df.drop_duplicates(subset="vintage_id", keep="first")
    return df


def convert_raw_to_parquet(
    data_dir: Path = DATA_DIR / "raw",
    output_path: Path = DATA_DIR / "wines.parquet",
) -> Path:
    """Load all raw JSON, flatten, deduplicate, and save as Parquet."""
    df = load_all_raw(data_dir)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info("Saved %s wines to %s (%.1f MB)", len(df), output_path, output_path.stat().st_size / 1e6)
    return output_path
