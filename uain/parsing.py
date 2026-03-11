from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from config import COUNTRY_CODE, ROOT_PATH

logger = logging.getLogger(__name__)

DATA_DIR = Path(ROOT_PATH) / "data"

# Precompiled regex for speed and clarity.
_INTERNAL_APOSTROPHE_RE = re.compile(r"(?<=[A-Za-z])'(?=[A-Za-z])")


def _get_csv_path(color: str) -> Path:
    return DATA_DIR / f"{COUNTRY_CODE.lower()}_{color}.csv"


def _load_color_df(color: str) -> pd.DataFrame:
    path = _get_csv_path(color)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame, required_columns: set[str], context: str) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{context}: missing required columns: {sorted(missing)}")


def _normalize_json_like_string(
    value: Any,
    *,
    replace_python_literals: bool = False,
) -> str | None:
    """
    Normalize malformed JSON-like strings into something json.loads can parse.

    Handles common issues seen in scraped/exported payloads:
    - apostrophes inside words: don't -> dont
    - single quotes used as string delimiters
    - Python literals: True/False/None -> JSON literals
    """
    if not isinstance(value, str) or not value.strip():
        return None

    cleaned = _INTERNAL_APOSTROPHE_RE.sub("", value)
    cleaned = cleaned.replace("'", '"')

    if replace_python_literals:
        cleaned = re.sub(r"\bTrue\b", "true", cleaned)
        cleaned = re.sub(r"\bFalse\b", "false", cleaned)
        cleaned = re.sub(r"\bNone\b", "null", cleaned)

    return cleaned


def _safe_json_loads(
    value: Any,
    *,
    replace_python_literals: bool = False,
    context: str,
) -> list[dict[str, Any]]:
    """
    Parse a JSON-like string into a list of dicts.
    Returns an empty list on invalid input instead of mixing return types.
    """
    cleaned = _normalize_json_like_string(
        value,
        replace_python_literals=replace_python_literals,
    )
    if cleaned is None:
        return []

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to parse %s: %s | raw=%r", context, exc, value)
        return []

    if not isinstance(parsed, list):
        logger.warning("Expected list for %s, got %s | raw=%r", context, type(parsed).__name__, value)
        return []

    return [item for item in parsed if isinstance(item, dict)]


def _explode_records(
    df: pd.DataFrame,
    *,
    source_column: str,
    output_column: str,
    parser: Callable[[Any], list[dict[str, Any]]],
    columns_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse a nested column into a list of records, explode it, and flatten it.
    """
    result = df.copy()
    result[output_column] = result[source_column].apply(parser)
    result = result.explode(output_column)

    result = result[result[output_column].notna()]
    result = result[result[output_column].map(lambda x: isinstance(x, dict) and len(x) > 0)]

    expanded = result[output_column].apply(pd.Series)
    result = pd.concat([result.drop(columns=[output_column] + (columns_to_drop or [])), expanded], axis=1)

    return result.drop_duplicates()


def _parse_flavor_records(value: Any) -> list[dict[str, Any]]:
    parsed = _safe_json_loads(value, context="flavor")
    records: list[dict[str, Any]] = []

    for item in parsed:
        group = item.get("group")
        stats = item.get("stats", {})
        if not isinstance(stats, dict):
            continue

        records.append(
            {
                "flavor": group,
                "flavor_count": stats.get("count"),
                "flavor_score": stats.get("score"),
            }
        )

    return records


def _parse_grape_records(value: Any) -> list[dict[str, Any]]:
    parsed = _safe_json_loads(value, replace_python_literals=True, context="style_grapes")
    return [{"grapes": item.get("seo_name")} for item in parsed if item.get("seo_name") is not None]


def _parse_food_records(value: Any) -> list[dict[str, Any]]:
    parsed = _safe_json_loads(value, replace_python_literals=True, context="style_food")
    return [{"style_food": item.get("seo_name")} for item in parsed if item.get("seo_name") is not None]


def _parse_flavour_tuples(value: Any) -> list[dict[str, Any]]:
    parsed = _safe_json_loads(value, replace_python_literals=True, context="flavor")
    records: list[dict[str, Any]] = []

    for item in parsed:
        stats = item.get("stats", {})
        if not isinstance(stats, dict):
            continue

        flavor_name = item.get("group")
        flavor_score = stats.get("score")

        if flavor_name is not None:
            records.append(
                {
                    "flavor_name": flavor_name,
                    "flavor_score": flavor_score,
                }
            )

    return records


def get_flavors(color: str = "red") -> pd.DataFrame:
    """
    Load wine flavor scores for the requested color and return a pivoted table:
    one row per id, one column per flavor, values = flavor_score.
    """
    df = _load_color_df(color)
    _validate_columns(df, {"id", "flavor"}, context="get_flavors")

    flattened = _explode_records(
        df,
        source_column="flavor",
        output_column="parsed_flavor",
        parser=_parse_flavor_records,
    )

    flattened = flattened[flattened["flavor"].notna()]

    return (
        pd.pivot_table(
            flattened,
            index="id",
            columns="flavor",
            values="flavor_score",
            aggfunc="first",
        )
        .fillna(0.0)
        .reset_index()
        .rename_axis(None, axis=1)
    )


def get_grapes(color: str = "red") -> pd.DataFrame:
    """
    Load grape indicators for the requested color and return a one-hot encoded table:
    one row per id, one column per grape.
    """
    df = _load_color_df(color)
    _validate_columns(df, {"id", "style_grapes"}, context="get_grapes")

    flattened = _explode_records(
        df,
        source_column="style_grapes",
        output_column="parsed_grapes",
        parser=_parse_grape_records,
        columns_to_drop=["style_grapes"],
    )

    flattened = flattened[flattened["grapes"].notna()]

    return (
        pd.get_dummies(flattened[["id", "grapes"]], columns=["grapes"])
        .groupby("id", as_index=False)
        .sum()
        .rename_axis(None, axis=1)
    )


def get_food(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a one-hot encoded food pairing table from a dataframe containing:
    - id
    - style_food
    """
    _validate_columns(df, {"id", "style_food"}, context="get_food")

    flattened = _explode_records(
        df,
        source_column="style_food",
        output_column="parsed_food",
        parser=_parse_food_records,
        columns_to_drop=["style_food"],
    )

    flattened = flattened[flattened["style_food"].notna()]

    return (
        pd.get_dummies(
            flattened[["id", "style_food"]],
            columns=["style_food"],
            prefix="",
            prefix_sep="",
        )
        .groupby("id", as_index=False)
        .sum()
        .rename_axis(None, axis=1)
    )


def get_flavour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a pivoted flavor-score table from a dataframe containing:
    - id
    - flavor

    Note:
        Function name kept as `get_flavour` for backward compatibility.
        Internally everything uses `flavor`.
    """
    _validate_columns(df, {"id", "flavor"}, context="get_flavour")

    flattened = _explode_records(
        df,
        source_column="flavor",
        output_column="parsed_flavor_desc",
        parser=_parse_flavour_tuples,
    )

    return (
        pd.pivot_table(
            data=flattened,
            values="flavor_score",
            index="id",
            columns="flavor_name",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
        .fillna(0.0)
    )