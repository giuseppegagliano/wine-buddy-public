from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from uain.config import (
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    MEDIUM_THRESHOLD,
    MIN_CANDIDATE_WINES,
    REQUIRED_TASTE_COLUMNS,
)


@dataclass(frozen=True)
class RuleResult:
    name: str
    dataframe: pd.DataFrame


def _score(profile: Mapping[str, tuple[object, int]], key: str) -> int:
    value = profile.get(key)
    if not isinstance(value, tuple) or len(value) < 2:
        raise ValueError(f"Invalid profile entry for '{key}'. Expected tuple-like value with score at index 1.")

    score = value[1]
    if not isinstance(score, int):
        raise ValueError(f"Invalid score for '{key}': expected int, got {type(score).__name__}")

    return score


def _validate_wine_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_TASTE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required wine columns: {sorted(missing)}")


def _validate_food_profile(food_nonaromas: Mapping[str, tuple[object, int]]) -> None:
    required = {"acid", "sweet", "bitter", "salt", "piquant", "fat"}
    missing = required - set(food_nonaromas.keys())
    if missing:
        raise ValueError(f"Missing required food non-aroma keys: {sorted(missing)}")


def _validate_food_weight(food_weight: tuple[object, int]) -> None:
    if not isinstance(food_weight, tuple) or len(food_weight) < 2 or not isinstance(food_weight[1], int):
        raise ValueError("food_weight must be a tuple-like value with integer score at index 1")


def weight_rule(df: pd.DataFrame, food_weight: tuple[object, int]) -> pd.DataFrame:
    """Wine body should match or exceed the food's weight by at most one level."""
    weight_score = food_weight[1]
    return df.loc[(df["weight"] >= weight_score) & (df["weight"] <= weight_score + 1)]


def acidity_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """The wine should be at least as acidic as the food."""
    return df.loc[df["acid"] >= _score(food_nonaromas, "acid")]


def sweetness_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """The wine should be at least as sweet as the food."""
    return df.loc[df["sweet"] >= _score(food_nonaromas, "sweet")]


def bitterness_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """Bitter wines do not pair well with very bitter foods."""
    if _score(food_nonaromas, "bitter") == HIGH_THRESHOLD:
        return df.loc[df["bitter"] <= LOW_THRESHOLD]
    return df


def bitter_salt_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """Bitter and salt do not go well together."""
    result = df

    if _score(food_nonaromas, "bitter") == HIGH_THRESHOLD:
        result = result.loc[result["salt"] <= LOW_THRESHOLD]

    if _score(food_nonaromas, "salt") == HIGH_THRESHOLD:
        result = result.loc[result["bitter"] <= LOW_THRESHOLD]

    return result


def acid_bitter_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """Acid and bitterness do not go well together."""
    result = df

    if _score(food_nonaromas, "acid") == HIGH_THRESHOLD:
        result = result.loc[result["bitter"] <= LOW_THRESHOLD]

    if _score(food_nonaromas, "bitter") == HIGH_THRESHOLD:
        result = result.loc[result["acid"] <= LOW_THRESHOLD]

    return result


def acid_piquant_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """Acid and piquant do not go well together."""
    result = df

    if _score(food_nonaromas, "acid") == HIGH_THRESHOLD:
        result = result.loc[result["piquant"] <= LOW_THRESHOLD]

    if _score(food_nonaromas, "piquant") == HIGH_THRESHOLD:
        result = result.loc[result["acid"] <= LOW_THRESHOLD]

    return result


def tannin_fat_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """High-fat foods prefer tannic wines."""
    if _score(food_nonaromas, "fat") >= MEDIUM_THRESHOLD:
        return df.loc[df["tannin"] >= MEDIUM_THRESHOLD]
    return df


def tannin_bitter_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """Avoid tannic wines with bitter foods."""
    if _score(food_nonaromas, "bitter") >= MEDIUM_THRESHOLD:
        return df.loc[df["tannin"] <= LOW_THRESHOLD]
    return df


def tannin_piquant_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """Avoid tannic wines with spicy foods."""
    if _score(food_nonaromas, "piquant") >= MEDIUM_THRESHOLD:
        return df.loc[df["tannin"] <= LOW_THRESHOLD]
    return df


def tannin_salt_rule(df: pd.DataFrame, food_nonaromas: Mapping[str, tuple[object, int]]) -> pd.DataFrame:
    """Avoid tannic wines with very salty foods."""
    if _score(food_nonaromas, "salt") == HIGH_THRESHOLD:
        return df.loc[df["tannin"] <= LOW_THRESHOLD]
    return df


_NONAROMA_RULES: tuple[
    Callable[[pd.DataFrame, Mapping[str, tuple[object, int]]], pd.DataFrame],
    ...,
] = (
    acidity_rule,
    sweetness_rule,
    bitterness_rule,
    bitter_salt_rule,
    acid_bitter_rule,
    acid_piquant_rule,
    tannin_fat_rule,
    tannin_bitter_rule,
    tannin_piquant_rule,
    tannin_salt_rule,
)


def _apply_rule_if_not_too_aggressive(
    df: pd.DataFrame,
    rule: Callable[[pd.DataFrame, Mapping[str, tuple[object, int]]], pd.DataFrame],
    food_nonaromas: Mapping[str, tuple[object, int]],
    *,
    min_candidate_wines: int,
) -> pd.DataFrame:
    """
    Apply a rule only if it leaves more than `min_candidate_wines` rows.
    """
    candidate_df = rule(df, food_nonaromas)
    if candidate_df.shape[0] > min_candidate_wines:
        return candidate_df
    return df


def nonaroma_rules(
    wine_df: pd.DataFrame,
    food_nonaromas: Mapping[str, tuple[object, int]],
    food_weight: tuple[object, int],
    *,
    min_candidate_wines: int = MIN_CANDIDATE_WINES,
) -> pd.DataFrame:
    """
    Apply non-aroma pairing rules sequentially.

    Rules are kept only when they leave more than `min_candidate_wines`
    candidate wines, preventing overly aggressive filtering.
    """
    _validate_wine_df(wine_df)
    _validate_food_profile(food_nonaromas)
    _validate_food_weight(food_weight)

    df = weight_rule(wine_df.copy(), food_weight)

    for rule in _NONAROMA_RULES:
        df = _apply_rule_if_not_too_aggressive(
            df,
            rule,
            food_nonaromas,
            min_candidate_wines=min_candidate_wines,
        )

    return df
