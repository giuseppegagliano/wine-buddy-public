from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from uain.config import CONTRASTING_RULES, TASTE_LEVEL_THRESHOLD


def _validate_food_nonaromas(food_nonaromas: Mapping[str, tuple[object, int]]) -> None:
    required = set(CONTRASTING_RULES.keys())
    missing = required - set(food_nonaromas.keys())
    if missing:
        raise ValueError(f"Missing food non-aroma tastes: {sorted(missing)}")

    invalid = [key for key, value in food_nonaromas.items() if not isinstance(value, tuple) or len(value) < 2]
    if invalid:
        raise ValueError(f"food_nonaromas values must be tuples with score at index 1. Invalid keys: {sorted(invalid)}")


def _validate_dataframe_columns(df: pd.DataFrame) -> None:
    required = set(CONTRASTING_RULES.keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required wine taste columns: {sorted(missing)}")


def _mark_congruent_pairings(
    df: pd.DataFrame,
    food_nonaromas: Mapping[str, tuple[object, int]],
) -> pd.Series:
    """
    Mark wines as congruent when they match or exceed at least one
    of the most defining food taste intensities.
    """
    max_food_score = max(score for _, score in food_nonaromas.values())
    defining_tastes = [taste for taste, (_, score) in food_nonaromas.items() if score == max_food_score]

    congruent_mask = pd.Series(False, index=df.index)
    for taste in defining_tastes:
        congruent_mask |= df[taste] >= max_food_score

    return pd.Series(
        np.where(congruent_mask, "congruent", ""),
        index=df.index,
        dtype="object",
    )


def _apply_contrasting_rules(
    df: pd.DataFrame,
    pairing_type: pd.Series,
    food_nonaromas: Mapping[str, tuple[object, int]],
) -> pd.Series:
    """
    Override pairing_type to 'contrasting' when the food has a strong
    defining taste and the wine contains one of the contrasting tastes
    at the threshold level.
    """
    result = pairing_type.copy()

    for food_taste, contrasting_wine_tastes in CONTRASTING_RULES.items():
        food_score = food_nonaromas[food_taste][1]
        if food_score != TASTE_LEVEL_THRESHOLD:
            continue

        contrast_mask = (df.loc[:, list(contrasting_wine_tastes)] == TASTE_LEVEL_THRESHOLD).any(axis=1)
        result = pd.Series(
            np.where(contrast_mask, "contrasting", result),
            index=df.index,
            dtype="object",
        )

    return result


def congruent_or_contrasting(
    df: pd.DataFrame,
    food_nonaromas: Mapping[str, tuple[object, int]],
) -> pd.DataFrame:
    """
    Classify each wine-food pair as congruent, contrasting, or neither.

    Args:
        df:
            DataFrame containing wine taste intensity columns:
            sweet, acid, salt, piquant, fat, bitter
        food_nonaromas:
            Mapping from taste name to a tuple where the taste intensity
            is stored at index 1.

    Returns:
        A copy of the input DataFrame with a new `pairing_type` column.
    """
    _validate_dataframe_columns(df)
    _validate_food_nonaromas(food_nonaromas)

    result = df.copy()

    pairing_type = _mark_congruent_pairings(result, food_nonaromas)
    pairing_type = _apply_contrasting_rules(result, pairing_type, food_nonaromas)

    result["pairing_type"] = pairing_type
    return result
