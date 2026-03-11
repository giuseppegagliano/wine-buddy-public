# // ------------------------ GENERAL PARAMS ------------------------------
column_names = [
    "id", "year", "wine_id", "wine_seo_name", "winery_seo_name",
    "is_natural", "structure_acidity", "structure_fizziness",
    "structure_intensity", "structure_sweetness", "structure_tannin",
    "flavor", "ratings_count", "ratings_average",
    "style_body", "style_acidity", "style_food", "style_grapes",
    "region_seo_name", "region_country_code",
]
COUNTRY_CODE = "IT"
WINE_TYPES = {
    "1": "red", "2": "white", "3": "sparkling",
    "4": "pink", "7": "dessert", "24": "liquor",
}

# ------------------------ FOOD PAIRING PARAMS ------------------------------
TASTE_LEVEL_THRESHOLD = 4

# food taste -> wine tastes that create a contrasting pairing
CONTRASTING_RULES: dict[str, tuple[str, ...]] = {
    "sweet": ("bitter", "fat", "piquant", "salt", "acid"),
    "acid": ("sweet", "fat", "salt"),
    "salt": ("bitter", "sweet", "piquant", "fat", "acid"),
    "piquant": ("sweet", "fat", "salt"),
    "fat": ("bitter", "sweet", "piquant", "salt", "acid"),
    "bitter": ("sweet", "fat", "salt"),
}

MIN_CANDIDATE_WINES = 5
LOW_THRESHOLD = 2
MEDIUM_THRESHOLD = 3
HIGH_THRESHOLD = 4

REQUIRED_TASTE_COLUMNS = {
    "weight",
    "acid",
    "sweet",
    "bitter",
    "salt",
    "piquant",
    "fat",
    "tannin",
}

FOOD_WEIGHTS = {
    "weight":  {1: (0, 0.3),  2: (0.3, 0.5),  3: (0.5, 0.7),  4: (0.7, 1)},
    "sweet":   {1: (0, 0.45), 2: (0.45, 0.6), 3: (0.6, 0.8),  4: (0.8, 1)},
    "acid":    {1: (0, 0.4),  2: (0.4, 0.55), 3: (0.55, 0.7), 4: (0.7, 1)},
    "salt":    {1: (0, 0.3),  2: (0.3, 0.55), 3: (0.55, 0.8), 4: (0.8, 1)},
    "piquant": {1: (0, 0.4),  2: (0.4, 0.6),  3: (0.6, 0.8),  4: (0.8, 1)},
    "fat":     {1: (0, 0.4),  2: (0.4, 0.5),  3: (0.5, 0.6),  4: (0.6, 1)},
    "bitter":  {1: (0, 0.3),  2: (0.3, 0.5),  3: (0.5, 0.65), 4: (0.65, 1)},
}

WINE_WEIGHTS = {
    "weight":  {1: (0, 0.25), 2: (0.25, 0.45), 3: (0.45, 0.75), 4: (0.75, 1)},
    "sweet":   {1: (0, 0.25), 2: (0.25, 0.6),  3: (0.6, 0.75),  4: (0.75, 1)},
    "acid":    {1: (0, 0.05), 2: (0.05, 0.25), 3: (0.25, 0.5),  4: (0.5, 1)},
    "salt":    {1: (0, 0.15), 2: (0.15, 0.25), 3: (0.25, 0.7),  4: (0.7, 1)},
    "piquant": {1: (0, 0.15), 2: (0.15, 0.3),  3: (0.3, 0.6),   4: (0.6, 1)},
    "fat":     {1: (0, 0.25), 2: (0.25, 0.5),  3: (0.5, 0.7),   4: (0.7, 1)},
    "bitter":  {1: (0, 0.2),  2: (0.2, 0.37),  3: (0.37, 0.6),  4: (0.6, 1)},
}