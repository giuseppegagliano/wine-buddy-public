import io
import logging
import os
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT_PATH = Path(os.environ.get("WINE_BUDDY_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT_PATH / "data"

logger = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load .env file from project root if it exists (no extra dependency)."""
    env_file = ROOT_PATH / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _ensure_data() -> None:
    """Download and extract data zip from DATA_ZIP_URL if data dir is missing."""
    if DATA_DIR.exists() and (any(DATA_DIR.glob("*.parquet")) or any(DATA_DIR.glob("*.csv"))):
        return
    url = os.environ.get("GDRIVE_DATA_LINK")
    if not url:
        return
    # Convert Google Drive share link to direct download
    if "drive.google.com" in url:
        file_id = url.split("/d/")[1].split("/")[0]
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
    logger.info("Downloading data from GDRIVE_DATA_LINK...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(DATA_DIR)
    logger.info("Data extracted to %s", DATA_DIR)


_ensure_data()

# // ------------------------ GENERAL PARAMS ------------------------------
column_names = [
    "id",
    "year",
    "wine_id",
    "wine_seo_name",
    "winery_seo_name",
    "is_natural",
    "structure_acidity",
    "structure_fizziness",
    "structure_intensity",
    "structure_sweetness",
    "structure_tannin",
    "flavor",
    "ratings_count",
    "ratings_average",
    "style_body",
    "style_acidity",
    "style_food",
    "style_grapes",
    "region_seo_name",
    "region_country_code",
]
COUNTRY_CODE = "IT"
WINE_TYPES = {
    "1": "red",
    "2": "white",
    "3": "sparkling",
    "4": "pink",
    "7": "dessert",
    "24": "liquor",
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
    "weight": {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    "sweet": {1: (0, 0.45), 2: (0.45, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
    "acid": {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.7), 4: (0.7, 1)},
    "salt": {1: (0, 0.3), 2: (0.3, 0.55), 3: (0.55, 0.8), 4: (0.8, 1)},
    "piquant": {1: (0, 0.4), 2: (0.4, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
    "fat": {1: (0, 0.4), 2: (0.4, 0.5), 3: (0.5, 0.6), 4: (0.6, 1)},
    "bitter": {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.65), 4: (0.65, 1)},
}

WINE_WEIGHTS = {
    "weight": {1: (0, 0.25), 2: (0.25, 0.45), 3: (0.45, 0.75), 4: (0.75, 1)},
    "sweet": {1: (0, 0.25), 2: (0.25, 0.6), 3: (0.6, 0.75), 4: (0.75, 1)},
    "acid": {1: (0, 0.05), 2: (0.05, 0.25), 3: (0.25, 0.5), 4: (0.5, 1)},
    "salt": {1: (0, 0.15), 2: (0.15, 0.25), 3: (0.25, 0.7), 4: (0.7, 1)},
    "piquant": {1: (0, 0.15), 2: (0.15, 0.3), 3: (0.3, 0.6), 4: (0.6, 1)},
    "fat": {1: (0, 0.25), 2: (0.25, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
    "bitter": {1: (0, 0.2), 2: (0.2, 0.37), 3: (0.37, 0.6), 4: (0.6, 1)},
}
