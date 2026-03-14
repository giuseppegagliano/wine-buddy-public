from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import json

import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from uain.config import COUNTRY_CODE, WINE_TYPES, column_names

logger = logging.getLogger(__name__)

BASE_URL = "https://www.vivino.com/api"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) Gecko/20100101 Firefox/136.0"
DEFAULT_TIMEOUT = 20
MAX_EXPLORE_PAGES = 100
REVIEWS_PER_PAGE = 50

DATA_DIR = Path("data")
RATINGS_DIR = DATA_DIR / "ratings"


class VivinoClient:
    """HTTP client wrapper for Vivino API calls."""

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        user_agent: str = USER_AGENT,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.session = self._build_session(
            user_agent=user_agent,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )

    @staticmethod
    def _build_session(
        *,
        user_agent: str,
        max_retries: int,
        backoff_factor: float,
    ) -> Session:
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})

        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.timeout)
        self._raise_for_bad_response(response, url=url)
        return self._parse_json(response=response, url=url)

    @staticmethod
    def _raise_for_bad_response(response: Response, *, url: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(f"Request failed for {url} with status {response.status_code}") from exc

    @staticmethod
    def _parse_json(*, response: Response, url: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(f"Invalid JSON returned from {url}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected payload type from {url}: {type(payload).__name__}")

        return payload

    def get_wine_reviews(self, wine_id: int | str, year: int | str, page: int) -> dict[str, Any]:
        url = f"{BASE_URL}/wines/{wine_id}/reviews"
        params = {
            "per_page": REVIEWS_PER_PAGE,
            "year": year,
            "page": page,
        }
        return self._get_json(url, params=params)

    def get_explore_page(
        self,
        *,
        country_code: str,
        wine_type_id: int | str,
        page: int,
        min_price: int | float,
        max_price: int | float,
    ) -> dict[str, Any]:
        url = f"{BASE_URL}/explore/explore"
        params = {
            "country_code": country_code,
            "country_codes[]": country_code.lower(),
            "currency_code": "EUR",
            "grape_filter": "varietal",
            "min_rating": "1",
            "order_by": "price",
            "order": "asc",
            "page": page,
            "price_range_max": str(max_price),
            "price_range_min": str(min_price),
            "wine_type_ids[]": wine_type_id,
        }
        return self._get_json(url, params=params)


def _extract_wine_record(match: dict[str, Any]) -> tuple[Any, ...] | None:
    """
    Extract the flattened wine record from one explore API match.
    Returns None when required nested fields are missing.
    """
    try:
        vintage = match["vintage"]
        wine = vintage["wine"]
        taste = wine["taste"]
        structure = taste["structure"]
        statistics = wine["statistics"]
        style = wine["style"]
        region = wine["region"]
        country = region["country"]

        return (
            vintage["id"],
            vintage["year"],
            wine["id"],
            wine["seo_name"],
            wine["winery"]["seo_name"],
            wine["is_natural"],
            structure["acidity"],
            structure["fizziness"],
            structure["intensity"],
            structure["sweetness"],
            structure["tannin"],
            taste["flavor"],
            statistics["ratings_count"],
            statistics["ratings_average"],
            style["body"],
            style["acidity"],
            style["food"],
            style["grapes"],
            region["seo_name"],
            country["code"],
        )
    except (KeyError, TypeError) as exc:
        logger.debug("Skipping malformed wine match: %s | payload=%r", exc, match)
        return None


def get_wine_data(
    wine_id: int | str,
    year: int | str,
    page: int,
    *,
    client: VivinoClient | None = None,
) -> dict[str, Any]:
    """
    Backward-compatible wrapper around the reviews endpoint.
    """
    api_client = client or VivinoClient()
    return api_client.get_wine_reviews(wine_id=wine_id, year=year, page=page)


def scrape_wines(
    country_code: str,
    wine_type_id: int | str,
    wine_type_name: str,
    min_price: int | float = 5,
    max_price: int | float = 50,
    *,
    client: VivinoClient | None = None,
    max_pages: int = MAX_EXPLORE_PAGES,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    """
    Scrape wine listings from the Vivino explore API.

    Returns one row per unique vintage id.
    """
    api_client = client or VivinoClient()

    results: list[tuple[Any, ...]] = []
    distinct_ids: set[Any] = set()
    previous_distinct_count = 0

    for page in range(1, max_pages + 1):
        payload = api_client.get_explore_page(
            country_code=country_code,
            wine_type_id=wine_type_id,
            page=page,
            min_price=min_price,
            max_price=max_price,
        )

        matches = payload.get("explore_vintage", {}).get("matches", [])
        if not matches:
            logger.info("No more matches for %s after page %s", wine_type_name, page)
            break

        if page % 25 == 0:
            logger.info("Page %s - %s wines retrieved for %s", page, len(matches), wine_type_name)

        for match in matches:
            record = _extract_wine_record(match)
            if record is None:
                continue

            results.append(record)
            distinct_ids.add(record[0])  # vintage id

        if len(distinct_ids) <= previous_distinct_count:
            logger.info("Reached apparent maximum number of distinct wines for %s", wine_type_name)
            break

        previous_distinct_count = len(distinct_ids)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    df = pd.DataFrame(results, columns=column_names)
    df["wine_type"] = wine_type_name

    return df.groupby("id", as_index=False).first()


def _extract_review_record(review: dict[str, Any], *, wine_id: Any, year: Any) -> list[Any]:
    stats = review.get("statistics", {}) or {}
    user = review.get("user", {}) or {}

    return [
        year,
        wine_id,
        user.get("id"),
        stats.get("followers_count"),
        stats.get("followings_count"),
        stats.get("ratings_count"),
        stats.get("ratings_sum"),
        stats.get("reviews_count"),
        stats.get("purchase_order_count"),
        review.get("rating"),
        review.get("note"),
        review.get("created_at"),
    ]


def scrape_ratings(
    wine_df: pd.DataFrame,
    *,
    client: VivinoClient | None = None,
    output_dir: Path = RATINGS_DIR,
    sleep_seconds: float = 0.0,
) -> None:
    """
    Scrape individual wine ratings and save one CSV per wine_id.

    Expected columns in wine_df:
    - wine_id
    - year
    """
    required_columns = {"wine_id", "year"}
    missing = required_columns - set(wine_df.columns)
    if missing:
        raise ValueError(f"scrape_ratings: missing required columns: {sorted(missing)}")

    api_client = client or VivinoClient()
    output_dir.mkdir(parents=True, exist_ok=True)

    ratings_columns = [
        "year",
        "wine_id",
        "customer_id",
        "followers_count",
        "followings_count",
        "ratings_count",
        "ratings_sum",
        "reviews_count",
        "purchase_order_count",
        "rating",
        "note",
        "created_at",
    ]

    for row in wine_df.itertuples(index=False):
        wine_id = row.wine_id
        year = row.year

        logger.info("Getting ratings for wine_id=%s year=%s", wine_id, year)

        ratings: list[list[Any]] = []
        page = 1

        while True:
            payload = api_client.get_wine_reviews(wine_id=wine_id, year=year, page=page)
            reviews = payload.get("reviews", [])

            if not reviews:
                break

            for review in reviews:
                ratings.append(_extract_review_record(review, wine_id=wine_id, year=year))

            page += 1

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        ratings_df = pd.DataFrame(ratings, columns=ratings_columns)
        output_path = output_dir / f"{COUNTRY_CODE.lower()}_{wine_id}.csv"
        ratings_df.to_csv(output_path, index=False)


def scrape_wines_raw(
    country_code: str,
    wine_type_id: int | str,
    wine_type_name: str,
    min_price: int | float = 5,
    max_price: int | float = 50,
    *,
    client: VivinoClient | None = None,
    max_pages: int = MAX_EXPLORE_PAGES,
    sleep_seconds: float = 0.0,
    output_path: Path | None = None,
    checkpoint_every: int = 10,
) -> list[dict[str, Any]]:
    """
    Scrape wine listings and return raw JSON matches (no parsing).

    When *output_path* is provided, intermediate results are flushed to
    disk every *checkpoint_every* pages so progress survives crashes.
    """
    api_client = client or VivinoClient()

    all_matches: list[dict[str, Any]] = []
    seen_ids: set[Any] = set()
    previous_seen_count = 0

    def _checkpoint() -> None:
        if output_path is not None:
            output_path.write_text(
                json.dumps(all_matches, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Checkpoint: %s matches saved to %s", len(all_matches), output_path)

    for page in range(1, max_pages + 1):
        payload = api_client.get_explore_page(
            country_code=country_code,
            wine_type_id=wine_type_id,
            page=page,
            min_price=min_price,
            max_price=max_price,
        )

        matches = payload.get("explore_vintage", {}).get("matches", [])
        if not matches:
            logger.info("No more matches for %s after page %s", wine_type_name, page)
            break

        if page % 25 == 0:
            logger.info("Page %s - %s wines retrieved for %s", page, len(matches), wine_type_name)

        for match in matches:
            vintage_id = match.get("vintage", {}).get("id")
            all_matches.append(match)
            if vintage_id is not None:
                seen_ids.add(vintage_id)

        if page % checkpoint_every == 0:
            _checkpoint()

        if len(seen_ids) <= previous_seen_count:
            logger.info("Reached apparent maximum number of distinct wines for %s", wine_type_name)
            break

        previous_seen_count = len(seen_ids)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    _checkpoint()
    return all_matches


def scrape_all_raw(
    *,
    country_code: str = COUNTRY_CODE,
    wine_types: dict[Any, str] | None = None,
    output_dir: Path = DATA_DIR,
    client: VivinoClient | None = None,
    sleep_seconds: float = 0.0,
) -> None:
    """
    Scrape all wine types and dump raw JSON per type.
    """
    if wine_types is None:
        wine_types = WINE_TYPES
    output_dir.mkdir(parents=True, exist_ok=True)
    api_client = client or VivinoClient()

    for type_id, type_name in wine_types.items():
        logger.info("Scraping %s wines (raw)...", type_name)
        output_path = output_dir / f"{country_code.lower()}_{type_name}_raw.json"
        matches = scrape_wines_raw(
            country_code=country_code,
            wine_type_id=type_id,
            wine_type_name=type_name,
            client=api_client,
            sleep_seconds=sleep_seconds,
            output_path=output_path,
        )
        logger.info("Saved %s raw %s matches to %s", len(matches), type_name, output_path)


def scrape_all_wine_types(
    *,
    country_code: str = COUNTRY_CODE,
    wine_types: dict[Any, str] = WINE_TYPES,
    output_dir: Path = DATA_DIR,
    client: VivinoClient | None = None,
    sleep_seconds: float = 0.0,
) -> None:
    """
    Scrape all configured wine types and save each to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    api_client = client or VivinoClient()

    for type_id, type_name in wine_types.items():
        logger.info("Scraping %s wines...", type_name)
        df = scrape_wines(
            country_code=country_code,
            wine_type_id=type_id,
            wine_type_name=type_name,
            client=api_client,
            sleep_seconds=sleep_seconds,
        )
        output_path = output_dir / f"{country_code.lower()}_{type_name}.csv"
        df.to_csv(output_path, index=False)
        logger.info("Saved %s %s wines to %s", len(df), type_name, output_path)
