"""Web service layer — delegates to shared services."""
from __future__ import annotations

from uain.services import find_similar


def find_similar_wines(query: str, k: int = 5) -> dict:
    return find_similar(query, k=k)
