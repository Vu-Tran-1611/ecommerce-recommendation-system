import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL")
CACHE_TTL_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 10

_cache = {"data": None, "expires_at": 0.0}


# Fetch catalog filter names from Laravel, cached for 10 minutes
def get_catalog_filters() -> dict | None:
    if _cache["data"] and time.monotonic() < _cache["expires_at"]:
        return _cache["data"]

    try:
        response = requests.get(
            f"{LARAVEL_API_URL}/api/ai/catalog/filters", timeout=REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        _cache["data"] = response.json()["data"]
        _cache["expires_at"] = time.monotonic() + CACHE_TTL_SECONDS
    except (requests.RequestException, KeyError, ValueError):
        pass  # Keep the last known list

    return _cache["data"]


# Format catalog filters for the agent prompt
def format_catalog_filters(filters: dict | None) -> str:
    if not filters:
        return "Catalog filter list is unavailable. Search by keywords only."

    subcategories = {}
    for sub in filters["subcategories"]:
        subcategories.setdefault(sub["category"] or "Other", []).append(sub["name"])

    lines = ["Categories and their subcategories:"]
    for category in filters["categories"]:
        lines.append(f"- {category}: {', '.join(subcategories.get(category, []))}")
    lines.append(f"Brands: {', '.join(filters['brands'])}")
    lines.append(f"Product types: {', '.join(filters['product_types'])}")
    return "\n".join(lines)
