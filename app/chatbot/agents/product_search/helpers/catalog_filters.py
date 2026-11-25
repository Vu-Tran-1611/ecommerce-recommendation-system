import time

import requests

from app.chatbot.utils.laravel_api import laravel_get

CACHE_TTL_SECONDS = 600

_cache = {"data": None, "expires_at": 0.0}


# Fetch catalog filter names from Laravel, cached for 10 minutes
def get_catalog_filters() -> dict | None:
    if _cache["data"] and time.monotonic() < _cache["expires_at"]:
        return _cache["data"]

    try:
        _cache["data"] = laravel_get("/api/ai/catalog/filters")["data"]
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
