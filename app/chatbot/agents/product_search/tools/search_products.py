import os
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL")
REQUEST_TIMEOUT_SECONDS = 10


@tool
def search_products(
    keywords: Optional[str] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    product_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Search active products in the store catalog. Returns up to 10 products.

    Use exact names from the catalog list in your instructions for category,
    subcategory, brand and product_type. Search terms must be in English.

    keywords -- Free text matched against product name, description and SKU
    category -- Category name
    subcategory -- Subcategory name
    brand -- Brand name
    min_price -- Minimum price in USD
    max_price -- Maximum price in USD
    product_type -- One of: top, featured, new_arrival, best

    Returns:
    products -- Matching products with id, name, brand, category, subcategory,
                price, original_price (set when on sale), short_description
    unmatched_filters -- Filters whose name does not exist in the catalog,
                         each with the closest real names as suggestions
    """
    params = {
        "keywords": keywords,
        "category": category,
        "subcategory": subcategory,
        "brand": brand,
        "min_price": min_price,
        "max_price": max_price,
        "product_type": product_type,
    }

    try:
        response = requests.get(
            f"{LARAVEL_API_URL}/api/ai/products/search",
            params={key: value for key, value in params.items() if value is not None},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        return {"error": f"Product search is unavailable: {error}"}

    return {
        "products": data.get("data", []),
        "unmatched_filters": data.get("unmatched_filters", {}),
    }
