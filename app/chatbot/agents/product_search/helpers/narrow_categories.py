import re

from app.chatbot.agents.product_search.helpers.catalog_filters import get_catalog_filters
from app.chatbot.agents.product_search.helpers.search_result import NOT_FOUND
from app.chatbot.agents.product_search.product_search_agent import MAX_PRODUCTS, ProductSearchResult
from app.chatbot.agents.product_search.tools.search_products import search_products


# Lowercase words with a trailing "s" removed, padded for whole-word matching
def normalize(text: str) -> str:
    words = [word[:-1] if word.endswith("s") else word for word in re.findall(r"[a-z0-9-]+", text.lower())]
    return f" {' '.join(words)} "


# Return the subcategories the request names, e.g. "hoodies" -> Hoodies & Sweaters
def mentioned_subcategories(request: str, catalog: dict) -> list[dict]:
    text = normalize(request)
    return [
        sub for sub in catalog["subcategories"]
        if any(part.strip() and normalize(part) in text for part in sub["name"].split("&"))
    ]


# Keep the agent's narrowing only for a whole category; if a subcategory was named, search it directly
def check_narrowing(request: str, result: ProductSearchResult) -> ProductSearchResult:
    catalog = get_catalog_filters()
    if not result.narrow_categories or not catalog:
        return result.model_copy(update={"narrow_categories": []})

    subcategories = mentioned_subcategories(request, catalog)
    if subcategories:
        return search_subcategories(subcategories)

    categories = [name for name in result.narrow_categories if name in catalog["categories"]]
    if not categories:
        return result.model_copy(update={"narrow_categories": []})

    message = f"Sure! Which kind of {' or '.join(categories)} products are you looking for? Pick one below."
    return ProductSearchResult(message=message, product_ids=[], narrow_categories=categories)


# Search the named subcategories, the same way a clicked badge does
def search_subcategories(subcategories: list[dict]) -> ProductSearchResult:
    product_ids = []
    for sub in subcategories:
        found = search_products.invoke({"category": sub["category"], "subcategory": sub["name"]})
        product_ids += [product["id"] for product in found.get("products", []) if product["id"] not in product_ids]

    if not product_ids:
        return ProductSearchResult(message=NOT_FOUND, product_ids=[])

    names = " and ".join(sub["name"] for sub in subcategories)
    return ProductSearchResult(message=f"Here are some {names}.", product_ids=product_ids[:MAX_PRODUCTS])
