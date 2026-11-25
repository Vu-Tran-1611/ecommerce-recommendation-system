import json

from langchain_core.messages import ToolMessage

from app.chatbot.agents.product_search.product_search_agent import MAX_PRODUCTS, ProductSearchResult
from app.chatbot.agents.product_search.tools.search_products import search_products

NOT_FOUND = "Sorry, I couldn't find products that match your request. Please try other keywords."
FOUND = "Here are some products that match your request."


# Return the agent's structured result, or the products found before it stopped
def get_search_result(result: dict) -> ProductSearchResult:
    if result.get("structured_response"):
        return result["structured_response"]

    product_ids = found_product_ids(result["messages"])
    return ProductSearchResult(message=FOUND if product_ids else NOT_FOUND, product_ids=product_ids)


# Collect product IDs from search results, in order found
def found_product_ids(messages) -> list[int]:
    product_ids = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name != search_products.name:
            continue
        try:
            products = json.loads(message.content).get("products", [])
        except (TypeError, ValueError, AttributeError):
            continue
        for product in products:
            if product.get("id") not in product_ids:
                product_ids.append(product["id"])
    return product_ids[:MAX_PRODUCTS]
