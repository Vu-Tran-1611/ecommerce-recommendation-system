import json
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    ToolCallLimitMiddleware,
    dynamic_prompt,
)
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from app.chatbot.agents.product_search.catalog_filters import format_catalog_filters, get_catalog_filters
from app.chatbot.agents.product_search.tools.search_products import search_products
from app.chatbot.llm import get_chat_model

SYSTEM_PROMPT = """You are the Product Search Agent of an ecommerce store.
Find products that match the user's request with the search_products tool.

Rules:
- Use only the filters the user asked for. Do not add a product_type unless the user asks for it.
- Use only names from the catalog below for category, subcategory, brand and product_type.
- For several brands or items, search each one once (you can search in parallel).
- As soon as the results contain relevant products, stop searching and give the final answer.
- Map the user's words to the closest catalog names (e.g. "sneakers" -> a shoe subcategory).
- If the store clearly does not carry something, say so without searching.
- If unmatched_filters is returned, retry once with a suggestion or a better catalog name.
- If no products match, retry once with fewer filters. Never repeat the same search.
- If still nothing matches, finish with an empty product_ids list and say nothing was found.
- Pick only relevant products from the tool results, most relevant first.
- Never invent products. Product cards with images, prices and links are shown to the user
  separately, so do not list product details in your message.
- Write a short message (1-2 sentences) in the same language as the request (English by default).

{catalog}"""

MAX_SEARCHES = 3
MAX_MODEL_CALLS = 6
MAX_FALLBACK_PRODUCTS = 8
NOT_FOUND = "Sorry, I couldn't find products that match your request. Please try other keywords."
FOUND = "Here are some products that match your request."


class ProductSearchResult(BaseModel):
    """Final answer of the product search."""

    message: str = Field(description="Short message to the user, without product details")
    product_ids: list[int] = Field(description="IDs of the chosen products, most relevant first")


# Inject the live catalog into the prompt
@dynamic_prompt
def catalog_prompt(request: ModelRequest) -> str:
    return SYSTEM_PROMPT.format(catalog=format_catalog_filters(get_catalog_filters()))


# Build the product search agent once
@lru_cache
def build_product_search_agent():
    return create_agent(
        model=get_chat_model(),
        tools=[search_products],
        middleware=[
            catalog_prompt,
            ToolCallLimitMiddleware(tool_name="search_products", run_limit=MAX_SEARCHES),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ],
        response_format=ToolStrategy(ProductSearchResult),
        name="product_search_agent",
    )


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
    return product_ids[:MAX_FALLBACK_PRODUCTS]


# Run a product search and return the structured result
def run_product_search(request: str) -> ProductSearchResult:
    result = build_product_search_agent().invoke({"messages": [HumanMessage(content=request)]})
    if result.get("structured_response"):
        return result["structured_response"]

    # Fall back to products already found when the agent is stopped early
    product_ids = found_product_ids(result["messages"])
    return ProductSearchResult(message=FOUND if product_ids else NOT_FOUND, product_ids=product_ids)


# Expose the agent as a tool for the supervisor
@tool(response_format="content_and_artifact")
def delegate_to_product_search_agent(request: str) -> tuple[str, list[int]]:
    """Send a product search task to the Product Search Agent.

    Use it when the user wants to find, browse or compare products, e.g. by keyword,
    category, subcategory, brand, price range or type (top, featured, new arrival, best).
    The chosen products are shown to the user as cards below your answer.

    request -- The user's product request, rewritten as a clear standalone task.
    """
    result = run_product_search(request)
    return result.message, result.product_ids


if __name__ == "__main__":
    print(run_product_search("Men watches under $500"))
