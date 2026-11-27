from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    ToolCallLimitMiddleware,
    dynamic_prompt,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.chatbot.agents.product_search.helpers.catalog_filters import format_catalog_filters, get_catalog_filters
from app.chatbot.agents.product_search.tools.search_products import search_products
from app.chatbot.llm import get_chat_model

SYSTEM_PROMPT = """You are the Product Search Agent of an ecommerce store.
Find products that match the user's request with the search_products tool.

Rules:
- If the request has nothing to search for (no product, category, brand, use or budget),
  do not search. Finish with an empty product_ids list and ask what they are looking for,
  naming a few categories from the catalog.
- Keep keywords short (1-4 words, in English): a product name or the kind of product that
  fits the need. Search understands typos and meaning. Leave out filler like "gift",
  "birthday" or who it is for. Examples: "Pop Mart Lotso keychain", "kids toys" for a bored
  child, "video games" for a gamer, "phone case" for someone who drops their phone.
- Do not guess a category or subcategory for a named product. Use them only when the user
  asks for a whole category or type (e.g. "show me tablets", "do you have makeup?").
- Use only the filters the user asked for. Do not add a product_type unless the user asks for it.
- For price words ("cheap", "cheapest", "cheaper", "budget", "most expensive") use sort
  (price_asc or price_desc) instead of putting those words in keywords.
- Use only names from the catalog below for category, subcategory, brand and product_type.
- For several brands or items, search each one once (you can search in parallel).
- As soon as the results contain relevant products, stop searching and give the final answer.
- When a filter is needed, map the user's words to the closest catalog names (e.g. "cosmetics" -> Makeup).
- If the store clearly does not carry something, say so without searching.
- If unmatched_filters is returned, retry once with a suggestion or a better catalog name.
- If no products match, retry once with fewer filters. Never repeat the same search.
- If still nothing matches, finish with an empty product_ids list and say nothing was found.
- Pick only relevant products from the tool results, at most {max_products}, most relevant first.
- For several brands or items, include products from each one, split about evenly,
  and alternate them in product_ids (e.g. Apple, Samsung, Xiaomi, Apple, ...).
- Never invent products. Product cards with images, prices and links are shown to the user
  separately, so do not list product details in your message.
- Write a short message (1-2 sentences) in the same language as the request (English by default).

{catalog}"""

MAX_SEARCHES = 3
MAX_MODEL_CALLS = 6
MAX_PRODUCTS = 12


class ProductSearchResult(BaseModel):
    """Final answer of the product search."""

    message: str = Field(description="Short message to the user, without product details")
    product_ids: list[int] = Field(
        max_length=MAX_PRODUCTS,
        description=f"IDs of the chosen products, at most {MAX_PRODUCTS}, most relevant first",
    )


# Inject the live catalog into the prompt
@dynamic_prompt
def catalog_prompt(request: ModelRequest) -> str:
    return SYSTEM_PROMPT.format(
        max_products=MAX_PRODUCTS,
        catalog=format_catalog_filters(get_catalog_filters()),
    )


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


if __name__ == "__main__":
    product_search_agent = build_product_search_agent()
    result = product_search_agent.invoke({"messages": [HumanMessage(content="Men watches under $500")]})
    print(result.get("structured_response"))
