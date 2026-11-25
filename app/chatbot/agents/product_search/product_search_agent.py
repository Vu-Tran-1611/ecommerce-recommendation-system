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


if __name__ == "__main__":
    product_search_agent = build_product_search_agent()
    result = product_search_agent.invoke({"messages": [HumanMessage(content="Men watches under $500")]})
    print(result.get("structured_response"))
