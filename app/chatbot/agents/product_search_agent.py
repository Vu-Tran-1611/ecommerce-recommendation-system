from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import tool

from app.chatbot.agents.agent_utils import run_agent
from app.chatbot.llm import get_chat_model
from app.chatbot.tools.product_search_tools import search_products

SYSTEM_PROMPT = (
    "You are the Product Search Agent of an ecommerce store. "
    "Find products that match the user's request using the search_products tool. "
    "Only answer with products returned by the tool, never invent products, prices or links. "
    "Follow the output format described in the tool. "
    "Respond in the user's language."
)


# Build the product search agent once
@lru_cache
def build_product_search_agent():
    return create_agent(
        model=get_chat_model(),
        tools=[search_products],
        system_prompt=SYSTEM_PROMPT,
        name="product_search_agent",
    )


# Expose the agent as a tool for the supervisor
@tool
def delegate_to_product_search_agent(request: str) -> str:
    """Send a product search task to the Product Search Agent.

    Use it when the user wants to find, browse or compare products, e.g. by keyword,
    category, subcategory, brand, price range or type (top, featured, new arrival, best).

    request -- The user's product request, rewritten as a clear standalone task.
    """
    return run_agent(build_product_search_agent(), request)


if __name__ == "__main__":
    print(run_agent(build_product_search_agent(), "Men watches under $500"))
