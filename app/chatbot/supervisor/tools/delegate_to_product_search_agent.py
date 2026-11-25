from langchain.tools import tool
from langchain_core.messages import ToolMessage

from app.chatbot.agents.product_search.product_search_agent import run_product_search


@tool(response_format="content_and_artifact")
def delegate_to_product_search_agent(request: str) -> tuple[str, list[int]]:
    """Send a product search task to the Product Search Agent.

    Use it when the user wants to find, browse or compare products, e.g. by keyword,
    category, subcategory, brand, price range or type (top, featured, new arrival, best).
    The chosen products are shown to the user as cards below your answer.

    request -- The user's product request, rewritten as a clear standalone task.
    """
    result = run_product_search(request)
    # Content goes to the supervisor, product IDs stay in the artifact
    return result.message, result.product_ids


# Collect product IDs from this tool's results in a supervisor run
def collect_product_ids(messages) -> list[int]:
    product_ids = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == delegate_to_product_search_agent.name:
            for product_id in message.artifact or []:
                if product_id not in product_ids:
                    product_ids.append(product_id)
    return product_ids
