from langchain.tools import tool
from langchain_core.messages import HumanMessage

from app.chatbot.agents.product_search.helpers.search_result import get_search_result
from app.chatbot.agents.product_search.product_search_agent import build_product_search_agent


@tool(response_format="content_and_artifact")
def delegate_to_product_search_agent(request: str) -> tuple[str, list[int]]:
    """Send a product search task to the Product Search Agent.

    Use it when the user wants to find, browse or compare products, e.g. by keyword,
    category, subcategory, brand, price range or type (top, featured, new arrival, best).
    The chosen products are shown to the user as cards below your answer.

    request -- The user's product request, rewritten as a clear standalone task.
    """
    product_search_agent = build_product_search_agent()
    result = product_search_agent.invoke({"messages": [HumanMessage(content=request)]})
    search_result = get_search_result(result)
    # Content goes to the supervisor, product IDs stay in the artifact
    return search_result.message, search_result.product_ids
