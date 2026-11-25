from langchain.tools import tool
from langchain_core.messages import HumanMessage

from app.chatbot.agents.store_policy.store_policy_agent import build_store_policy_agent


@tool
def delegate_to_store_policy_agent(request: str) -> str:
    """Send a store policy question to the Store Policy Agent.

    Use it for questions about shipping, returns, refunds, warranty, payments,
    order cancellation, customer support, FAQ, privacy, terms or the store itself.

    request -- The user's policy question, rewritten as a clear standalone question.
    """
    store_policy_agent = build_store_policy_agent()
    result = store_policy_agent.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content
