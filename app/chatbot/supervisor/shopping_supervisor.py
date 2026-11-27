from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage

from app.chatbot.llm import get_chat_model
from app.chatbot.supervisor.tools.delegate_to_product_search_agent import delegate_to_product_search_agent
from app.chatbot.supervisor.tools.delegate_to_store_policy_agent import delegate_to_store_policy_agent

MAX_DELEGATIONS = 4

# Register sub-agents
SUB_AGENTS = [
    delegate_to_product_search_agent,
    delegate_to_store_policy_agent,
]

SYSTEM_PROMPT = (
    "You are the Shopping Supervisor of an ecommerce store. "
    "You do not answer store questions yourself. Route each task to the right sub-agent tool. "
    "If the user asks several things, split them and send each part to its sub-agent. "
    "Send all product requests in one message to the product search agent. "
    "If a request is too vague to act on (e.g. 'I want to search some products'), "
    "do not call any sub-agent: ask one short question about what they need. "
    "A described need, occasion or person is not vague (e.g. 'my kid is bored', "
    "'a gift for my brother who loves games'): send it to the product search agent. "
    "If only one sub-agent was used, return its answer as given. "
    "If several were used, join their answers without changing their content. "
    "Product cards are shown below your answer automatically, so never list product details. "
    "Use plain text with simple Markdown only, no HTML. "
    "If the request is unrelated to this ecommerce store, politely say you can only help "
    "with store-related questions. If the request is unclear, ask for clarification. "
    "Always respond in the same language as the user's message."
)


# Build the supervisor once
@lru_cache
def build_shopping_supervisor():
    return create_agent(
        model=get_chat_model(),
        tools=SUB_AGENTS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[ToolCallLimitMiddleware(run_limit=MAX_DELEGATIONS)],
        name="shopping_supervisor",
    )






if __name__ == "__main__":
    shopping_supervisor = build_shopping_supervisor()
    result = shopping_supervisor.invoke({"messages": [HumanMessage(content="How to request a return?")]})
    print(result["messages"][-1].content)
