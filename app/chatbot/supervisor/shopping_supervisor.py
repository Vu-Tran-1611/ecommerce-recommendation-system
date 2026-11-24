from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langsmith import traceable

from app.chatbot.agents.product_search.product_search_agent import delegate_to_product_search_agent
from app.chatbot.agents.store_policy.store_policy_agent import delegate_to_store_policy_agent
from app.chatbot.llm import get_chat_model

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


# Collect product IDs returned by the product search agent
def collect_product_ids(messages) -> list[int]:
    product_ids = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == delegate_to_product_search_agent.name:
            for product_id in message.artifact or []:
                if product_id not in product_ids:
                    product_ids.append(product_id)
    return product_ids


# Answer a user question through the supervisor
@traceable
def run_shopping_supervisor(question: str) -> dict:
    result = build_shopping_supervisor().invoke({"messages": [HumanMessage(content=question)]})
    return {
        "response": result["messages"][-1].content,
        "products": collect_product_ids(result["messages"]),
    }


if __name__ == "__main__":
    print(run_shopping_supervisor("Show me men watches under $500"))
