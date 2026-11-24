from functools import lru_cache

from langchain.agents import create_agent
from langsmith import traceable

from app.chatbot.agents.agent_utils import run_agent
from app.chatbot.agents.product_search_agent import delegate_to_product_search_agent
from app.chatbot.agents.store_policy_agent import delegate_to_store_policy_agent
from app.chatbot.llm import get_chat_model

# Register sub-agents here
SUB_AGENTS = [
    delegate_to_product_search_agent,
    delegate_to_store_policy_agent,
]

SYSTEM_PROMPT = (
    "You are the Shopping Supervisor of an ecommerce store. "
    "You do not answer store questions yourself. Route each task to the right sub-agent tool. "
    "If the user asks several things, split them and send each part to its sub-agent. "
    "If only one sub-agent was used, return its answer exactly as given, keeping all HTML and links. "
    "If several were used, join their answers without changing their content. "
    "If the request is unrelated to this ecommerce store, politely say you can only help "
    "with store-related questions. If the request is unclear, ask for clarification. "
    "Respond in the user's language."
)


# Build the supervisor once
@lru_cache
def build_shopping_supervisor():
    return create_agent(
        model=get_chat_model(),
        tools=SUB_AGENTS,
        system_prompt=SYSTEM_PROMPT,
        name="shopping_supervisor",
    )


# Answer a user question through the supervisor
@traceable
def run_shopping_supervisor(question: str) -> str:
    return run_agent(build_shopping_supervisor(), question)


if __name__ == "__main__":
    print(run_shopping_supervisor("How to request a return?"))
