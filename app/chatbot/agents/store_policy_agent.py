from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import tool

from app.chatbot.agents.agent_utils import run_agent
from app.chatbot.llm import get_chat_model
from app.chatbot.tools.store_policy_tools import retrieve_store_policies

SYSTEM_PROMPT = (
    "You are the Store Policy Agent of an ecommerce store. "
    "Answer questions about the store's policies and information using the "
    "retrieve_store_policies tool. "
    "Base the answer only on the retrieved context and do not guess. "
    "If the context does not cover the question, say so and suggest contacting customer support. "
    "Be concise and respond in the user's language."
)


# Build the store policy agent once
@lru_cache
def build_store_policy_agent():
    return create_agent(
        model=get_chat_model(),
        tools=[retrieve_store_policies],
        system_prompt=SYSTEM_PROMPT,
        name="store_policy_agent",
    )


# Expose the agent as a tool for the supervisor
@tool
def delegate_to_store_policy_agent(request: str) -> str:
    """Send a store policy question to the Store Policy Agent.

    Use it for questions about shipping, returns, refunds, warranty, payments,
    order cancellation, customer support, FAQ, privacy, terms or the store itself.

    request -- The user's policy question, rewritten as a clear standalone question.
    """
    return run_agent(build_store_policy_agent(), request)


if __name__ == "__main__":
    print(run_agent(build_store_policy_agent(), "How to request a return?"))
