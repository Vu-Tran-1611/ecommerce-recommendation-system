from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from app.chatbot.llm import get_chat_model
from app.chatbot.agents.store_policy.tools.retrieve_store_policies import retrieve_store_policies

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


if __name__ == "__main__":
    store_policy_agent = build_store_policy_agent()
    result = store_policy_agent.invoke({"messages": [HumanMessage(content="How to request a return?")]})
    print(result["messages"][-1].content)
