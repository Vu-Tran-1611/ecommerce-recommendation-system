from langchain_core.messages import HumanMessage


# Run an agent and return its final answer
def run_agent(agent, request: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=request)]})
    return result["messages"][-1].content
