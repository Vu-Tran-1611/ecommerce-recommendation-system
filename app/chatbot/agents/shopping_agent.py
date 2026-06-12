from dotenv import load_dotenv
from langchain.chat_models import init_chat_model 
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage 
from langsmith import traceable
from app.chatbot.tools.policy_retrieval_tools import retrieve_policies
from app.chatbot.tools.product_tools import search_products
load_dotenv() 
MAX_ITERATIONS = 5
MODEL = "gpt-5.5" 

# Agent Loop 
@traceable
def shopping_agent(questions):  
    # Tools 
    tools = [search_products, retrieve_policies] 
    tools_dict = {tool.name: tool for tool in tools} 

    # Initialize chat model
    llm = init_chat_model(MODEL,temperature=0) 
    llm_with_tools = llm.bind_tools(tools) 
    # Define Messages  
    messages = [
        SystemMessage(content="You are a helpful shopping assistant. " \
        "You can use the following tools to assist users in finding products and answer their questions about policies or concerns about the ecommerce platform." \
        "Here are the tools you can use: " \
        + "\n".join([f"- {tool.name}: {tool.description}" for tool in tools]) + "\n" \
        "When you return a final answer, be polite, friendly, and concise. Always provide helpful information to the user." \
        "If you use any tools, make sure to use them effectively to provide accurate and relevant information to the user. "  \
        "If the user asks a general question that is not specifically about this ecommerce website, " \
        "politely say sorry and inform them that you can only help with questions relating to the " \
        "ecommerce store."
        ),
        HumanMessage(content=questions)
    ] 

    # Agent Loop 
    print("Agent is thinking...") 

    for _ in range(MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages) 
        print(f"AI: {ai_message.content}") 

        tool_calls = ai_message.tool_calls 

        # Case 1: No tool calls, agent has provided a final answer 
        if not tool_calls: 
            print("\nNo tool calls found. Agent has provided a final answer.") 
            print(ai_message.content) 
            return ai_message.content 

        # Case 2: Tool calls found, execute them and add results to messages
        messages.append(ai_message)
        print(f"Tool calls found: {[call.get('name') for call in tool_calls]}") 

        for tool_call in tool_calls:
            tool_name = tool_call.get("name") 
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")

            print(f"Executing tool: {tool_name} with args: {tool_args} and id: {tool_id}") 
            tool_func = tools_dict.get(tool_name) 
            if tool_func: 
                try:
                    observation = tool_func.invoke(tool_args)
                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}") 
                    observation = f"Error executing tool {tool_name}: {e}"
            messages.append(ToolMessage(content=observation, tool_call_id=tool_id))
    print("\nMax iterations reached. Returning last AI message.")
    print(ai_message.content)
    return ai_message.content

if __name__ == "__main__":
    questions = "I want some men watches. Can you help me find some good options under $500? " 
    shopping_agent(questions)
