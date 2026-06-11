import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model 
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage 
from langsmith import traceable
from app.chatbot.tools.product_tools import search_products 

load_dotenv() 
MAX_ITERATIONS = 5
MODEL = "gpt-5.5" 

# Agent Loop 
@traceable
def shopping_agent(tools,questions):  
    # Tools 
    tools_dict = {tool.name: tool for tool in tools} 

    # Initialize chat model
    llm = init_chat_model(MODEL,temperature=0) 
    llm_with_tools = llm.bind_tools(tools) 
    # Define Messages 
    messages = [
        SystemMessage(content="You are a helpful shopping assistant. " \
        "You can use the following tools to assist users in finding products and making purchases." \
        "When you return a final answer, be polite, friendly, and concise. Always provide helpful information to the user." \
        "If products are found, start with short sentence such as:" \
        " 'Here are some products you might like:' or 'I found some products that match your criteria:' followed by the format:" \
        "- product name " \
        "- thumbnail image URL" \
        "- price" \
        "- brand" \
        "- category" \
        "- product URL" \
        " When listing products in the final answer, show at most 5 products." \
        "If no products are found, say 'I couldn't find any products that match your criteria. Please try different keywords. If the user's request is unclear, ask for clarification"
        "") 
        ,
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
    tools = [search_products] 
    questions = "I want to know the return policy for a product I bought last week." 
    shopping_agent(tools,questions)
