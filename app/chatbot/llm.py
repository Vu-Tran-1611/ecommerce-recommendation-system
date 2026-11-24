from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama

load_dotenv()

# Local LLMs
# MODEL = "gemma3:270m" # 270 million parameters - 291 MB
# MODEL = "qwen3:1.7b" # 1.7 billion parameters - 1.4 GB
# MODEL = "qwen3:8b" # 8 billion parameters - 5.2 GB

# Cloud LLMs
MODEL = "gpt-5.4-mini" # Quick Reponse, cheap but less correctness
# MODEL = "gpt-5.5"  # More correct but  slower and the most expensive
# MODEL = "gpt-6-luna" # Balance between correctness, cost and speed
# MODEL = "gpt-6.1-terra" # More correct but slower and more expensive than gpt-6-luna


# Create the chat model shared by all agents
def get_chat_model():
    # return ChatOllama(temperature=0, model=MODEL) # Using local Ollama LLM
    return ChatOpenAI(model=MODEL, reasoning_effort="none") # Using Cloud LLM (i.e. OpenAI API)
