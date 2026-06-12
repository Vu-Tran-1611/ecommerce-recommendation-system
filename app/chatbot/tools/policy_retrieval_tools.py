import os 
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import requests 
from langchain.tools import tool 
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)

@tool 
def retrieve_policies(query:str) -> Dict[str,Any]: 
    """Retrieve policies or any questions related to policies or concerns about the ecommerce platform
    Using context from the vector store to answer the questions without hallucinating. 
    Use this tool only when the user asks about this store's policies or information, including:
    - shipping
    - returns
    - refunds
    - warranty
    - checkout payments
    - order cancellation
    - customer support
    - FAQ
    - privacy policy
    - terms and conditions
    - about the store

    Do not use this tool for general questions unrelated to the ecommerce store.
    Do not use this tool for general banking, medical, legal, financial, or technical advice. 
    If the user asks a general question that is not specifically about this ecommerce website, 
    politely say sorry and inform them that you can only help with questions relating to the 
    ecommerce store.

    Keyword arguments:
    query -- User's query to search for relevant policies

    Return: Dictionary containing relevant policies 
    If the user's concerns/questions are unclear, ask for clarification. 

    Examples: 
    -"Do I need to create an account to make a purchase?" -> retrieve_policies(query="Do I need to create an account to make a purchase?") 
    -"What payment methods do you accept?" -> retrieve_policies(query="What payment methods do you accept?") 
    -"Can I return a product if I'm not satisfied?" -> retrieve_policies(query="Can I return a product if I'm not satisfied?") 
    -"When my order will arrive?" -> retrieve_policies(query="When will my order arrive?")
    -"How to cancel an order?" -> retrieve_policies(query="How to cancel an order?")
    -"What is the return policy for electronics?" -> retrieve_policies(query="return policy for electronics") 
    -"How long is the warranty for mobile phones?" -> retrieve_policies(query="warranty for mobile phones") 
    """
    retrieve_docs = vector_store.as_retriever().invoke(query,k=10) 
    context = "\n".join([f"Context: {doc.page_content}" for doc in retrieve_docs])
    return {"query": query, "context": context}