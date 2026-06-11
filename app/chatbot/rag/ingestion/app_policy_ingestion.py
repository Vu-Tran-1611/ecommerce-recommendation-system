import os 

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()  # Load environment variables from .env file