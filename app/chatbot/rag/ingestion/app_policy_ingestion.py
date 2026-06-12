import os 

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()  # Load environment variables from .env file 

if __name__ == "__main__":
    print("Starting document ingestion...")  

    #1. Load PDF document 
    loader = PyPDFLoader("data/policy_data/online_shop_policies.pdf")
    documents = loader.load() 
    print(f"Loaded {len(documents)} documents.")

    #2 Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    
    #3. Create Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",show_progress_bar=True,retry_min_seconds=10)

    #4. Ingest into Pinecone
    PineconeVectorStore.from_documents(documents = chunks, embedding = embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))
    print("Document ingestion completed successfully.") 