from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

print("API KEY FOUND:", os.getenv("GOOGLE_API_KEY") is not None)

loader = PyPDFLoader("data\\Java Notes .pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print(f"Chunks Created: {len(chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector = embeddings.embed_query(
    chunks[0].page_content
)

print("\nVector Dimension:")
print(len(vector))

print("\nFirst 10 Values:")
print(vector[:10])

