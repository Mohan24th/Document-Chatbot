from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

docs = vector_store.similarity_search(
    "Summarize the entire document",
    k=10
)

context = "\n\n".join(
    [doc.page_content for doc in docs]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

prompt = f"""
Create a concise summary of the document.

Context:
{context}

Include:
1. Main Topics
2. Key Concepts
3. Important Points
"""

response = llm.invoke(prompt)

print("\nSummary:\n")
print(response.content)