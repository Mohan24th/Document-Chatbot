from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

query = "What is inheritance?"

results = vector_store.similarity_search(
    query,
    k=3
)

print(f"\nQuestion: {query}\n")

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content[:500])
    print("\nMetadata:")
    print(doc.metadata)