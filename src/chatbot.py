from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        break

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer the question only from the provided context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources:")

    for doc in docs:
        print(
            f"Page {doc.metadata.get('page', 'Unknown') + 1}"
        )