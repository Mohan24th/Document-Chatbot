import os
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def create_vector_store(pdf_path):

    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vector_store


def get_answer(question):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    docs = vector_store.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = f"""
    Answer the question only from the provided context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    sources = []

    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        sources.append(f"Page {page + 1}")

    return response.content, list(set(sources))