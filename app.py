import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --------------------------
# PAGE CONFIG
# --------------------------

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3rem;
    font-weight: 600;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.5rem;
}

/* Success box */
.stSuccess {
    border-radius: 10px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #888;
    border-radius: 10px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# CACHE
# --------------------------

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

# --------------------------
# SESSION STATE
# --------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --------------------------
# SIDEBAR
# --------------------------

with st.sidebar:

    st.title("Document Assistant")

    st.markdown("""
    Chat with your PDF using Retrieval-Augmented Generation (RAG),
    semantic search, and Gemini AI.
    """)    

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    process_btn = st.button(
        "Process Document",
        use_container_width=True
    )

    st.markdown("---")

    summary_btn = st.button(
        "Generate Summary",
        use_container_width=True
    )

    notes_btn = st.button(
        "Generate Study Notes",
        use_container_width=True
    )

    st.markdown("---")

    if st.session_state.pdf_name:
        st.success(
            f"Current Document:\n\n{st.session_state.pdf_name}"
        )

# --------------------------
# HEADER
# --------------------------

st.title("Document Assistant")

st.caption(
    "Upload a PDF and interact with it using Retrieval-Augmented Generation."
)

# --------------------------
# PDF PROCESSING
# --------------------------

if uploaded_file and process_btn:

    try:

        with st.spinner(
            "Processing document..."
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp:

                tmp.write(
                    uploaded_file.getvalue()
                )

                pdf_path = tmp.name

            loader = PyPDFLoader(
                pdf_path
            )

            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(
                documents
            )

            embeddings = load_embeddings()

            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.vector_store = (
                vector_store
            )

            st.session_state.pdf_name = (
                uploaded_file.name
            )

            st.session_state.messages = []

        st.success(
            "Document processed successfully."
        )

    except Exception as e:

        st.error(
            f"Processing failed: {str(e)}"
        )

# --------------------------
# CHAT SECTION
# --------------------------

if st.session_state.vector_store:

    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):
            st.markdown(
                msg["content"]
            )

    prompt = st.chat_input(
        "Ask a question about the document..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        try:

            with st.spinner(
                "Generating answer..."
            ):

                docs = (
                    st.session_state.vector_store
                    .similarity_search(
                        prompt,
                        k=3
                    )
                )

                context = "\n\n".join(
                    [
                        doc.page_content
                        for doc in docs
                    ]
                )

                llm = load_llm()

                rag_prompt = f"""
Answer ONLY from the provided context.

Context:
{context}

Question:
{prompt}
"""

                response = llm.invoke(
                    rag_prompt
                )

                answer = (
                    response.content
                )

                source_pages = []

                for doc in docs:

                    page = (
                        doc.metadata.get(
                            "page",
                            0
                        )
                    )

                    source_pages.append(
                        f"Page {page + 1}"
                    )

                answer += (
                    "\n\n---\n\n**Sources**\n\n"
                )

                answer += "\n".join(
                    list(
                        set(
                            source_pages
                        )
                    )
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            with st.chat_message(
                "assistant"
            ):
                st.markdown(answer)

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

# --------------------------
# SUMMARY
# --------------------------

if summary_btn and st.session_state.vector_store:

    try:

        with st.spinner(
            "Generating summary..."
        ):

            docs = (
                st.session_state.vector_store
                .similarity_search(
                    "Summarize document",
                    k=10
                )
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            llm = load_llm()

            response = llm.invoke(
                f"""
Generate a concise summary.

Context:
{context}
"""
            )

        st.subheader(
            "Document Summary"
        )

        st.write(
            response.content
        )

    except Exception as e:

        st.error(str(e))

# --------------------------
# STUDY NOTES
# --------------------------

if notes_btn and st.session_state.vector_store:

    try:

        with st.spinner(
            "Generating study notes..."
        ):

            docs = (
                st.session_state.vector_store
                .similarity_search(
                    "Important concepts",
                    k=10
                )
            )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            llm = load_llm()

            response = llm.invoke(
                f"""
Generate:

1. Important Topics
2. Quick Revision Notes
3. 5 Mark Questions
4. 10 Mark Questions

Context:
{context}
"""
            )

        st.subheader(
            "Study Notes"
        )

        st.write(
            response.content
        )

    except Exception as e:

        st.error(str(e))

# --------------------------
# EMPTY STATE
# --------------------------

if not st.session_state.vector_store:

    st.info(
        "Upload a PDF and click 'Process Document' to begin."
    )