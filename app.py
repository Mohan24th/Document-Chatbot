import streamlit as st
import os

from src.rag import create_vector_store
from src.rag import get_answer

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄"
)

st.title("📄 AI Document Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    os.makedirs("data", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Process Document"):

        with st.spinner("Creating Vector Database..."):

            create_vector_store(pdf_path)

        st.success("Document Processed Successfully")

    question = st.text_input(
        "Ask a question"
    )

    if question:

        with st.spinner("Thinking..."):

            answer, sources = get_answer(question)

        st.subheader("Answer")

        st.write(answer)

        st.subheader("Sources")

        for source in sources:
            st.write(source)