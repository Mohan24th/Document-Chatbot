# AI Document Assistant (RAG Powered)

## Live Demo

**Application URL:** https://document-chatbot-rag-demo.streamlit.app/

---

## Project Overview

AI Document Assistant is a Retrieval-Augmented Generation (RAG) application built to understand the complete workflow behind modern AI-powered document question-answering systems.

The primary goal of this project was not just to build a chatbot, but to gain hands-on experience with:

* LangChain
* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embeddings
* Semantic Search
* Gemini LLM Integration
* Streamlit Deployment

The application allows users to upload PDF documents, ask questions about the content, generate summaries, and create study notes using AI.

---

## Features

### Document Upload

* Upload PDF documents dynamically.
* No hardcoded documents.

### Semantic Search

* Retrieves relevant document chunks based on meaning rather than keyword matching.

### AI-Powered Question Answering

* Uses Gemini to answer questions based on retrieved document context.

### Source Citations

* Displays source page references used for generating answers.

### Document Summarization

* Generates concise summaries of uploaded documents.

### Study Notes Generation

* Creates:

  * Important Topics
  * Quick Revision Notes
  * 5-Mark Questions
  * 10-Mark Questions

### Interactive Chat Interface

* Chat-style user experience using Streamlit.

---

## Learning Objectives

This project was built primarily as a learning exercise to understand the complete RAG pipeline.

### Concepts Explored

#### LangChain

* Document Loaders
* Text Splitters
* Embeddings
* Vector Stores
* Retrieval Chains
* Prompt Engineering

#### RAG Architecture

* Document Ingestion
* Chunking
* Embedding Generation
* Vector Storage
* Similarity Search
* Context Augmentation
* LLM Response Generation

#### Vector Databases

* ChromaDB
* Similarity Search
* Embedding Storage

#### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

#### Large Language Models

* Gemini 2.5 Flash

#### Deployment

* Streamlit Cloud

---

## Architecture

```text
PDF Upload
     ↓
Document Loader
     ↓
Chunking
     ↓
Embeddings
     ↓
ChromaDB
     ↓
Retriever
     ↓
Relevant Chunks
     ↓
Gemini
     ↓
Final Answer
```

---

## Tech Stack

### Frontend

* Streamlit

### Backend

* Python

### AI / LLM

* Google Gemini 2.5 Flash

### Framework

* LangChain

### Vector Database

* ChromaDB

### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

### PDF Processing

* PyPDF

---

## Project Structure

```text
Document-Chatbot/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│
├── chroma_db/
│
└── src/
```

---

## Challenges Faced & Solutions

### 1. Gemini Embedding API Issues

#### Problem

Initially attempted to use Gemini embeddings and encountered:

* 404 Model Not Found
* Quota Limit Errors
* Rate Limiting Issues

#### Solution

Switched to:

```python
sentence-transformers/all-MiniLM-L6-v2
```

This provided:

* Local embeddings
* Faster processing
* No API usage limits

---

### 2. Vector Database Returning Incorrect Results

#### Problem

After uploading a new PDF, the system still answered questions using content from a previously processed document.

Example:

* Uploaded SQL Notes
* Retrieved Java Notes content

#### Root Cause

The application was loading an old Chroma vector store.

#### Solution

Implemented dynamic vector store creation and session-based storage to ensure retrieval occurs only from the currently uploaded document.

---

### 3. Streamlit Rerun Behavior

#### Problem

Every interaction caused the application to rerun.

#### Solution

Used:

```python
st.session_state
```

to preserve:

* Chat history
* Vector store
* Current document state

---

### 4. Slow Processing Times

#### Problem

Embedding generation and model loading increased response times.

#### Solution

Implemented:

```python
@st.cache_resource
```

for:

* Embedding models
* Gemini model initialization

---

## Key Takeaways

Through this project I learned:

* How RAG systems work internally.
* Why chunking strategy matters.
* How semantic search differs from keyword search.
* How vector databases retrieve relevant context.
* How LLMs can be grounded using external knowledge.
* Common deployment challenges when building AI applications.
* Practical usage of LangChain v1 architecture.

---

## Future Improvements

### Planned Features

* Multi-PDF Knowledge Base
* Conversation Memory
* Download Notes as PDF
* Advanced Citation Support
* User Authentication
* Persistent Vector Storage
* Hybrid Search (BM25 + Vector Search)
* Better UI/UX Design
* Document Analytics
* Exam Preparation Mode

---

## Deployment

The application is deployed on Streamlit Cloud.

Live Demo:

https://document-chatbot-rag-demo.streamlit.app/

---

## Acknowledgements

This project was built as part of my learning journey in:

* LangChain
* Retrieval-Augmented Generation (RAG)
* Vector Databases
* LLM Engineering
* Generative AI Application Development

The goal was to move beyond tutorials and gain practical experience by implementing a complete end-to-end RAG application.
