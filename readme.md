#  Document Chatbot using RAG, LangChain, and ChromaDB

## Overview

This project is an AI-powered Document Question Answering system built using **LangChain**, **ChromaDB**, and **Google Gemini/OpenAI LLMs**.

The chatbot uses the **Retrieval-Augmented Generation (RAG)** architecture to answer user questions based on the contents of uploaded PDF documents.

Instead of relying only on the LLM's pre-trained knowledge, the system retrieves relevant information from the uploaded document and provides context-aware responses.

--

## Project Goal

The objective of this project is to understand and implement the complete RAG pipeline, including:

- Document Loading
- Text Chunking
- Embedding Generation
- Vector Database Storage
- Semantic Search
- Context Retrieval
- LLM-Based Answer Generation

This project serves as a practical introduction to building AI applications that can interact with external knowledge sources.

---

## Features

- Upload and process PDF documents
- Split documents into searchable chunks
- Generate embeddings for document content
- Store embeddings in Chroma Vector Database
- Retrieve relevant context using similarity search
- Generate accurate answers using an LLM
- Query documents in natural language

---

## Architecture

```text
PDF Document
      │
      ▼
Document Loader
      │
      ▼
Text Splitter
      │
      ▼
Embedding Model
      │
      ▼
Chroma Vector Database
      │
      ▼
Retriever
      │
      ▼
Large Language Model
      │
      ▼
Generated Answer
```

---

## Tech Stack

### Framework

- LangChain

### Vector Database

- ChromaDB

### Embedding Model

- Gemini Embeddings / OpenAI Embeddings

### Large Language Model

- Google Gemini
- OpenAI GPT Models
- Groq (Optional)

### Programming Language

- Python

---

## Learning Objectives

Through this project, I aim to understand:

### Retrieval-Augmented Generation (RAG)

- Why RAG is needed
- How retrieval improves LLM responses
- Limitations of traditional prompting

### Embeddings

- Converting text into vector representations
- Semantic similarity search

### Vector Databases

- Storage of embeddings
- Efficient document retrieval

### LangChain Components

- Document Loaders
- Text Splitters
- Embedding Models
- Retrievers
- Chains

### AI Application Development

- Integrating LLMs with external knowledge
- Building practical Generative AI applications

---

## Future Enhancements

- Multi-PDF support
- Source citations in answers
- Conversation memory
- FastAPI backend
- Streamlit/Web interface
- Docker deployment
- Advanced RAG techniques

---

## Project Status

 Currently under development for learning and experimentation with:

- LangChain
- RAG
- Embeddings
- ChromaDB
- LLM Integration
