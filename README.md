# NeuroVault AI

## What is RAG?

RAG (Retrieval-Augmented Generation) is an advanced AI architecture that combines:

- Information Retrieval
- Semantic Search
- Large Language Models (LLMs)

Traditional AI chatbots generate answers only from pre-trained knowledge, which can cause hallucinations.

RAG solves this problem by:
1. Retrieving relevant information from uploaded documents
2. Sending that context to the LLM
3. Generating grounded and context-aware answers

This improves:
- Accuracy
- Reliability
- Context understanding
- Hallucination reduction

---

# How RAG Works

```text
User Question
      ↓
Semantic Search
      ↓
Retrieve Relevant Chunks
      ↓
Send Context to LLM
      ↓
Generate AI Answer
```

---

# Project Overview

NeuroVault AI is an intelligent RAG-based assistant that allows users to upload PDF documents and ask natural language questions based on uploaded knowledge.

The system performs:
- PDF ingestion
- Text chunking
- Embedding generation
- Vector storage
- Semantic retrieval
- LLM-based response generation

---

# Problem Statement

Most traditional chatbots:
- hallucinate answers
- lack domain knowledge
- cannot understand private company documents
- cannot perform semantic document search

Organizations require AI assistants capable of:
- understanding internal PDFs
- retrieving relevant information
- answering accurately
- reducing hallucination

Examples:
- Medical Assistant
- HR Assistant
- Legal Assistant
- Enterprise Knowledge Assistant
- College Assistant

---

# Our Solution

NeuroVault AI uses:
- HuggingFace Embeddings
- ChromaDB Vector Store
- Groq LLM API
- Semantic Retrieval
- Memory-based conversations

to create an intelligent AI assistant capable of answering questions from uploaded PDFs.

---

# Complete Workflow

```text
                ┌──────────────────┐
                │   User Uploads   │
                │      PDF         │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │   PDF Loader     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Text Chunking   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Embedding Model  │
                │ HuggingFace      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ ChromaDB Vector  │
                │     Database     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Semantic Search  │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │   Groq LLM API   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Final AI Answer  │
                └──────────────────┘
```

---

# How We Built This Project

---

# Step 1 — PDF Upload

Users upload PDF documents through the Streamlit interface.

Supported:
- Medical PDFs
- Notes
- Research Papers
- Company Documents
- Reports

---

# Step 2 — PDF Processing

PDFs are processed using:

```python
PyPDFLoader
```

The loader extracts raw textual content from uploaded documents.

---

# Step 3 — Text Chunking

Large text is divided into smaller chunks using:

```python
RecursiveCharacterTextSplitter
```

Chunking improves:
- semantic retrieval
- context matching
- retrieval accuracy

---

# Step 4 — Embedding Generation

Each chunk is converted into vector embeddings using:

```python
sentence-transformers/all-MiniLM-L6-v2
```

These embeddings represent semantic meaning numerically.

---

# Step 5 — ChromaDB Vector Storage

Generated embeddings are stored inside:

```text
ChromaDB
```

ChromaDB enables:
- vector similarity search
- semantic retrieval
- fast document querying

---

# Step 6 — Semantic Retrieval

When a user asks a question:

1. Question embedding generated
2. Similar chunks searched
3. Relevant context retrieved

This process is called:

```text
Semantic Search
```

---

# Step 7 — LLM Response Generation

Retrieved chunks are sent to:

```python
ChatGroq
```

using:

```python
model_name="llama-3.3-70b-versatile"
```

The LLM generates grounded answers using retrieved document context.

---

# Memory System

The project also includes conversation memory using:

```python
ConversationBufferMemory
```

This helps:
- maintain chat history
- support conversational interaction
- improve contextual continuity

---

# Current Project Architecture

```text
NeuroVault-AI/
│
├── agents/
│   ├── llm_agent.py
│   ├── retriever_agent.py
│   ├── response_agent.py
│   └── memory_agent.py
│
├── ingestion/
│   ├── pdf_loader.py
│   ├── text_chunker.py
│   └── embedding_model.py
│
├── database/
│   └── chroma_store.py
│
├── ui/
│   └── streamlit_ui.py
│
├── chroma_db/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Backend Development |
| Streamlit | Frontend UI |
| LangChain | RAG Framework |
| ChromaDB | Vector Database |
| HuggingFace | Embedding Model |
| Groq API | LLM Inference |
| PyPDFLoader | PDF Processing |

---

# Current Features

- PDF Upload
- Semantic Search
- Context-Aware Answers
- ChromaDB Vector Search
- HuggingFace Embeddings
- Groq LLM Integration
- Memory-Based Chat
- Hallucination Reduction
- Modular Architecture

---

# Example Questions

- What is the diagnosis?
- Summarize the document
- What medicines are prescribed?
- Explain the treatment plan
- What are the symptoms?
- What recommendations are given?

---

# API Integration

The project uses Groq API for LLM inference.

Environment variable:

```env
GROQ_API_KEY=your_api_key
```

The API key is loaded using:

```python
os.getenv("GROQ_API_KEY")
```

LLM requests are sent through:

```python
ChatGroq
```

---

# Current Limitations

- Single PDF processing
- No OCR support
- No image understanding
- No web search integration
- No multi-agent orchestration

---

# Future Enhancements

- Agentic RAG
- Multi-PDF Support
- OCR Integration
- Voice Assistant
- Web Search RAG
- Multi-Agent Workflow
- Root Cause Analysis Engine
- Knowledge Graph Integration

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/NeuroVault-AI.git
```

---

# Create Virtual Environment

```bash
py -3.10 -m venv venv
```

---

# Activate Environment

```bash
venv\Scripts\activate
```

---

# Install Requirements

```bash
pip install -r requirements.txt
```

---

# Create .env File

```env
GROQ_API_KEY=your_groq_api_key
```

---

# Run Application

```bash
streamlit run app.py
```

---

# Author

Saravanakumar  
B.Tech Artificial Intelligence & Data Science Student  
Python Developer | GenAI Enthusiast