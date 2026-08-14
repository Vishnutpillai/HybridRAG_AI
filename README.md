⚡ Hybrid RAG AI Assistant

Production-style Retrieval-Augmented Generation (RAG) application
combining Vector Search, BM25, Hybrid Reciprocal Rank Fusion (RRF),
confidence scoring, Groq LLMs, FastAPI, Streamlit, and Docker.







📌 Overview

Hybrid RAG AI Assistant is an end-to-end document question-answering
system designed to provide grounded answers from uploaded PDF documents.

Instead of relying on a single retrieval strategy, the system combines:

🔵 Semantic Vector Search for meaning-based retrieval

🟢 BM25 Keyword Search for exact-term matching

🟣 Hybrid RRF Retrieval to combine both ranking signals

📊 Confidence Scoring to estimate retrieval/evidence quality

🤖 Groq LLM for grounded answer generation

📄 PDF ingestion with automatic chunking and indexing

⚡ FastAPI backend for production-style APIs

🎨 Streamlit frontend for interactive usage

🐳 Docker Compose for reproducible deployment

The application is designed so that answers are generated from retrieved
document context rather than from unrestricted model knowledge.

✨ Key Features

Feature                             Description

📄 PDF Ingestion                    Upload and index PDF documents
through the API

✂️ Recursive Chunking               Splits documents into manageable
overlapping chunks

🔎 Vector Search                    Retrieves semantically similar
document chunks

🔤 BM25 Search                      Retrieves chunks using keyword
relevance

🔀 Hybrid Search                    Combines vector and BM25 rankings
using RRF

📈 Confidence Scoring               Calculates retrieval, evidence, and
overall confidence

🤖 Groq LLM                         Generates answers using retrieved
context

🧠 Grounded Responses               Reduces unsupported answers by
restricting context

🚀 FastAPI                          REST API with automatic Swagger
documentation

🖥️ Streamlit UI                     Interactive AI assistant interface

🐳 Docker                           Backend and frontend run as
separate containers

🔁 Persistent ChromaDB              Stores vector embeddings locally

🏗️ System Architecture

                         ┌─────────────────────────┐
                         │       PDF Documents     │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      PDF Loader         │
                         │       PyMuPDF           │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Recursive Chunking    │
                         │  chunk_size / overlap   │
                         └────────────┬────────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
              ┌───────────────────┐     ┌───────────────────┐
              │   Embeddings      │     │       BM25        │
              │ Sentence/ HF      │     │ Keyword Retrieval │
              └─────────┬─────────┘     └─────────┬─────────┘
                        │                           │
                        ▼                           ▼
              ┌───────────────────┐     ┌───────────────────┐
              │     ChromaDB      │     │   BM25 Index      │
              │  Vector Search    │     │                   │
              └─────────┬─────────┘     └─────────┬─────────┘
                        │                           │
                        └─────────────┬─────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │   Hybrid RRF Search     │
                         │ Reciprocal Rank Fusion  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │   Confidence Scoring    │
                         │ Retrieval + Evidence    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       RAG Prompt        │
                         │  Retrieved Context + Q  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       Groq LLM          │
                         │   Grounded Generation   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       Final Answer      │
                         │ + Confidence + Sources  │
                         └─────────────────────────┘

🧩 Project Architecture

rag-hybrid-search/
│
├── data/
│   ├── raw/
│   │   ├── PDF documents
│   │   └── ...
│   └── chroma_db/
│
├── frontend/
│   ├── ui.py
│   └── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── loader.py
│   ├── splitter.py
│   ├── embedded.py
│   ├── vectorstore.py
│   ├── bm25_retriever.py
│   ├── hybrid_search.py
│   ├── confidence.py
│   ├── rag_pipeline.py
│   └── groq_con.py
│
├── .env
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── ...

Security: .env, PDF files, and the local ChromaDB directory
should remain excluded from Git when they contain private data or
credentials.

🔄 RAG Pipeline

The application follows this workflow:

1. Document Ingestion

A PDF is uploaded through:

POST /v1/ingest

The backend:

Receives the PDF

Stores it in the raw-data directory

Extracts text using PyMuPDF

Splits text into chunks

Generates embeddings

Stores embeddings in ChromaDB

Updates the BM25 index

2. Query Processing

A user submits a question through:

POST /v1/ask

The system performs:

User Question
      ↓
Vector Search
      +
BM25 Search
      ↓
Hybrid RRF Ranking
      ↓
Top-K Chunks
      ↓
Confidence Scoring
      ↓
RAG Prompt
      ↓
Groq LLM
      ↓
Grounded Answer

🔀 Why Hybrid Search?

A single retrieval method can miss useful information.

Vector Search

Good for:

Semantic similarity

Different wording

Concept-based questions

Example:

"What is neural network training?"

can retrieve content discussing:

"optimizing the parameters of a neural model"

even when the exact words differ.

BM25

Good for:

Exact keywords

Technical terminology

Names and identifiers

Specific phrases

Hybrid Search

The project combines both approaches and uses Reciprocal Rank Fusion
(RRF) to produce a more robust ranking.

Vector Ranking
       +
BM25 Ranking
       ↓
    RRF Fusion
       ↓
Hybrid Ranking

📊 Confidence Scoring

The application exposes confidence-related metrics with the answer.

Typical response structure:

{
  "question": "What does hybrid search combine?",
  "answer": "Hybrid search combines semantic vector search with keyword-based retrieval such as BM25.",
  "confidence": {
    "retrieval_confidence": 0.33,
    "evidence_confidence": 1.0,
    "overall_confidence": 0.6
  },
  "retrieved_chunks": 5,
  "sources": []
}

These scores provide additional visibility into retrieval quality
instead of returning only the generated answer.

🛠️ Technology Stack

Backend

Python 3.11

FastAPI

Uvicorn

LangChain

PyMuPDF

Sentence Transformers / Hugging Face embeddings

ChromaDB

BM25

Groq API

python-dotenv

Frontend

Streamlit

Requests

Infrastructure

Docker

Docker Compose

🚀 Run Locally with Docker

Docker is the recommended way to run the complete application.

Prerequisites

Install:

Docker Desktop

Git

Make sure Docker Desktop is running.

1. Clone the repository

git clone https://github.com/Vishnutpillai/rag-hybrid-search.git
cd rag-hybrid-search

Replace the repository URL if your GitHub repository uses a different
name.

2. Create .env

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

Do not commit this file to GitHub.

3. Build and start the application

docker compose up -d --build

The first build can take several minutes because the backend installs
ML, embedding, vector database, and RAG dependencies.

4. Check containers

docker compose ps

Expected services:

hybrid-rag-backend
hybrid-rag-frontend

5. Open the application

Streamlit Frontend

http://localhost:8501

FastAPI Swagger Documentation

http://localhost:8000/docs

FastAPI Root

http://localhost:8000/

Health Check

http://localhost:8000/health

🐳 Docker Services

The application runs two containers:

┌───────────────────────────────────────────┐
│              Docker Compose               │
│                                           │
│  ┌─────────────────┐  ┌────────────────┐ │
│  │     Backend     │  │    Frontend    │ │
│  │     FastAPI     │  │   Streamlit    │ │
│  │     :8000       │  │     :8501      │ │
│  └────────┬────────┘  └───────┬────────┘ │
│           │                   │           │
│           └────── HTTP ───────┘           │
└───────────────────────────────────────────┘

The Streamlit frontend communicates with the FastAPI backend using the
Docker service name.

🔌 API Endpoints

System

GET /

Returns API information.

GET /health

Checks whether the backend is healthy.

RAG

POST /v1/ask

Ask a question against the indexed document collection.

Example request:

{
  "question": "What is deep learning?",
  "top_k": 5
}

Documents

GET /v1/documents

Lists indexed documents.

Example response:

{
  "count": 3,
  "documents": [
    {
      "filename": "Machine_Learning.pdf",
      "path": "data/raw/Machine_Learning.pdf",
      "type": "PDF",
      "indexed": true
    }
  ]
}

POST /v1/ingest

Uploads and indexes a PDF document.

The endpoint accepts:

multipart/form-data

with a PDF file.

🧪 Example Questions

After indexing documents, try questions such as:

What is machine learning?

What is deep learning?

What does hybrid search combine?

What is the difference between supervised and unsupervised learning?

What is BM25?

The answers should be based on the indexed document context.

📁 Document Ingestion Example

From the FastAPI Swagger UI:

http://localhost:8000/docs

Navigate to:

POST /v1/ingest

Then:

Try it out
    ↓
Choose File
    ↓
Select PDF
    ↓
Execute

A successful response reports the indexed document and chunk
information.

🖥️ Frontend

The Streamlit application provides:

Backend URL configuration

API health check

Question input

Suggested questions

Generated answers

Retrieval quality information

Document/source information

Default Docker backend URL:

http://backend:8000

When accessing the API directly from the host machine:

http://localhost:8000

📈 Current Pipeline Characteristics

The project has been tested with a document collection containing:

Machine Learning PDF

Deep Learning PDF

Additional RAG test PDF

The previously tested collection contained approximately 3,100+
chunks after recursive splitting and indexing.

The exact number can change when documents are added, removed, or
re-indexed.

🔐 Environment Variables

Variable           Required Description

GROQ_API_KEY          Yes API key used for Groq LLM requests

Example:

GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxx

Never expose your real API key in:

GitHub

README files

Screenshots

Source code

Docker images

Public logs

🧹 Stop the Application

docker compose down

Start it again later with:

docker compose up -d

You do not need to rebuild the Docker images every time.

Only rebuild when dependencies or Docker configuration change:

docker compose up -d --build

📋 Useful Docker Commands

Show running containers

docker compose ps

View all logs

docker compose logs -f

Backend logs

docker compose logs -f backend

Frontend logs

docker compose logs -f frontend

Stop containers

docker compose down

Rebuild containers

docker compose build --no-cache

Remove unused Docker build cache

docker builder prune -af

Use Docker cleanup commands carefully because they can remove cached
build layers used by other projects.

🧰 Run Without Docker

If you want to run the backend and frontend directly from your Python
environment:

Backend

conda activate rag
pip install -r requirements.txt
uvicorn src.api:app --reload

Backend:

http://localhost:8000

Frontend

Open another terminal:

conda activate rag
pip install -r frontend/requirements.txt
streamlit run frontend/ui.py

Frontend:

http://localhost:8501

⚠️ Important Notes

PDF files

Large or copyrighted source documents should generally not be committed
to the public Git repository.

Keep them under:

data/raw/

and ignore PDFs with:

*.pdf

ChromaDB

The local vector database can become large and should normally remain
outside Git:

data/chroma_db/

API keys

Keep secrets in .env:

.env

🔮 Future Improvements

Potential production improvements include:

Authentication and authorization

PostgreSQL metadata storage

Redis caching

Background document processing

Streaming LLM responses

Re-ranking with a cross-encoder

Advanced evaluation with RAGAS

Observability and tracing

Prometheus/Grafana monitoring

Cloud deployment

CI/CD pipeline

Automated tests

Document deletion/re-indexing API

Multi-user document collections

Better citation-level source grounding

🎯 Project Highlights

This project demonstrates practical experience with:

Python
│
├── Machine Learning / NLP
├── Embeddings
├── Vector Databases
├── Information Retrieval
├── Hybrid Search
├── RAG
└── LLM Integration

and production-oriented engineering:

FastAPI
│
├── REST APIs
├── Docker
├── Docker Compose
├── Streamlit
├── Environment Management
└── Service-to-Service Communication

👨‍💻 Author

Vishnu T Pillai

Aspiring Data Scientist / AI Engineer focused on Machine Learning, Deep
Learning, RAG systems, and production-oriented AI applications.

Connect

LinkedIn: https://www.linkedin.com/in/vishnu-t-pillai

GitHub: https://github.com/Vishnutpillai

⭐ If You Find This Project Useful

If this project helped you understand Hybrid Search, RAG, FastAPI,
Docker, and LLM integration, consider giving the repository a ⭐ on
GitHub.