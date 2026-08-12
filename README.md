Hybrid RAG AI Assistant
<p align="center"> <strong>Production-oriented Retrieval-Augmented Generation system for document-grounded question answering</strong> </p>

<p align="center"> <a href="https://github.com/Vishnutpillai/rag-hybrid-search"> <img src="https://img.shields.io/badge/GitHub-Repository-181717?logo=github" alt="GitHub"> </a> <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11"> <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI"> <img src="https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit"> <img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-5B4BDB" alt="ChromaDB"> <img src="https://img.shields.io/badge/BM25-Keyword%20Search-2F855A" alt="BM25"> <img src="https://img.shields.io/badge/Groq-LLM-F55036" alt="Groq"> <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white" alt="Docker"> </p>

Overview
Hybrid RAG AI Assistant is an end-to-end Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from PDF documents.

The system combines:

Semantic vector retrieval using ChromaDB and embeddings

Lexical keyword retrieval using BM25

Reciprocal Rank Fusion (RRF) for hybrid ranking

Confidence scoring for retrieval and evidence quality

Groq LLM for grounded answer generation

FastAPI for the backend REST API

Streamlit for the interactive web interface

Docker for reproducible backend deployment

The current document collection contains 1,127 PDF pages and produces 3,107 filtered chunks.

Grounding principle: the LLM is instructed to answer from retrieved document context rather than relying on outside knowledge. When the required information is not found in the provided documents, the system can return a grounded "not found" response.

Key Features
Feature	Description
📄 PDF ingestion	Loads PDF documents from data/raw/
✂️ Recursive chunking	Splits documents into overlapping text chunks
🧠 Embeddings	Generates semantic representations for chunks
🗄️ ChromaDB	Persistent vector database for semantic retrieval
🔎 BM25	Keyword-based retrieval for exact terminology
🔀 Hybrid Search	Combines semantic and lexical retrieval
🏆 RRF	Fuses independent rankings into a unified ranking
📊 Confidence	Calculates retrieval, evidence, and overall confidence
🤖 Groq	Generates answers from retrieved context
⚡ FastAPI	Exposes the RAG pipeline through an API
🖥️ Streamlit	Provides an interactive question-answering UI
🐳 Docker	Containerizes the FastAPI backend
📚 Source tracking	Returns source PDF, page, and retrieval scores
Architecture
High-level flow
                         ┌─────────────────────┐
                         │        User         │
                         └──────────┬──────────┘
                                    │
                          ┌─────────▼─────────┐
                          │   Streamlit UI    │
                          │   localhost:8501  │
                          └─────────┬─────────┘
                                    │ HTTP
                                    ▼
                          ┌───────────────────┐
                          │     FastAPI       │
                          │   localhost:8000  │
                          └─────────┬─────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────┐
                    │       Hybrid RAG Pipeline  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
          ┌─────────▼─────────┐       ┌────────▼────────┐
          │ Semantic Retrieval│       │ Keyword Retrieval│
          │     ChromaDB      │       │       BM25       │
          └─────────┬─────────┘       └────────┬────────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                        ┌──────────────────┐
                        │ RRF / Hybrid Rank│
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │ Confidence Score │
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │ Context Builder  │
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │     Groq LLM     │
                        └────────┬─────────┘
                                 ▼
                 ┌─────────────────────────────────┐
                 │ Answer + Confidence + Sources  │
                 └─────────────────────────────────┘
Document ingestion pipeline
PDF files
   │
   ▼
PyMuPDF / PDF loader
   │
   ▼
Document pages
   │
   ▼
RecursiveCharacterTextSplitter
   │
   ├──────────────► Embeddings ─────► ChromaDB
   │
   └──────────────► Tokenization ───► BM25 index
Query pipeline
Question
   │
   ├──────────────► ChromaDB semantic search
   │
   └──────────────► BM25 keyword search
                         │
                         ▼
                  Reciprocal Rank Fusion
                         │
                         ▼
                  Top-K retrieved chunks
                         │
                         ▼
                  Confidence calculation
                         │
                         ▼
                    RAG prompt
                         │
                         ▼
                      Groq LLM
                         │
                         ▼
               Grounded final answer
Project Structure
rag-hybrid-search/
│
├── data/
│   ├── raw/
│   │   ├── Machine_Learning.pdf
│   │   └── Deep_Learning.pdf
│   │
│   └── chroma_db/
│
├── frontend/
│   └── ui.py
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── bm25_retriever.py
│   ├── confidence.py
│   ├── embedded.py
│   ├── groq_con.py
│   ├── hybrid_search.py
│   ├── loader.py
│   ├── rag_pipeline.py
│   ├── splitter.py
│   └── vectorstore.py
│
├── .env
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
.env, PDF files, generated ChromaDB data, Python cache files, and other local artifacts should remain excluded through .gitignore.

Current Dataset / Index Statistics
Metric	Current value
PDF documents	2
Total pages	1,127
Initial chunks	3,113
Small chunks removed	6
Final chunks	3,107
Machine Learning chunks	676
Deep Learning chunks	2,431
Default retrieval	Top 5
Vector store	ChromaDB
Keyword retriever	BM25Okapi
Fusion method	Reciprocal Rank Fusion
Tech Stack
Backend
Python 3.11

FastAPI

Uvicorn

Retrieval
ChromaDB

Sentence-transformer embeddings

BM25Okapi

Reciprocal Rank Fusion (RRF)

LLM
Groq API

LLaMA 3.3 70B Versatile

Document Processing
PyMuPDF / PDF loader

LangChain text splitting utilities

Frontend
Streamlit

Requests

Deployment
Docker

Uvicorn

Environment-based configuration

Installation
1. Clone the repository
git clone https://github.com/Vishnutpillai/rag-hybrid-search.git
cd rag-hybrid-search
2. Create the Conda environment
conda create -n rag python=3.11
conda activate rag
3. Install dependencies
pip install -r requirements.txt
4. Configure environment variables
Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
Do not commit .env to GitHub.

Run the RAG Pipeline Directly
To test the complete RAG pipeline from the terminal:

python -m src.rag_pipeline
Then enter a question such as:

What is deep learning?
The pipeline performs:

Load PDFs
   ↓
Split documents
   ↓
Load embeddings
   ↓
Load ChromaDB
   ↓
Build BM25 index
   ↓
Hybrid search
   ↓
Confidence calculation
   ↓
Build RAG prompt
   ↓
Groq
   ↓
Answer + sources
Run the FastAPI Backend with Docker
Build the Docker image
docker build -t hybrid-rag-ai .
Run the backend
From the project root:

docker run --env-file .env -p 8000:8000 -v "%cd%\data:/app/data" hybrid-rag-ai
The API will be available at:

http://localhost:8000
Swagger documentation:

http://localhost:8000/docs
FastAPI API
POST /ask
Example request:

{
  "question": "What is deep learning?"
}
Example response:

{
  "question": "What is deep learning?",
  "answer": "Deep learning is ...",
  "confidence": {
    "retrieval_confidence": 0.82,
    "evidence_confidence": 1.0,
    "overall_confidence": 0.91
  },
  "retrieved_chunks": 5,
  "sources": [
    {
      "rank": 1,
      "source": "data\\raw\\Deep_Learning.pdf",
      "page": 42,
      "dense_score": 0.81,
      "bm25_score": 15.2,
      "rrf_score": 0.03
    }
  ]
}
Test with Python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is deep learning?"},
    timeout=120
)

print(response.json())
Run the Streamlit Frontend
The frontend is located at:

frontend/ui.py
Start the FastAPI Docker backend first.

Then open a second terminal:

conda activate rag
streamlit run frontend/ui.py
Open:

http://localhost:8501
Frontend features
The Streamlit interface displays:

Question input

Generated answer

Retrieval confidence

Evidence confidence

Overall confidence

Number of retrieved chunks

Source PDF

Page number

Dense retrieval score

BM25 score

RRF score

Raw API response

Running the Full Application
You need two processes during local development.

Terminal 1 — FastAPI
cd C:\Users\abhig\Downloads\rag-hybrid-search
conda activate rag

docker run --env-file .env -p 8000:8000 -v "%cd%\data:/app/data" hybrid-rag-ai
Terminal 2 — Streamlit
cd C:\Users\abhig\Downloads\rag-hybrid-search
conda activate rag

streamlit run frontend/ui.py
Then open:

http://localhost:8501
Architecture:

Browser
   │
   ▼
Streamlit :8501
   │
   │ POST /ask
   ▼
FastAPI :8000
   │
   ▼
Hybrid RAG
   │
   ▼
Groq
Retrieval Strategy
1. Semantic Retrieval
The question is embedded and compared with document embeddings stored in ChromaDB.

This helps retrieve conceptually similar text even when the exact words are different.

2. BM25 Retrieval
The same question is tokenized and searched against the BM25 index.

BM25 is particularly useful for:

Exact terminology

Technical keywords

Named entities

Specific phrases

3. Hybrid Search
The semantic and BM25 rankings are combined.

Semantic results ──┐
                   ├──► RRF ──► Hybrid ranking
BM25 results ──────┘
This gives the system both semantic and lexical retrieval capabilities.

Confidence Scoring
The application reports three heuristic confidence values:

Retrieval Confidence
        +
Evidence Confidence
        ↓
Overall Confidence
Example:

{
  "retrieval_confidence": 0.44,
  "evidence_confidence": 1.0,
  "overall_confidence": 0.67
}
These values are heuristic confidence scores, not calibrated probabilities.

This distinction is important when interpreting the output.

Grounded Answering
The RAG prompt instructs the model to:

Use only the retrieved document context.

Avoid inventing facts.

Avoid using outside knowledge.

Return a "not found" response when the answer cannot be supported by the retrieved context.

For example:

Question:
What is the price of iPhone 17?

Answer:
I could not find the answer in the provided documents.
This demonstrates the system's document-grounding behavior.

Example Queries
Example 1
What is machine learning?
Example 2
What is deep learning?
Example 3
What is gradient descent?
Example 4
What is backpropagation?
Example 5 — Out of domain
What is the price of iPhone 17?
The system should not fabricate an answer when the information is absent from the indexed documents.

Dockerfile
The backend uses a lightweight Python 3.11 image:

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

RUN mkdir -p /app/data/raw /app/data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
The local data/ directory is mounted into the container so the PDF and ChromaDB data can be persisted outside the container.

Configuration
Important configuration values are defined in the project source files.

Typical environment configuration:

GROQ_API_KEY=your_groq_api_key
Keep secrets outside source control.

Recommended .gitignore entries:

.env
*.pdf
__pycache__/
*.pyc
data/chroma_db/
Troubleshooting
ImportError: attempted relative import with no known parent package
Run package modules with:

python -m src.rag_pipeline
instead of:

python src/rag_pipeline.py
ModuleNotFoundError: No module named 'loader'
Inside package modules, use relative imports:

from .loader import load_pdfs
Run the module from the project root:

python -m src.splitter
ImportError: cannot import name 'create_bm25'
Make sure the function name used by the caller matches the actual function:

from .bm25_retriever import create_bm25_retriever
and:

bm25 = create_bm25_retriever(chunks)
'tuple' object has no attribute 'get_scores'
create_bm25_retriever() must return the BM25 object expected by hybrid_search.py.

If it returns a tuple such as:

bm25, tokenized_corpus
then pass the BM25 object rather than the entire tuple.

Docker cannot find PDFs
Make sure the host has:

data/
└── raw/
    ├── Machine_Learning.pdf
    └── Deep_Learning.pdf
Run Docker with the data volume mounted:

docker run --env-file .env -p 8000:8000 -v "%cd%\data:/app/data" hybrid-rag-ai
Streamlit cannot connect to FastAPI
Make sure the backend is running first:

http://localhost:8000/docs
Then start:

streamlit run frontend/ui.py
The frontend should call:

http://127.0.0.1:8000/ask
Development Workflow
1. Add / update documents
        ↓
2. Run ingestion and indexing
        ↓
3. Test RAG pipeline
        ↓
4. Test FastAPI /docs
        ↓
5. Test Streamlit frontend
        ↓
6. Build Docker image
        ↓
7. Run Docker container
        ↓
8. Commit changes
        ↓
9. Push to GitHub
        ↓
10. Deploy
Roadmap
PDF ingestion

Recursive text chunking

Embedding generation

ChromaDB vector storage

BM25 keyword retrieval

Hybrid retrieval

Reciprocal Rank Fusion

Confidence scoring

Groq LLM integration

FastAPI REST API

Swagger documentation

Docker backend

Streamlit frontend

Docker Compose for frontend + backend

Automated RAG evaluation

Retrieval metrics such as Recall@K / MRR

Production cloud deployment

Monitoring and observability

Authentication and rate limiting

Why This Project Matters
This project demonstrates an end-to-end AI Engineering / RAG workflow, rather than only an LLM API call.

It covers:

Document ingestion

Text preprocessing

Embedding generation

Vector databases

Information retrieval

Keyword search

Hybrid retrieval

Ranking fusion

Context construction

LLM inference

Confidence estimation

REST API development

Frontend development

Docker containerization

Git/GitHub workflow

Author
Vishnu T Pillai

<p> <a href="https://github.com/Vishnutpillai">GitHub</a> · <a href="https://www.linkedin.com/in/vishnu-t-pillai">LinkedIn</a> </p>

License
This project is available under the MIT License.

<p align="center"> Built with Python, ChromaDB, BM25, FastAPI, Streamlit, Docker, and Groq. </p>

