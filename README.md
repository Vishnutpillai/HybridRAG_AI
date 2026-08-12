<div align="center">

⚡ Hybrid RAG AI Assistant

Intelligent document-grounded question answering with Hybrid Search

Semantic Retrieval + BM25 + RRF + Confidence Scoring + Groq LLM

<br>



<br>

<a href="https://github.com/Vishnutpillai/rag-hybrid-search">
  <img src="https://img.shields.io/badge/⭐%20View%20on%20GitHub-181717?style=for-the-badge&logo=github" alt="GitHub">
</a>

</div>

🧠 What is this?

Hybrid RAG AI Assistant is an end-to-end Retrieval-Augmented Generation application that answers questions from a controlled collection of Machine Learning and Deep Learning documents.

Instead of sending a question directly to an LLM, the system first retrieves relevant evidence using two complementary search strategies:

                         USER QUESTION
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             🧠 Vector Search       🔎 BM25 Search
                ChromaDB             Keyword Search
                    │                   │
                    └─────────┬─────────┘
                              ▼
                         🔀 RRF FUSION
                              │
                              ▼
                     📊 CONFIDENCE SCORE
                              │
                              ▼
                       📚 TOP-K CONTEXT
                              │
                              ▼
                         🤖 GROQ LLM
                              │
                              ▼
                    💬 GROUNDED ANSWER
                              │
                              ▼
                  📄 SOURCES + PAGE NUMBERS

The result is a more robust retrieval pipeline that can handle both semantic similarity and exact technical terminology.

✨ Why Hybrid Search?

A single retrieval method can miss useful evidence.

🧠 Vector Search

Good for understanding meaning and concepts.

"How does a neural network learn?"

can retrieve content discussing optimization even if the exact words are different.

🔎 BM25

Good for exact words, technical terms, and phrases.

"log-likelihood gradient"

can benefit from lexical matching.

🔀 RRF

The two rankings are combined using Reciprocal Rank Fusion, producing a unified hybrid ranking.

Vector Search ──────┐
                    ├──► RRF ───► Best Combined Results
BM25 Search ────────┘

🚀 Highlights

Capability

Implementation

📄 PDF ingestion

PyMuPDF / PDF loader

✂️ Chunking

Recursive text splitting

🧠 Semantic retrieval

Embeddings + ChromaDB

🔎 Lexical retrieval

BM25Okapi

🔀 Hybrid ranking

Reciprocal Rank Fusion

📊 Confidence

Retrieval + evidence + overall

🤖 Generation

Groq / LLaMA

🛡️ Grounding

Context-only RAG prompt

⚡ Backend

FastAPI

🎨 Frontend

Streamlit

🐳 Containerization

Docker

📚 Citations

Source + page metadata

📚 Current Knowledge Base

The application is currently indexed on:

<div align="center">

📄 Documents

📑 Pages

🧩 Chunks

2

1,127

3,107

</div>

Documents

data/raw/
├── Machine_Learning.pdf
└── Deep_Learning.pdf

Index statistics

Machine Learning PDF     → 676 chunks
Deep Learning PDF        → 2,431 chunks
                         ───────────────
Total filtered chunks    → 3,107

🏗️ System Architecture

flowchart TB

    U[👤 User]

    UI[🎨 Streamlit Frontend<br/>localhost:8501]

    API[⚡ FastAPI<br/>localhost:8000]

    Q[❓ User Question]

    V[🧠 ChromaDB<br/>Semantic Retrieval]
    B[🔎 BM25<br/>Keyword Retrieval]

    R[🔀 Reciprocal Rank Fusion<br/>Hybrid Ranking]

    C[📊 Confidence Scoring]

    CTX[📚 Retrieved Context]

    LLM[🤖 Groq LLM<br/>LLaMA]

    A[💬 Grounded Answer]
    S[📄 Sources + Pages]

    U --> UI
    UI --> API
    API --> Q

    Q --> V
    Q --> B

    V --> R
    B --> R

    R --> C
    R --> CTX

    C --> LLM
    CTX --> LLM

    LLM --> A
    LLM --> S

🔄 End-to-End Pipeline

01 — Document Ingestion

PDF
 │
 ▼
PDF Loader
 │
 ▼
Pages
 │
 ▼
Recursive Character Splitter
 │
 ▼
3,107 chunks

02 — Indexing

Each chunk is processed through two retrieval paths:

                  Document Chunk
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Embedding             Tokenization
             │                   │
             ▼                   ▼
         ChromaDB               BM25

03 — Query Retrieval

Question
   │
   ├──────────────► ChromaDB ─────► Semantic Ranking
   │
   └──────────────► BM25 ─────────► Keyword Ranking
                                      │
                         ┌────────────┘
                         ▼
                    RRF Fusion
                         │
                         ▼
                     Top-K = 5

04 — Answer Generation

Top-K Evidence
      │
      ▼
Confidence Calculation
      │
      ▼
RAG Prompt
      │
      ▼
Groq LLM
      │
      ▼
Answer + Sources

🎯 Grounded RAG

The application is designed to avoid unsupported answers.

The LLM receives instructions to answer only from the retrieved document context.

For example:

Question:
What is the price of iPhone 17?

Result:
I could not find the answer in the provided documents.

This is important because a RAG system should distinguish between:

✅ Information supported by the knowledge base

vs.

❌ Information that is not present in the knowledge base

📊 Confidence Layer

The system exposes three heuristic scores:

Retrieval Confidence
        │
        ├──────────────┐
        │              │
        ▼              ▼
Evidence Confidence → Overall Confidence

Example API response:

{
  "confidence": {
    "retrieval_confidence": 0.44,
    "evidence_confidence": 1.0,
    "overall_confidence": 0.67
  }
}

Note: These are heuristic scores, not calibrated probabilities.

🖥️ Application

The project includes a Streamlit interface for interacting with the RAG API.

Interface flow

┌───────────────────────────────────────────────┐
│        ⚡ Hybrid RAG AI Assistant             │
│                                               │
│  Ask questions about the indexed documents   │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │ What is deep learning?                  │  │
│  └─────────────────────────────────────────┘  │
│                                               │
│              🔍 Ask Question                  │
│                                               │
│  ───────────────────────────────────────────  │
│                                               │
│  🧠 Answer                                    │
│  ┌─────────────────────────────────────────┐  │
│  │ Grounded response generated from the    │  │
│  │ retrieved document context...           │  │
│  └─────────────────────────────────────────┘  │
│                                               │
│  📊 Confidence                                │
│  📄 Sources                                   │
└───────────────────────────────────────────────┘

🗂️ Project Structure

rag-hybrid-search/
│
├── 📁 data/
│   ├── 📁 raw/
│   │   ├── Machine_Learning.pdf
│   │   └── Deep_Learning.pdf
│   │
│   └── 📁 chroma_db/
│
├── 📁 frontend/
│   └── ui.py
│
├── 📁 src/
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

🛠️ Tech Stack

<div align="center">

Layer

Technology

Language

Python 3.11

Document Processing

PyMuPDF / LangChain

Embeddings

Sentence Transformer

Vector Database

ChromaDB

Keyword Retrieval

BM25Okapi

Ranking

RRF

LLM

Groq / LLaMA 3.3 70B Versatile

API

FastAPI + Uvicorn

UI

Streamlit

Container

Docker

Version Control

Git + GitHub

</div>

⚙️ Installation

1. Clone

git clone https://github.com/Vishnutpillai/rag-hybrid-search.git
cd rag-hybrid-search

2. Create environment

conda create -n rag python=3.11
conda activate rag

3. Install dependencies

pip install -r requirements.txt

4. Configure Groq

Create .env:

GROQ_API_KEY=your_groq_api_key

⚠️ Never commit .env to GitHub.

🧪 Run the RAG Pipeline

For a direct terminal test:

python -m src.rag_pipeline

Example:

Enter your question: what is deep learning?

Expected flow:

Loading documents
      ↓
Splitting
      ↓
Embeddings
      ↓
ChromaDB
      ↓
BM25
      ↓
Hybrid Search
      ↓
Confidence
      ↓
Groq
      ↓
Final Answer

⚡ Run the API

Build Docker image

docker build -t hybrid-rag-ai .

Start FastAPI

From the project root:

docker run --env-file .env -p 8000:8000 -v "%cd%\data:/app/data" hybrid-rag-ai

API

http://localhost:8000

Swagger

http://localhost:8000/docs

🎨 Run the Frontend

Keep the FastAPI container running.

Open a second terminal:

cd C:\Users\abhig\Downloads\rag-hybrid-search
conda activate rag
streamlit run frontend/ui.py

Open:

http://localhost:8501

Local architecture

Browser
   │
   ▼
Streamlit :8501
   │
   │ HTTP POST
   ▼
FastAPI :8000
   │
   ▼
Hybrid RAG
   │
   ▼
Groq

🔌 API Example

Request

POST /ask

{
  "question": "What is deep learning?"
}

Response

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

🐳 Docker

The backend is containerized using a lightweight Python image.

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

The host data/ directory is mounted into the container so document and ChromaDB data can persist outside the container.

🔐 Security

The project keeps secrets and large local artifacts outside Git tracking.

Recommended .gitignore:

.env
*.pdf
__pycache__/
*.pyc
data/chroma_db/

Never commit:

❌ GROQ_API_KEY
❌ .env
❌ PDF datasets
❌ ChromaDB local database
❌ Python cache files

🧩 Troubleshooting

Relative import error

❌

python src/rag_pipeline.py

✅

python -m src.rag_pipeline

BM25 tuple error

If you see:

'tuple' object has no attribute 'get_scores'

make sure the hybrid search receives the actual BM25Okapi object rather than a tuple returned alongside tokenized data.

Docker cannot find PDFs

Verify:

data/
└── raw/
    ├── Machine_Learning.pdf
    └── Deep_Learning.pdf

and run:

docker run --env-file .env -p 8000:8000 -v "%cd%\data:/app/data" hybrid-rag-ai

Streamlit cannot connect

First verify:

http://localhost:8000/docs

Then start:

streamlit run frontend/ui.py

📈 Roadmap

✅ Completed

PDF ingestion

Document chunking

Embedding generation

ChromaDB vector store

BM25 retrieval

Hybrid search

RRF ranking

Confidence scoring

Groq LLM integration

FastAPI backend

Swagger API documentation

Docker backend

Streamlit frontend

Source and page tracking

🔜 Next

Docker Compose

Automated retrieval evaluation

Recall@K / MRR evaluation

RAG benchmark dataset

API authentication

Rate limiting

Logging and observability

Cloud deployment

CI/CD pipeline

Production monitoring

💡 Engineering Concepts Demonstrated

This project goes beyond a basic LLM chatbot.

Retrieval

Vector Search
      +
BM25
      ↓
Hybrid Retrieval
      ↓
RRF

Generation

Retrieved Evidence
      ↓
Context Construction
      ↓
Groq LLM
      ↓
Grounded Response

Production Engineering

Python
  +
FastAPI
  +
Streamlit
  +
Docker
  +
Git/GitHub

This makes the project suitable as an AI Engineering / RAG portfolio project.

👤 Author

<div align="center">

Vishnu T Pillai

Aspiring Data Scientist | AI / ML Enthusiast

<br>

<a href="https://github.com/Vishnutpillai">
  <img src="https://img.shields.io/badge/GitHub-Vishnutpillai-181717?style=for-the-badge&logo=github" alt="GitHub">
</a>

<a href="https://www.linkedin.com/in/vishnu-t-pillai">
  <img src="https://img.shields.io/badge/LinkedIn-Vishnu%20T%20Pillai-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
</a>

</div>

<div align="center">

⭐ If you find this project useful, consider giving it a star!

Built with Python • Retrieval • RAG • FastAPI • Streamlit • Docker • Groq

</div>