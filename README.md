# ⚡ Hybrid RAG AI Assistant

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=flat-square&logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> A production-ready Retrieval-Augmented Generation (RAG) application combining **Hybrid Search**, **Vector Embeddings**, **BM25 Keyword Matching**, **Confidence Scoring**, and **Groq LLMs** for intelligent document question-answering.

---

## 🎯 Overview

**Hybrid RAG AI Assistant** is an end-to-end document question-answering system engineered for accuracy and transparency. It goes beyond traditional RAG by combining multiple retrieval strategies to ensure comprehensive and grounded answers directly from your uploaded PDF documents.

### Why Choose Hybrid RAG?

Instead of relying on a single retrieval method, this system intelligently combines:

| Component | Benefit |
|-----------|---------|
| 🔵 **Semantic Vector Search** | Captures meaning-based similarity across documents |
| 🟢 **BM25 Keyword Search** | Excels at exact-term and technical terminology matching |
| 🟣 **Hybrid RRF Fusion** | Combines both signals for more robust rankings |
| 📊 **Confidence Scoring** | Provides transparency into answer quality and evidence strength |
| 🤖 **Groq LLM Integration** | Fast, efficient grounded answer generation |

---

## ✨ Key Features

```
📄 PDF Ingestion              Upload and index PDFs through REST API
✂️  Recursive Chunking         Intelligent overlapping chunks with configurable sizes
🔎 Vector Search (ChromaDB)   Semantic similarity retrieval with HF embeddings
🔤 BM25 Search                Keyword-based relevance matching
🔀 Hybrid RRF Fusion          Combines vector + keyword rankings
📈 Confidence Scoring          Retrieval, evidence, and overall confidence metrics
🤖 Groq LLM                    Generates answers from retrieved context
🧠 Grounded Responses         Prevents hallucinations via context restriction
🚀 Production FastAPI          Auto-generated Swagger documentation
🖥️  Streamlit Dashboard        Interactive, user-friendly interface
🐳 Docker Compose             Reproducible, containerized deployment
🔁 Persistent ChromaDB        Local vector database with embeddings
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PDF Documents                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              PDF Loader (PyMuPDF)                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          Recursive Chunking & Splitting                 │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
  ┌──────────────────┐        ┌──────────────────┐
  │  Embeddings      │        │  BM25 Indexing   │
  │  (Hugging Face)  │        │  (Keyword Index) │
  └────────┬─────────┘        └────────┬─────────┘
           │                           │
           ▼                           ▼
  ┌──────────────────┐        ┌──────────────────┐
  │  ChromaDB        │        │  BM25 Index      │
  │  (Vector Store)  │        │  (Local)         │
  └────────┬─────────┘        └────────┬─────────┘
           └──────────────┬───────────┘
                          ▼
        ┌─────────────────────────────┐
        │   Hybrid RRF Retrieval      │
        │  (Rank Fusion)              │
        └─────────────────┬───────────┘
                          ▼
        ┌─────────────────────────────┐
        │  Confidence Scoring         │
        │  (Quality Metrics)          │
        └─────────────────┬───────────┘
                          ▼
        ┌─────────────────────────────┐
        │  Groq LLM                   │
        │  (Grounded Generation)      │
        └─────────────────┬───────────┘
                          ▼
        ┌─────────────────────────────┐
        │  Answer + Confidence Scores │
        │  + Source Attribution       │
        └─────────────────────────────┘
```

---

## 📁 Project Structure

```
rag-hybrid-search/
│
├── 📂 data/
│   ├── raw/                    # PDF documents (gitignored)
│   └── chroma_db/              # Vector database (gitignored)
│
├── 📂 frontend/
│   ├── ui.py                   # Streamlit application
│   └── requirements.txt         # Frontend dependencies
│
├── 📂 src/
│   ├── __init__.py
│   ├── api.py                  # FastAPI server
│   ├── loader.py               # PDF text extraction
│   ├── splitter.py             # Recursive chunking
│   ├── embedded.py             # Embedding generation
│   ├── vectorstore.py          # ChromaDB integration
│   ├── bm25_retriever.py       # BM25 search
│   ├── hybrid_search.py        # RRF fusion logic
│   ├── confidence.py           # Scoring metrics
│   ├── rag_pipeline.py         # RAG orchestration
│   └── groq_con.py             # Groq LLM connection
│
├── 📄 .env                     # Environment variables (gitignored)
├── 📄 .gitignore
├── 📄 Dockerfile
├── 📄 docker-compose.yml       # Orchestration
├── 📄 requirements.txt          # Backend dependencies
├── 📄 README.md
└── 📄 LICENSE
```

---

## 🔀 Why Hybrid Search Matters

### The Problem with Single Retrieval Methods

| Issue | Impact |
|-------|--------|
| **Vector-Only Search** | May miss technical terminology or exact phrase matching |
| **Keyword-Only Search** | Struggles with paraphrased or differently worded queries |
| **No Ranking Fusion** | Suboptimal result ordering from overlapping strategies |

### The Hybrid Solution

```
Vector Ranking (semantic)
        ↓
    RRF Fusion  ← Reciprocal Rank Fusion
        ↓
BM25 Ranking (keyword)
        ↓
   Optimal Results
```

**Result**: A more robust, accurate retrieval system that captures both semantic meaning and exact terminology.

---

## 📊 Confidence Scoring

Every answer includes transparency metrics:

```json
{
  "question": "What does hybrid search combine?",
  "answer": "Hybrid search combines semantic vector search with keyword-based retrieval such as BM25.",
  "confidence": {
    "retrieval_confidence": 0.85,
    "evidence_confidence": 0.95,
    "overall_confidence": 0.89
  },
  "retrieved_chunks": 5,
  "sources": [
    "Machine_Learning.pdf"
  ]
}
```

**Metrics Explained**:
- **Retrieval Confidence** (0-1): How well documents matched the query
- **Evidence Confidence** (0-1): How relevant the evidence to the question
- **Overall Confidence** (0-1): Combined assessment of answer quality

---

## 🛠️ Technology Stack

### Backend
- **Python 3.11** – Core language
- **FastAPI** – High-performance REST API framework
- **Uvicorn** – ASGI server
- **LangChain** – RAG orchestration
- **PyMuPDF** – PDF text extraction
- **Sentence Transformers / HuggingFace** – Embeddings
- **ChromaDB** – Vector database
- **BM25** – Keyword retrieval
- **Groq API** – Fast LLM inference

### Frontend
- **Streamlit** – Interactive web interface
- **Requests** – HTTP client

### Infrastructure
- **Docker** & **Docker Compose** – Containerization and orchestration

---

## 🚀 Quick Start with Docker

### Prerequisites

- Docker Desktop ([Download](https://www.docker.com/products/docker-desktop))
- Git
- Groq API Key ([Get one free](https://groq.com))

### Installation Steps

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Vishnutpillai/rag-hybrid-search.git
cd rag-hybrid-search
```

#### 2️⃣ Create `.env` File

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

⚠️ **Security**: Never commit `.env` to version control.

#### 3️⃣ Build & Start

```bash
docker compose up -d --build
```

The first build may take a few minutes as dependencies install.

#### 4️⃣ Verify Services

```bash
docker compose ps
```

Expected output:
```
NAME                   STATUS
hybrid-rag-backend     Up
hybrid-rag-frontend    Up
```

#### 5️⃣ Access the Application

| Service | URL | Purpose |
|---------|-----|---------|
| 🎨 **Streamlit Frontend** | http://localhost:8501 | Interactive assistant UI |
| 📚 **Swagger API Docs** | http://localhost:8000/docs | API documentation & testing |
| ✅ **Health Check** | http://localhost:8000/health | Backend status |
| 📄 **API Root** | http://localhost:8000/ | API information |

---

## 📡 API Endpoints

### System Endpoints

```http
GET /
Returns API information and metadata
```

```http
GET /health
Backend health check status
```

### RAG Endpoints

```http
POST /v1/ask
Query the document collection

Request:
{
  "question": "What is deep learning?",
  "top_k": 5
}

Response:
{
  "question": "...",
  "answer": "...",
  "confidence": {...},
  "sources": [...]
}
```

### Document Management

```http
GET /v1/documents
List all indexed documents

Response:
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
```

```http
POST /v1/ingest
Upload and index a PDF document

Content-Type: multipart/form-data
Body: file (PDF)
```

---

## 💡 Example Queries

After indexing documents, try these questions:

- "What is machine learning?"
- "Explain the difference between supervised and unsupervised learning"
- "What does hybrid search combine?"
- "How do neural networks work?"
- "What is the role of embeddings in RAG?"
- "Compare BM25 and vector search"

---

## 📊 Frontend Interface

The Streamlit application provides:

✅ Backend URL configuration  
✅ Real-time API health monitoring  
✅ Question input with examples  
✅ Suggested question templates  
✅ Generated answers with grounding  
✅ Confidence score visualization  
✅ Retrieved sources attribution  
✅ Document collection overview  

---

## 🐳 Docker Compose Architecture

```
┌─────────────────────────────────────────────────┐
│           Docker Compose Network                │
│                                                 │
│  ┌──────────────────┐  ┌─────────────────────┐ │
│  │  FastAPI Backend │  │  Streamlit Frontend │ │
│  │   Port: 8000     │  │   Port: 8501        │ │
│  │   (Uvicorn)      │  │   (Python app)      │ │
│  └────────┬─────────┘  └──────────┬──────────┘ │
│           │                       │             │
│           └───────── HTTP ────────┘             │
│                                                 │
│  Shared Volumes:                                │
│  • data/raw/         (PDFs)                     │
│  • data/chroma_db/   (Vector DB)                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 📋 Useful Docker Commands

### Container Management

```bash
# View running containers
docker compose ps

# View all logs
docker compose logs -f

# Backend-specific logs
docker compose logs -f backend

# Frontend-specific logs
docker compose logs -f frontend

# Stop services
docker compose down

# Restart services
docker compose restart

# Rebuild without cache
docker compose build --no-cache
```

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused Docker images
docker image prune

# Clear build cache
docker builder prune -af
```

---

## 🧰 Running Without Docker (Local Development)

### Backend Setup

```bash
# Create and activate conda environment
conda activate rag
pip install -r requirements.txt

# Start FastAPI server
uvicorn src.api:app --reload

# Backend runs on http://localhost:8000
```

### Frontend Setup

```bash
# Open another terminal
conda activate rag
pip install -r frontend/requirements.txt

# Start Streamlit app
streamlit run frontend/ui.py

# Frontend runs on http://localhost:8501
```

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | API key for Groq LLM inference |

### Example `.env`

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxx
```

### Security Best Practices

❌ **Never expose your API key in**:
- GitHub repositories
- README files
- Screenshots or logs
- Source code
- Docker images
- Public documentation

✅ **Always store secrets in**:
- `.env` files (locally)
- Environment variables
- Secure vault systems (production)

---

## ⚠️ Important Notes

### PDF Files

Large or copyrighted documents should **not** be committed to Git:

```bash
# Add to .gitignore
*.pdf
data/raw/
```

Store PDFs in `data/raw/` directory locally.

### Vector Database

ChromaDB can become large and should be excluded from Git:

```bash
# Add to .gitignore
data/chroma_db/
```

Database persists between container restarts in Docker volumes.

### Git Hygiene

```
.gitignore should include:
- .env (secrets)
- *.pdf (documents)
- data/chroma_db/ (vector DB)
- __pycache__/
- *.egg-info/
- .DS_Store
```

---

## 🔮 Future Roadmap

### Near-term
- [ ] Authentication & authorization
- [ ] PostgreSQL metadata storage
- [ ] Redis caching layer
- [ ] Background document processing

### Medium-term
- [ ] Streaming LLM responses
- [ ] Cross-encoder re-ranking
- [ ] RAGAS evaluation framework
- [ ] Document deletion/re-indexing API

### Long-term
- [ ] Multi-user document collections
- [ ] Prometheus/Grafana monitoring
- [ ] CI/CD pipeline
- [ ] Automated test suite
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Advanced citation & source grounding

---

## 🎯 What This Project Demonstrates

### Machine Learning & NLP
```
Python Expertise
├── Machine Learning & NLP
├── Embeddings & Vector Spaces
├── Information Retrieval
├── Hybrid Search Algorithms
├── RAG Systems Architecture
└── LLM Integration & Prompting
```

### Production Engineering
```
Full-Stack Development
├── REST API Design (FastAPI)
├── Containerization (Docker)
├── Orchestration (Docker Compose)
├── UI Development (Streamlit)
├── Environment Management
└── Service Architecture
```

---

## 👨‍💻 About the Author

**Vishnu T Pillai**  
Aspiring Data Scientist & AI Engineer focused on Machine Learning, Deep Learning, RAG systems, and production-oriented AI applications.

### Connect & Follow

🔗 **LinkedIn**: [linkedin.com/in/vishnu-t-pillai](https://www.linkedin.com/in/vishnu-t-pillai)  
🐙 **GitHub**: [github.com/Vishnutpillai](https://github.com/Vishnutpillai)  

---

## ⭐ Show Your Support

If this project helped you understand Hybrid Search, RAG systems, FastAPI, Docker, or LLM integration:

**Please consider giving this repository a ⭐ on GitHub!**

Your support encourages further development and improvements.

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📞 Support & Feedback

Have questions or suggestions? Feel free to:

- 📧 Open an issue on GitHub
- 💬 Reach out via LinkedIn
- 🐛 Report bugs with detailed examples

---

<div align="center">

### Built with ❤️ for the AI Community

**[⬆ back to top](#-hybrid-rag-ai-assistant)**

</div>
