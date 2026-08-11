from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .loader import load_pdfs
from .splitter import split_documents
from .embedded import create_embedding_model
from .vectorstore import Chroma
from .bm25_retriever import create_bm25_retriever
from .rag_pipeline import answer_question


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATHS = [
    "data/raw/Machine_Learning.pdf",
    "data/raw/Deep_Learning.pdf",
]

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag_documents"
TOP_K = 5


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hybrid RAG API",
    description=(
        "Retrieval-Augmented Generation API using "
        "Vector Search + BM25 + Hybrid Search + Groq"
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system",
        examples=["What is deep learning?"],
    )

    top_k: int = Field(
        default=TOP_K,
        ge=1,
        le=10,
        description="Number of documents to retrieve",
    )


# ============================================================
# GLOBAL RAG COMPONENTS
# ============================================================

vector_store = None
bm25 = None
chunks = None


# ============================================================
# INITIALIZE RAG SYSTEM
# ============================================================

def initialize_rag():

    global vector_store
    global bm25
    global chunks

    print("\n" + "=" * 60)
    print("INITIALIZING RAG SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1: LOAD DOCUMENTS
    # --------------------------------------------------------

    print("\n📚 Loading documents...")

    documents = load_pdfs(PDF_PATHS)

    print(
        f"✅ Total pages loaded: "
        f"{len(documents)}"
    )

    # --------------------------------------------------------
    # STEP 2: SPLIT DOCUMENTS
    # --------------------------------------------------------

    print("\n✂️ Splitting documents...")

    chunks = split_documents(
        documents,
        min_chunk_size=100,
    )

    print(
        f"✅ Total chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # STEP 3: LOAD EMBEDDING MODEL
    # --------------------------------------------------------

    print("\n🧠 Loading embedding model...")

    embedding_model = create_embedding_model()

    print("✅ Embedding model loaded!")

    # --------------------------------------------------------
    # STEP 4: LOAD CHROMADB
    # --------------------------------------------------------

    print("\n🗄️ Loading ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DIR,
    )

    print("✅ ChromaDB loaded!")

    # --------------------------------------------------------
    # STEP 5: CREATE BM25
    # --------------------------------------------------------

    print("\n🔎 Creating BM25 index...")

    bm25 = create_bm25_retriever(chunks)

    print("✅ BM25 index created!")

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("✅ RAG SYSTEM READY")
    print("=" * 60)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    try:

        initialize_rag()

    except Exception as e:

        print("\n❌ RAG INITIALIZATION FAILED")
        print(f"Error: {e}")

        raise


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Hybrid RAG API is running",
        "status": "healthy",
        "docs": "/docs",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    if (
        vector_store is None
        or bm25 is None
        or chunks is None
    ):

        return {
            "status": "initializing",
        }

    return {
        "status": "healthy",
        "chunks": len(chunks),
        "top_k": TOP_K,
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if (
        vector_store is None
        or bm25 is None
        or chunks is None
    ):

        raise HTTPException(
            status_code=503,
            detail="RAG system is not ready.",
        )

    try:

        # ----------------------------------------------------
        # RUN RAG
        # ----------------------------------------------------

        answer, results, confidence = (
            answer_question(
                request.question,
                vector_store,
                bm25,
                chunks,
                top_k=request.top_k,
            )
        )

        # ----------------------------------------------------
        # BUILD SOURCES
        # ----------------------------------------------------

        sources = []

        for rank, result in enumerate(
            results,
            start=1,
        ):

            if isinstance(result, dict):

                document = result["document"]

                source = document.metadata.get(
                    "source",
                    "unknown",
                )

                page = document.metadata.get(
                    "page",
                    "unknown",
                )

                if isinstance(page, int):
                    page += 1

                sources.append(
                    {
                        "rank": rank,
                        "source": source,
                        "page": page,
                        "dense_score": result.get(
                            "dense_score",
                            0.0,
                        ),
                        "bm25_score": result.get(
                            "bm25_score",
                            0.0,
                        ),
                        "rrf_score": result.get(
                            "rrf_score",
                            0.0,
                        ),
                    }
                )

            else:

                document = result[0]
                score = result[1]

                source = document.metadata.get(
                    "source",
                    "unknown",
                )

                page = document.metadata.get(
                    "page",
                    "unknown",
                )

                if isinstance(page, int):
                    page += 1

                sources.append(
                    {
                        "rank": rank,
                        "source": source,
                        "page": page,
                        "score": float(score),
                    }
                )

        # ----------------------------------------------------
        # RETURN API RESPONSE
        # ----------------------------------------------------

        return {
            "question": request.question,

            "answer": answer,

            "confidence": {
                "retrieval_confidence": confidence[
                    "retrieval_confidence"
                ],
                "evidence_confidence": confidence[
                    "evidence_confidence"
                ],
                "overall_confidence": confidence[
                    "overall_confidence"
                ],
            },

            "retrieved_chunks": len(results),

            "sources": sources,
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )