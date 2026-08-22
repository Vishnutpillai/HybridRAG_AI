from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
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

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

PDF_PATHS = [
    str(RAW_DIR / "Machine_Learning.pdf"),
    str(RAW_DIR / "Deep_Learning.pdf"),
]

COLLECTION_NAME = "rag_documents"
TOP_K = 5


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Hybrid RAG API",
    description=(
        "Production-style Retrieval-Augmented Generation API "
        "using Vector Search + BM25 + Hybrid RRF Search + Groq."
    ),
    version="2.0.0",
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
        description="Number of chunks to retrieve",
    )


# ============================================================
# GLOBAL RAG COMPONENTS
# ============================================================

vector_store = None
bm25 = None
chunks = None
embedding_model = None


# ============================================================
# INITIALIZE RAG
# ============================================================

def initialize_rag():

    global vector_store
    global bm25
    global chunks
    global embedding_model

    print("\n" + "=" * 60)
    print("INITIALIZING RAG SYSTEM")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DOCUMENTS
    # --------------------------------------------------------

    print("\n Loading documents...")

    documents = load_pdfs(PDF_PATHS)

    print(
        f" Total pages loaded: {len(documents)}"
    )

    if not documents:
        raise ValueError(
            "No documents were loaded."
        )

    # --------------------------------------------------------
    # SPLIT DOCUMENTS
    # --------------------------------------------------------

    print("\n Splitting documents...")

    chunks = split_documents(
        documents,
        min_chunk_size=100,
    )

    print(
        f" Total chunks: {len(chunks)}"
    )

    if not chunks:
        raise ValueError(
            "No chunks were created."
        )

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    print("\n Loading embedding model...")

    embedding_model = create_embedding_model()

    print(" Embedding model loaded!")

    # --------------------------------------------------------
    # CHROMADB
    # --------------------------------------------------------

    print("\n Loading ChromaDB...")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )

    print(" ChromaDB loaded!")

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print("\n🔎 Creating BM25 index...")

    bm25 = create_bm25_retriever(chunks)

    print(" BM25 index created!")

    print("\n" + "=" * 60)
    print(" RAG SYSTEM READY")
    print("=" * 60)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    try:
        RAW_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        CHROMA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        initialize_rag()

    except Exception as e:

        print("\n RAG INITIALIZATION FAILED")
        print(f"Error: {e}")

        raise


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():

    return {
        "name": "Hybrid RAG API",
        "version": "2.0.0",
        "status": "healthy",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    tags=["System"],
)
def health():

    ready = (
        vector_store is not None
        and bm25 is not None
        and chunks is not None
    )

    return {
        "status": "healthy" if ready else "initializing",
        "ready": ready,
        "chunks": len(chunks) if chunks else 0,
        "top_k": TOP_K,
    }


# ============================================================
# SOURCE METADATA HELPER
# ============================================================

def build_sources(results):

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
                    "dense_score": float(
                        result.get(
                            "dense_score",
                            0.0,
                        )
                    ),
                    "bm25_score": float(
                        result.get(
                            "bm25_score",
                            0.0,
                        )
                    ),
                    "rrf_score": float(
                        result.get(
                            "rrf_score",
                            0.0,
                        )
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

    return sources


# ============================================================
# ASK QUESTION CORE
# ============================================================

def run_question(request: QuestionRequest):

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

        answer, results, confidence = (
            answer_question(
                request.question,
                vector_store,
                bm25,
                chunks,
                top_k=request.top_k,
            )
        )

        sources = build_sources(results)

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

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ============================================================
# V1 ASK
# ============================================================

@app.post(
    "/v1/ask",
    tags=["RAG"],
    summary="Ask a question",
)
def ask_v1(request: QuestionRequest):

    return run_question(request)


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

@app.post(
    "/ask",
    tags=["Legacy"],
    include_in_schema=False,
)
def ask_legacy(request: QuestionRequest):

    return run_question(request)


# ============================================================
# LIST INDEXED DOCUMENTS
# ============================================================

@app.get(
    "/v1/documents",
    tags=["Documents"],
    summary="List indexed documents",
)
def list_documents():

    documents = []

    for pdf_path in PDF_PATHS:

        path = Path(pdf_path)

        if path.exists():

            documents.append(
                {
                    "filename": path.name,
                    "path": str(path),
                    "type": "PDF",
                    "indexed": True,
                }
            )

    # Also include PDFs uploaded through /v1/ingest
    known_paths = {
        item["path"]
        for item in documents
    }

    for path in RAW_DIR.glob("*.pdf"):

        path_string = str(path)

        if path_string not in known_paths:

            documents.append(
                {
                    "filename": path.name,
                    "path": path_string,
                    "type": "PDF",
                    "indexed": True,
                }
            )

    return {
        "count": len(documents),
        "documents": documents,
    }


# ============================================================
# INGEST DOCUMENT
# ============================================================

@app.post(
    "/v1/ingest",
    tags=["Documents"],
    summary="Upload and index a PDF document",
)
async def ingest_document(
    file: UploadFile = File(...),
):

    global chunks
    global bm25

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    # --------------------------------------------------------
    # SAVE PDF
    # --------------------------------------------------------

    safe_filename = Path(
        file.filename
    ).name

    destination = RAW_DIR / safe_filename

    try:

        content = await file.read()

        if not content:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        destination.write_bytes(content)

        print(
            f"\n New document uploaded: "
            f"{safe_filename}"
        )

        # ----------------------------------------------------
        # LOAD NEW DOCUMENT
        # ----------------------------------------------------

        new_documents = load_pdfs(
            [str(destination)]
        )

        if not new_documents:

            raise HTTPException(
                status_code=400,
                detail="Could not read the uploaded PDF.",
            )

        # ----------------------------------------------------
        # SPLIT
        # ----------------------------------------------------

        new_chunks = split_documents(
            new_documents,
            min_chunk_size=100,
        )

        if not new_chunks:

            raise HTTPException(
                status_code=400,
                detail="No usable chunks found in PDF.",
            )

        # ----------------------------------------------------
        # ADD TO VECTOR STORE
        # ----------------------------------------------------

        vector_store.add_documents(
            new_chunks
        )

        # ----------------------------------------------------
        # UPDATE GLOBAL CHUNKS
        # ----------------------------------------------------

        chunks.extend(new_chunks)

        # ----------------------------------------------------
        # REBUILD BM25
        # ----------------------------------------------------

        bm25 = create_bm25_retriever(
            chunks
        )

        print(
            f" Indexed {len(new_chunks)} new chunks"
        )

        return {
            "status": "success",
            "message": "Document indexed successfully.",
            "filename": safe_filename,
            "pages": len(new_documents),
            "chunks_added": len(new_chunks),
            "total_chunks": len(chunks),
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )