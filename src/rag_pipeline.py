# ============================================================
# RAG PIPELINE
# ============================================================

from .loader import load_pdfs
from .splitter import split_documents
from .embedded import create_embedding_model
from .vectorstore import Chroma
from .bm25_retriever import create_bm25_retriever
from .hybrid_search import hybrid_search

from .groq_con import ask_groq

from .confidence import (
    calculate_retrieval_confidence,
    calculate_evidence_confidence,
    calculate_overall_confidence,
)


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
# HELPER: GET DOCUMENT FROM RESULT
# ============================================================

def get_document(result):
    """
    Extract the LangChain Document from a hybrid-search result.

    Supports:
    - dictionary-based hybrid results
    - tuple-based hybrid results
    """

    if isinstance(result, dict):
        return result["document"]

    return result[0]


# ============================================================
# BUILD RAG PROMPT
# ============================================================

def build_rag_prompt(query, results):
    """
    Build a prompt using retrieved document chunks.
    """

    context_parts = []

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = get_document(result)

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
            "unknown",
        )

        # Convert zero-based PDF page number
        # to human-readable page number
        if isinstance(page, int):
            page = page + 1

        content = document.page_content

        context_parts.append(
            f"""
SOURCE {rank}

Source: {source}
Page: {page}

Content:
{content}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the
provided context from the documents.

If the answer cannot be found in the provided
context, say:

"I could not find the answer in the provided documents."

Do not invent facts.
Do not use outside knowledge.

---------------- CONTEXT ----------------

{context}

-------------- END CONTEXT --------------

QUESTION:
{query}

Answer clearly and concisely.
"""

    return prompt


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):
    """
    Display retrieved document sources.
    """

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = get_document(result)

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page",
            "unknown",
        )

        if isinstance(page, int):
            page = page + 1

        # ----------------------------------------------------
        # Dictionary-based hybrid result
        # ----------------------------------------------------

        if isinstance(result, dict):

            dense_score = result.get(
                "dense_score",
                0.0,
            )

            bm25_score = result.get(
                "bm25_score",
                0.0,
            )

            rrf_score = result.get(
                "rrf_score",
                0.0,
            )

            print(
                f"\n{rank}. {source}"
            )

            print(
                f"   Page: {page}"
            )

            print(
                f"   Dense Score: {dense_score:.6f}"
            )

            print(
                f"   BM25 Score: {bm25_score:.6f}"
            )

            print(
                f"   RRF Score: {rrf_score:.6f}"
            )

        # ----------------------------------------------------
        # Tuple-based result
        # ----------------------------------------------------

        else:

            score = result[1]

            print(
                f"{rank}. {source} | "
                f"Page: {page} | "
                f"Score: {score:.6f}"
            )


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def answer_question(
    query,
    vector_store,
    bm25,
    chunks,
    top_k=TOP_K,
):
    """
    Complete RAG pipeline.

    Question
        ↓
    Hybrid Search
        ↓
    Confidence Calculation
        ↓
    Retrieved Context
        ↓
    Groq
        ↓
    Answer + Confidence + Sources
    """

    print("\n" + "=" * 60)
    print("RAG QUESTION ANSWERING")
    print("=" * 60)

    print(
        f"\nQuestion: {query}"
    )
    

    # ========================================================
    # STEP 1: HYBRID SEARCH
    # ========================================================

    results = hybrid_search(
        vector_store,
        bm25,
        chunks,
        query,
        top_k=top_k,
    )

    if not results:

        raise ValueError(
            "No documents retrieved."
        )

    print(
        f"\nRetrieved chunks: {len(results)}"
    )

    # ========================================================
    # STEP 2: RETRIEVAL CONFIDENCE
    # ========================================================

    retrieval_confidence = (
        calculate_retrieval_confidence(
            results
        )
    )

    print(
        f"\nRetrieval confidence: "
        f"{retrieval_confidence:.2f}"
    )

    # ========================================================
    # STEP 3: EVIDENCE CONFIDENCE
    # ========================================================

    evidence_confidence = (
        calculate_evidence_confidence(
            results
        )
    )

    print(
        f"Evidence confidence: "
        f"{evidence_confidence:.2f}"
    )

    # ========================================================
    # STEP 4: OVERALL CONFIDENCE
    # ========================================================

    confidence = (
        calculate_overall_confidence(
            retrieval_confidence,
            evidence_confidence,
        )
    )

    print(
        f"Overall confidence: "
        f"{confidence['overall_confidence']:.2f}"
    )

    # ========================================================
    # STEP 5: BUILD RAG PROMPT
    # ========================================================

    prompt = build_rag_prompt(
        query,
        results,
    )

    # ========================================================
    # STEP 6: SEND CONTEXT TO GROQ
    # ========================================================

    print(
        "\nSending context to Groq..."
    )

    answer = ask_groq(
        prompt
    )

    # ========================================================
    # STEP 7: DISPLAY ANSWER
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(
        "\n" + answer
    )

    # ========================================================
    # STEP 8: DISPLAY CONFIDENCE
    # ========================================================

    print("\n" + "=" * 60)
    print("CONFIDENCE SCORES")
    print("=" * 60)

    print(
        f"\nRetrieval confidence: "
        f"{confidence['retrieval_confidence']:.2f}"
    )

    print(
        f"Evidence confidence: "
        f"{confidence['evidence_confidence']:.2f}"
    )

    print(
        f"Overall confidence: "
        f"{confidence['overall_confidence']:.2f}"
    )

    print(
        "\n⚠️ These are heuristic confidence scores, "
        "not calibrated probabilities."
    )

    # ========================================================
    # STEP 9: DISPLAY SOURCES
    # ========================================================

    display_sources(
        results
    )

    return (
        answer,
        results,
        confidence,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        # ====================================================
        # STEP 1: LOAD DOCUMENTS
        # ====================================================

        print(
            "\nLoading documents..."
        )

        documents = load_pdfs(
            PDF_PATHS
        )

        print(
            f"Total pages: "
            f"{len(documents)}"
        )

        # ====================================================
        # STEP 2: CREATE CHUNKS
        # ====================================================

        chunks = split_documents(
            documents,
            min_chunk_size=100,
        )

        print(
            f"Total chunks: "
            f"{len(chunks)}"
        )

        # ====================================================
        # STEP 3: LOAD EMBEDDING MODEL
        # ====================================================

        print(
            "\nLoading embedding model..."
        )

        embedding_model = (
            create_embedding_model()
        )

        # ====================================================
        # STEP 4: LOAD CHROMADB
        # ====================================================

        print(
            "\nLoading ChromaDB..."
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )

        print(
            "ChromaDB loaded successfully!"
        )

        # ====================================================
        # STEP 5: CREATE BM25 INDEX
        # ====================================================

        print(
            "\nCreating BM25 index..."
        )

        bm25 = create_bm25_retriever(
            chunks
        )

        print(
            "BM25 index created!"
        )

        # ====================================================
        # STEP 6: ASK QUESTION
        # ====================================================

        question = input(
            "\nEnter your question: "
        ).strip()

        if not question:

            raise ValueError(
                "Question cannot be empty."
            )

        # ====================================================
        # STEP 7: RUN RAG PIPELINE
        # ====================================================

        answer, results, confidence = (
            answer_question(
                question,
                vector_store,
                bm25,
                chunks,
                top_k=TOP_K,
            )
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        print("\n" + "=" * 60)
        print("RAG VALIDATION")
        print("=" * 60)

        if (
            answer
            and results
            and confidence
        ):

            print(
                "Documents loaded       ✅"
            )

            print(
                "Chunks created         ✅"
            )

            print(
                "ChromaDB search        ✅"
            )

            print(
                "BM25 search            ✅"
            )

            print(
                "Hybrid search          ✅"
            )

            print(
                "Confidence calculation ✅"
            )

            print(
                "Groq generation        ✅"
            )

            print(
                "Final answer generated ✅"
            )

            print(
                "\n🎉 COMPLETE RAG PIPELINE WORKING!"
            )

        else:

            print(
                "❌ RAG pipeline validation failed."
            )

    except Exception as e:

        print("\n" + "=" * 60)
        print("RAG PIPELINE ERROR")
        print("=" * 60)

        print(
            f"\nError: {str(e)}"
        )

        raise