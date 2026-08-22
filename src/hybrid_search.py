from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma

from .loader import load_pdfs
from .splitter import split_documents
from .embedded import create_embedding_model


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
RRF_K = 60


# ============================================================
# TOKENIZATION FOR BM25
# ============================================================

def tokenize(text):
    """Convert text into lowercase tokens."""

    return text.lower().split()


# ============================================================
# BUILD BM25
# ============================================================

def create_bm25(chunks):
    """Create BM25 index from document chunks."""

    corpus = [
        chunk.page_content
        for chunk in chunks
    ]

    tokenized_corpus = [
        tokenize(text)
        for text in corpus
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# ============================================================
# CHROMA VECTOR SEARCH
# ============================================================

def vector_search(
    vector_store,
    query,
    k=TOP_K
):
    """
    Retrieve documents using ChromaDB
    and keep their similarity scores.
    """

    results = vector_store.similarity_search_with_score(
        query,
        k=k
    )

    formatted_results = []

    for document, score in results:

        formatted_results.append(
            {
                "document": document,
                "dense_score": float(score),
            }
        )

    return formatted_results


# ============================================================
# BM25 SEARCH
# ============================================================

def bm25_search(
    bm25,
    chunks,
    query,
    k=TOP_K
):
    """
    Retrieve documents using BM25
    and keep their BM25 scores.
    """

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = []

    for index in ranked_indexes:

        results.append(
            {
                "document": chunks[index],
                "bm25_score": float(scores[index]),
            }
        )

    return results


# ============================================================
# CREATE DOCUMENT ID
# ============================================================

def document_id(document):
    """
    Create a unique identifier for a chunk.
    """

    source = document.metadata.get(
        "source",
        "unknown"
    )

    page = document.metadata.get(
        "page",
        "unknown"
    )

    content = document.page_content

    return (
        f"{source}|{page}|{content}"
    )


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    k=RRF_K
):
    """
    Combine vector and BM25 rankings using RRF.

    Keeps:
        - dense score
        - BM25 score
        - RRF score
    """

    scores = {}
    documents = {}
    dense_scores = {}
    bm25_scores = {}

    # --------------------------------------------------------
    # Vector search rankings
    # --------------------------------------------------------

    for rank, result in enumerate(
        vector_results,
        start=1
    ):

        document = result["document"]
        dense_score = result["dense_score"]

        doc_id = document_id(document)

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        documents[doc_id] = document

        dense_scores[doc_id] = dense_score

    # --------------------------------------------------------
    # BM25 rankings
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        document = result["document"]
        bm25_score = result["bm25_score"]

        doc_id = document_id(document)

        scores[doc_id] = (
            scores.get(doc_id, 0)
            + 1 / (k + rank)
        )

        documents[doc_id] = document

        bm25_scores[doc_id] = bm25_score

    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    ranked_results = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for doc_id, rrf_score in ranked_results:

        document = documents[doc_id]

        results.append(
            {
                "document": document,
                "dense_score": dense_scores.get(
                    doc_id,
                    0.0
                ),
                "bm25_score": bm25_scores.get(
                    doc_id,
                    0.0
                ),
                "rrf_score": float(rrf_score),
            }
        )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    vector_store,
    bm25,
    chunks,
    query,
    top_k=TOP_K
):
    """
    Perform hybrid retrieval.

    1. ChromaDB semantic search
    2. BM25 keyword search
    3. RRF score fusion
    4. Return all retrieval scores
    """

    print("\n" + "=" * 60)
    print("HYBRID SEARCH")
    print("=" * 60)

    print(f"\n Query: {query}")

    # --------------------------------------------------------
    # Vector retrieval
    # --------------------------------------------------------

    vector_results = vector_search(
        vector_store,
        query,
        k=top_k
    )

    print(
        f"\n Vector results: "
        f"{len(vector_results)}"
    )

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_results = bm25_search(
        bm25,
        chunks,
        query,
        k=top_k
    )

    print(
        f" BM25 results: "
        f"{len(bm25_results)}"
    )

    # --------------------------------------------------------
    # RRF fusion
    # --------------------------------------------------------

    fused_results = reciprocal_rank_fusion(
        vector_results,
        bm25_results
    )

    final_results = fused_results[:top_k]

    print(
        f" Hybrid results: "
        f"{len(final_results)}"
    )

    return final_results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):

    print("\n" + "=" * 60)
    print("HYBRID SEARCH RESULTS")
    print("=" * 60)

    for rank, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        dense_score = result["dense_score"]
        bm25_score = result["bm25_score"]
        rrf_score = result["rrf_score"]

        print("\n" + "-" * 60)
        print(f"RESULT {rank}")
        print("-" * 60)

        print(
            f"\n Dense Score: "
            f"{dense_score:.6f}"
        )

        print(
            f" BM25 Score: "
            f"{bm25_score:.6f}"
        )

        print(
            f" RRF Score: "
            f"{rrf_score:.6f}"
        )

        print("\n Source:")

        print(
            document.metadata.get(
                "source",
                "unknown"
            )
        )

        print("\n Page:")

        print(
            document.metadata.get(
                "page",
                "unknown"
            )
        )

        print("\n Content:")

        print(
            document.page_content[:500]
        )


# ============================================================
# VALIDATION
# ============================================================

def validate_hybrid(
    chunks,
    vector_results,
    bm25_results,
    hybrid_results
):
    """Validate the hybrid retrieval pipeline."""

    print("\n" + "=" * 60)
    print("HYBRID SEARCH VALIDATION")
    print("=" * 60)

    checks = {

        "Chunks available":
            len(chunks) > 0,

        "Expected chunks":
            len(chunks) == 3107,

        "Vector search returned results":
            len(vector_results) > 0,

        "BM25 returned results":
            len(bm25_results) > 0,

        "Hybrid search returned results":
            len(hybrid_results) > 0,

        "Dense scores available":
            all(
                "dense_score" in result
                for result in hybrid_results
            ),

        "BM25 scores available":
            all(
                "bm25_score" in result
                for result in hybrid_results
            ),

        "RRF scores available":
            all(
                "rrf_score" in result
                for result in hybrid_results
            ),

        "Hybrid results contain metadata":
            all(
                result["document"].metadata
                for result in hybrid_results
            ),

        "Hybrid results contain content":
            all(
                len(result["document"].page_content) > 0
                for result in hybrid_results
            ),
    }

    for check, result in checks.items():

        status = "✅" if result else "❌"

        print(
            f"{status} {check}"
        )

    return all(checks.values())


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        # ====================================================
        # STEP 1: LOAD DOCUMENTS
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 1: LOADING DOCUMENTS")
        print("=" * 60)

        documents = load_pdfs(
            PDF_PATHS
        )

        print(
            f"\n Total pages: "
            f"{len(documents)}"
        )

        # ====================================================
        # STEP 2: CREATE CHUNKS
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 2: CREATING CHUNKS")
        print("=" * 60)

        chunks = split_documents(
            documents,
            min_chunk_size=100
        )

        print(
            f"\n Total chunks: "
            f"{len(chunks)}"
        )

        # ====================================================
        # STEP 3: LOAD EMBEDDING MODEL
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 3: LOADING EMBEDDING MODEL")
        print("=" * 60)

        embedding_model = (
            create_embedding_model()
        )

        # ====================================================
        # STEP 4: LOAD CHROMADB
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 4: LOADING CHROMADB")
        print("=" * 60)

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )

        print(
            " ChromaDB loaded successfully!"
        )

        # ====================================================
        # STEP 5: BUILD BM25
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 5: BUILDING BM25")
        print("=" * 60)

        bm25 = create_bm25(chunks)

        print(
            " BM25 index created!"
        )

        # ====================================================
        # STEP 6: RUN HYBRID SEARCH
        # ====================================================

        query = "What is machine learning?"

        hybrid_results = hybrid_search(
            vector_store,
            bm25,
            chunks,
            query,
            top_k=TOP_K
        )

        # ====================================================
        # STEP 7: DISPLAY RESULTS
        # ====================================================

        display_results(
            hybrid_results
        )

        # ====================================================
        # STEP 8: VALIDATE
        # ====================================================

        vector_results = vector_search(
            vector_store,
            query,
            k=TOP_K
        )

        bm25_results = bm25_search(
            bm25,
            chunks,
            query,
            k=TOP_K
        )

        is_valid = validate_hybrid(
            chunks,
            vector_results,
            bm25_results,
            hybrid_results
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        if is_valid:

            print("\n" + "=" * 60)
            print(
                " PHASE 7: HYBRID SEARCH PASSED"
            )
            print("=" * 60)

            print(
                "\n ChromaDB + BM25 "
                "hybrid retrieval is working!"
            )

        else:

            print("\n" + "=" * 60)
            print(
                " PHASE 7: VALIDATION FAILED"
            )
            print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print(" HYBRID SEARCH ERROR")
        print("=" * 60)

        print(
            f"\nError: {str(e)}"
        )

        raise
