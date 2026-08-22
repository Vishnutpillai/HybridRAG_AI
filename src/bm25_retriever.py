from rank_bm25 import BM25Okapi

from .loader import load_pdfs
from .splitter import split_documents


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATHS = [
    "data/raw/Machine_Learning.pdf",
    "data/raw/Deep_Learning.pdf",
]

TOP_K = 5


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    """
    Convert text into tokens for BM25.

    Example:
        "What is machine learning?"
        ->
        ["what", "is", "machine", "learning"]
    """

    return text.lower().split()


# ============================================================
# CREATE BM25 RETRIEVER
# ============================================================

def create_bm25_retriever(chunks):
    """
    Create a BM25 index from document chunks.

    Args:
        chunks: List of LangChain Document objects

    Returns:
        BM25Okapi object
        Tokenized corpus
    """

    if not chunks:
        raise ValueError(" No chunks provided!")

    print("\n Creating BM25 index...")

    # Extract text from every chunk
    corpus = [
        chunk.page_content
        for chunk in chunks
    ]

    # Tokenize every chunk
    tokenized_corpus = [
        tokenize(text)
        for text in corpus
    ]

    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)

    print(" BM25 index created!")
    print(f" Documents indexed: {len(chunks)}")

    return bm25


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
    Search chunks using BM25 keyword retrieval.

    Args:
        bm25: BM25Okapi index
        chunks: Original document chunks
        query: User question
        k: Number of results

    Returns:
        List of (document, score) tuples
    """

    if not query.strip():
        raise ValueError(" Query cannot be empty!")

    print("\n" + "=" * 60)
    print("BM25 SEARCH")
    print("=" * 60)

    print(f"\n Query: {query}")

    # Tokenize query
    tokenized_query = tokenize(query)

    # Get BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Get indexes of highest scoring documents
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    # Create results
    results = []

    for index in ranked_indexes:
        results.append(
            (
                chunks[index],
                float(scores[index])
            )
        )

    print(
        f"\n Retrieved documents: "
        f"{len(results)}"
    )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):
    """
    Display BM25 search results.
    """

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 60)
        print(f"RESULT {rank}")
        print("-" * 60)

        print(
            f"\n BM25 Score: "
            f"{score:.4f}"
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
# VALIDATE BM25
# ============================================================

def validate_bm25(
    chunks,
    bm25,
    results
):
    """
    Validate the BM25 retrieval pipeline.
    """

    print("\n" + "=" * 60)
    print("BM25 VALIDATION")
    print("=" * 60)

    checks = {
        "Chunks available":
            len(chunks) > 0,

        "Expected chunks":
            len(chunks) == 3107,

        "BM25 index created":
            bm25 is not None,

        "Search returned results":
            len(results) > 0,

        "Top result has content":
            len(results[0][0].page_content) > 0,

        "Top result has metadata":
            bool(results[0][0].metadata),

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
        # STEP 1: LOAD PDF DOCUMENTS
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 1: LOADING DOCUMENTS")
        print("=" * 60)

        documents = load_pdfs(
            PDF_PATHS
        )

        print(
            f"\n Total pages loaded: "
            f"{len(documents)}"
        )

        # ====================================================
        # STEP 2: SPLIT DOCUMENTS
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 2: SPLITTING DOCUMENTS")
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
        # STEP 3: CREATE BM25 INDEX
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 3: BUILDING BM25 INDEX")
        print("=" * 60)

        bm25, tokenized_corpus = (
            create_bm25_retriever(chunks)
        )

        # ====================================================
        # STEP 4: TEST SEARCH
        # ====================================================

        query = "What is machine learning?"

        results = bm25_search(
            bm25,
            chunks,
            query,
            k=TOP_K
        )

        # ====================================================
        # STEP 5: DISPLAY RESULTS
        # ====================================================

        display_results(results)

        # ====================================================
        # STEP 6: VALIDATE
        # ====================================================

        is_valid = validate_bm25(
            chunks,
            bm25,
            results
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        if is_valid:

            print("\n" + "=" * 60)
            print(" PHASE 6: BM25 RETRIEVAL PASSED")
            print("=" * 60)

            print(
                "\n BM25 keyword retrieval "
                "is working correctly!"
            )

            print(
                " Indexed chunks: "
                f"{len(chunks)}"
            )

            print(
                " Retrieved results: "
                f"{len(results)}"
            )

        else:

            print("\n" + "=" * 60)
            print(" PHASE 6: BM25 VALIDATION FAILED")
            print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print(" PHASE 6: BM25 ERROR")
        print("=" * 60)

        print(
            f"\nError: {str(e)}"
        )

        raise

