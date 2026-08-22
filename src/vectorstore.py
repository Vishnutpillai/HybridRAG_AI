from pathlib import Path

from langchain_chroma import Chroma

from .loader import load_pdfs
from .splitter import split_documents
from .embedded import create_embedding_model


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag_documents"


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks, embedding_model):
    """
    Create and persist a ChromaDB vector store.

    Args:
        chunks: List of LangChain Document objects
        embedding_model: Hugging Face embedding model

    Returns:
        Chroma vector store
    """

    if not chunks:
        raise ValueError(" No chunks provided!")

    print("\n Creating ChromaDB vector store...")

    # Create database directory
    Path(CHROMA_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    # Create Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    print(" ChromaDB vector store created!")
    print(f" Collection: {COLLECTION_NAME}")
    print(f" Database: {CHROMA_DIR}")
    print(f" Documents stored: {len(chunks)}")

    return vector_store


# ============================================================
# TEST VECTOR SEARCH
# ============================================================

def test_similarity_search(vector_store, query, k=5):
    """
    Test similarity search using a natural-language query.

    Args:
        vector_store: Chroma vector store
        query: User question
        k: Number of results to retrieve

    Returns:
        Retrieved documents
    """

    print("\n" + "=" * 60)
    print("VECTOR SEARCH TEST")
    print("=" * 60)

    print(f"\n Query: {query}")

    results = vector_store.similarity_search(
        query,
        k=k
    )

    print(f"\n Retrieved documents: {len(results)}")

    for index, document in enumerate(results, start=1):

        print("\n" + "-" * 60)
        print(f"RESULT {index}")
        print("-" * 60)

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

    return results


# ============================================================
# DATABASE COUNT
# ============================================================

def get_collection_count(vector_store):
    """
    Return the number of documents stored in ChromaDB.
    """

    collection = vector_store._collection

    count = collection.count()

    return count


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    pdf_paths = [
        "data/raw/Machine_Learning.pdf",
        "data/raw/Deep_Learning.pdf",
    ]

    try:

        # ====================================================
        # STEP 1: LOAD DOCUMENTS
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 1: LOADING DOCUMENTS")
        print("=" * 60)

        documents = load_pdfs(pdf_paths)

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
        # STEP 3: CREATE EMBEDDING MODEL
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 3: LOADING EMBEDDING MODEL")
        print("=" * 60)

        embedding_model = create_embedding_model()

        # ====================================================
        # STEP 4: CREATE CHROMA VECTOR STORE
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 4: CREATING VECTOR STORE")
        print("=" * 60)

        vector_store = create_vector_store(
            chunks,
            embedding_model
        )

        # ====================================================
        # STEP 5: VERIFY DATABASE
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 5: VERIFYING CHROMADB")
        print("=" * 60)

        stored_count = get_collection_count(
            vector_store
        )

        print(
            f"\n Documents in ChromaDB: "
            f"{stored_count}"
        )

        # ====================================================
        # STEP 6: TEST SEARCH
        # ====================================================

        query = (
            "What is machine learning?"
        )

        results = test_similarity_search(
            vector_store,
            query,
            k=5
        )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        print("\n" + "=" * 60)
        print("VECTOR STORE VALIDATION")
        print("=" * 60)

        checks = {
            "Chunks created":
                len(chunks) > 0,

            "Embedding model loaded":
                embedding_model is not None,

            "ChromaDB created":
                vector_store is not None,

            "Documents stored":
                stored_count > 0,

            "Expected chunk count":
                stored_count == len(chunks),

            "Search returned results":
                len(results) > 0,
        }

        for check, result in checks.items():

            status = "✅" if result else "❌"

            print(
                f"{status} {check}"
            )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        if all(checks.values()):

            print("\n" + "=" * 60)
            print(" PHASE 5: VECTOR DATABASE PASSED")
            print("=" * 60)

            print(
                "\n ChromaDB is working correctly!"
            )

            print(
                f" Stored {stored_count} chunks"
            )

            print(
                "🔎 Similarity search is working!"
            )

        else:

            print("\n" + "=" * 60)
            print(" PHASE 5: VALIDATION FAILED")
            print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print(" VECTOR STORE ERROR")
        print("=" * 60)

        print(
            f"\nError: {str(e)}"
        )

        raise