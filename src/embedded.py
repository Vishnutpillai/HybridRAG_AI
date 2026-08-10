from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CREATE EMBEDDING MODEL
# ============================================================

def create_embedding_model():
    """
    Create and return the Hugging Face embedding model.

    Model:
        BAAI/bge-small-en-v1.5

    Output:
        384-dimensional embedding vectors
    """

    print("\n🔄 Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("✅ Embedding model loaded successfully!")

    return embedding_model


# ============================================================
# TEST EMBEDDING MODEL
# ============================================================

if __name__ == "__main__":

    from loader import load_pdfs
    from splitter import split_documents

    # ========================================================
    # PDF FILES
    # ========================================================

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

        print(f"\n📄 Total pages loaded: {len(documents)}")

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

        print(f"\n📝 Total chunks: {len(chunks)}")

        # ====================================================
        # STEP 3: CREATE EMBEDDING MODEL
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 3: CREATING EMBEDDING MODEL")
        print("=" * 60)

        embedding_model = create_embedding_model()

        # ====================================================
        # STEP 4: GET FIRST CHUNK
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 4: SELECTING TEST CHUNK")
        print("=" * 60)

        first_chunk = chunks[0]

        print(
            f"\n📌 Chunk length: "
            f"{len(first_chunk.page_content)} characters"
        )

        print(
            f"📄 Source: "
            f"{first_chunk.metadata.get('source')}"
        )

        print(
            f"📖 Page: "
            f"{first_chunk.metadata.get('page')}"
        )

        # ====================================================
        # STEP 5: CREATE EMBEDDING
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 5: CREATING EMBEDDING")
        print("=" * 60)

        vector = embedding_model.embed_query(
            first_chunk.page_content
        )

        # ====================================================
        # STEP 6: EMBEDDING RESULTS
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 6: EMBEDDING RESULTS")
        print("=" * 60)

        print(
            f"\n🔢 Vector type: "
            f"{type(vector)}"
        )

        print(
            f"📐 Vector dimensions: "
            f"{len(vector)}"
        )

        print("\n🔢 First 10 vector values:")

        for i, value in enumerate(vector[:10], start=1):
            print(f"  {i}. {value:.6f}")

        # ====================================================
        # STEP 7: CHUNK PREVIEW
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 7: SOURCE CHUNK")
        print("=" * 60)

        print("\n" + first_chunk.page_content[:500])

        # ====================================================
        # STEP 8: METADATA
        # ====================================================

        print("\n" + "=" * 60)
        print("STEP 8: CHUNK METADATA")
        print("=" * 60)

        print(
            f"\nSource: "
            f"{first_chunk.metadata.get('source')}"
        )

        print(
            f"Page: "
            f"{first_chunk.metadata.get('page')}"
        )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        print("\n" + "=" * 60)
        print("EMBEDDING VALIDATION")
        print("=" * 60)

        checks = {
            "Embedding model loaded": embedding_model is not None,
            "Chunks available": len(chunks) > 0,
            "Vector created": vector is not None,
            "Vector dimension = 384": len(vector) == 384,
            "Vector is not empty": len(vector) > 0,
        }

        for check, result in checks.items():

            status = "✅" if result else "❌"

            print(f"{status} {check}")

        # ====================================================
        # FINAL STATUS
        # ====================================================

        if all(checks.values()):

            print("\n" + "=" * 60)
            print("✅ PHASE 4: EMBEDDING TEST PASSED")
            print("=" * 60)

            print(
                "\nYour pipeline is ready for "
                "vector database storage."
            )

        else:

            print("\n" + "=" * 60)
            print("❌ PHASE 4: EMBEDDING TEST FAILED")
            print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print("❌ EMBEDDING ERROR")
        print("=" * 60)

        print(f"\nError: {str(e)}")

        raise