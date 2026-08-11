from hybrid_search import (
    load_pdfs,
    split_documents,
    create_embedding_model,
    Chroma,
    create_bm25,
    hybrid_search,
)

from groq_con import ask_groq


PDF_PATHS = [
    "data/raw/Machine_Learning.pdf",
    "data/raw/Deep_Learning.pdf",
]

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "rag_documents"

TOP_K = 5


def build_rag_prompt(query, results):
    """
    Build a prompt using the retrieved document chunks.
    """

    context_parts = []

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "unknown"
        )

        page = document.metadata.get(
            "page",
            "unknown"
        )

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


def answer_question(
    query,
    vector_store,
    bm25,
    chunks,
    top_k=TOP_K
):
    """
    Complete RAG pipeline.
    """

    print("\n" + "=" * 60)
    print("RAG QUESTION ANSWERING")
    print("=" * 60)

    print(f"\nQuestion: {query}")

    results = hybrid_search(
        vector_store,
        bm25,
        chunks,
        query,
        top_k=top_k
    )

    if not results:
        raise ValueError(
            "No documents retrieved."
        )

    print(
        f"\nRetrieved chunks: {len(results)}"
    )

    prompt = build_rag_prompt(
        query,
        results
    )

    print("\nSending context to Groq...")

    answer = ask_groq(prompt)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print("\n" + answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for rank, (document, score) in enumerate(
        results,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "unknown"
        )

        page = document.metadata.get(
            "page",
            "unknown"
        )

        print(
            f"{rank}. {source} | Page: {page}"
        )

    return answer, results


if __name__ == "__main__":

    try:

        print("\nLoading documents...")

        documents = load_pdfs(
            PDF_PATHS
        )

        print(
            f"Total pages: {len(documents)}"
        )

        chunks = split_documents(
            documents,
            min_chunk_size=100
        )

        print(
            f"Total chunks: {len(chunks)}"
        )

        print("\nLoading embedding model...")

        embedding_model = (
            create_embedding_model()
        )

        print("\nLoading ChromaDB...")

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_model,
            persist_directory=CHROMA_DIR,
        )

        print(
            "ChromaDB loaded successfully!"
        )

        print("\nCreating BM25 index...")

        bm25 = create_bm25(
            chunks
        )

        print(
            "BM25 index created!"
        )

        question = input(
            "\nEnter your question: "
        ).strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        answer, results = answer_question(
            question,
            vector_store,
            bm25,
            chunks,
            top_k=TOP_K
        )

        print("\n" + "=" * 60)
        print("RAG VALIDATION")
        print("=" * 60)

        if answer and results:

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