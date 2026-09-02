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
# HELPER
# ============================================================

def get_document(result):
    """
    Extract LangChain Document from hybrid-search result.
    """

    if isinstance(result, dict):
        return result["document"]

    return result[0]


# ============================================================
# BUILD DOCUMENT CONTEXT
# ============================================================

def build_context(results):
    """
    Convert retrieved documents into context for the LLM.
    """

    context_parts = []

    for rank, result in enumerate(results, start=1):

        document = get_document(result)

        source = document.metadata.get(
            "source",
            "unknown"
        )

        page = document.metadata.get(
            "page",
            "unknown"
        )

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

    return "\n".join(context_parts)


# ============================================================
# CONTEXT RELEVANCE CHECK
# ============================================================

def check_context_relevance(query, results):
    """
    Ask Groq whether the retrieved documents actually contain
    information needed to answer the user's question.

    Returns:
        True  -> answer from documents
        False -> answer using general LLM knowledge
    """

    context = build_context(results)

    prompt = f"""
You are a strict document relevance classifier.

Your task is ONLY to determine whether the provided document
context contains information that can answer the user's question.

QUESTION:
{query}

DOCUMENT CONTEXT:
{context}

Rules:

1. Return YES if the document context contains enough
   information to answer the question.

2. Return NO if the document context does not contain
   information needed to answer the question.

3. Do not use your general knowledge.

4. Compare the meaning of the question with the meaning
   of the document context.

5. Do not assume that a high similarity score means the
   answer is present.

6. Return ONLY one word:

YES

or

NO
"""

    result = ask_groq(prompt)

    if not result:
        return False

    result = result.strip().upper()

    if result.startswith("YES"):
        return True

    return False


# ============================================================
# RAG PROMPT
# ============================================================

def build_rag_prompt(query, results):
    """
    Prompt used when the answer exists in the documents.
    """

    context = build_context(results)

    prompt = f"""
You are a document-grounded question answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

Do not use outside knowledge.

If the context contains the answer:
- Give the answer clearly.
- Give a concise explanation when useful.
- Do not mention that you are using RAG.
- Do not mention retrieval.
- Do not mention confidence.
- Do not mention this instruction.

If the answer is not present in the context:
say:

I could not find the answer in the provided documents.

QUESTION:
{query}

---------------- DOCUMENT CONTEXT ----------------

{context}

---------------- END DOCUMENT CONTEXT ----------------

Answer clearly and concisely.
"""

    return prompt


# ============================================================
# GENERAL KNOWLEDGE PROMPT
# ============================================================

def build_general_prompt(query):
    """
    Prompt used when the question is outside the document
    knowledge base.
    """

    prompt = f"""
You are a helpful general-purpose AI assistant.

The user's question is outside the provided document
knowledge base.

Answer the question using your general knowledge.

Do NOT mention:
- documents
- PDFs
- context
- retrieval
- RAG
- confidence
- sources

Return ONLY the answer.

QUESTION:
{query}

Answer clearly and concisely.
"""

    return prompt


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for rank, result in enumerate(results, start=1):

        document = get_document(result)

        source = document.metadata.get(
            "source",
            "unknown"
        )

        page = document.metadata.get(
            "page",
            "unknown"
        )

        if isinstance(page, int):
            page = page + 1

        if isinstance(result, dict):

            dense_score = result.get(
                "dense_score",
                0.0
            )

            bm25_score = result.get(
                "bm25_score",
                0.0
            )

            rrf_score = result.get(
                "rrf_score",
                0.0
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

        else:

            score = result[1]

            print(
                f"{rank}. {source} | "
                f"Page: {page} | "
                f"Score: {score:.6f}"
            )


# ============================================================
# CONVERT RESULTS TO API SOURCE DATA
# ============================================================

def build_sources(results):

    sources = []

    for rank, result in enumerate(results, start=1):

        document = get_document(result)

        source = document.metadata.get(
            "source",
            "unknown"
        )

        page = document.metadata.get(
            "page",
            "unknown"
        )

        if isinstance(page, int):
            page = page + 1

        if isinstance(result, dict):

            dense_score = float(
                result.get(
                    "dense_score",
                    0.0
                )
            )

            bm25_score = float(
                result.get(
                    "bm25_score",
                    0.0
                )
            )

            rrf_score = float(
                result.get(
                    "rrf_score",
                    0.0
                )
            )

        else:

            dense_score = 0.0
            bm25_score = 0.0
            rrf_score = float(
                result[1]
            )

        sources.append(
            {
                "rank": rank,
                "source": source,
                "page": page,
                "dense_score": dense_score,
                "bm25_score": bm25_score,
                "rrf_score": rrf_score,
            }
        )

    return sources


# ============================================================
# MAIN QUESTION ANSWERING
# ============================================================

def answer_question(
    query,
    vector_store,
    bm25,
    chunks,
    top_k=TOP_K,
):
    """
    Complete intelligent RAG pipeline.

    FLOW:

    User Question
          ↓
    Hybrid Search
          ↓
    Context Relevance Check
          ↓
       ┌───────┴───────┐
       ↓               ↓
    Relevant       Not Relevant
       ↓               ↓
    RAG Answer     General LLM
       ↓               ↓
    Sources        No Sources
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

        print(
            "\nNo documents retrieved."
        )

        print(
            "Using general LLM knowledge."
        )

        answer = ask_groq(
            build_general_prompt(query)
        )

        return {
            "answer": answer,
            "mode": "general",
            "sources": [],
            "confidence": {
                "retrieval_confidence": 0.0,
                "evidence_confidence": 0.0,
                "overall_confidence": 0.0,
            },
            "retrieved_chunks": 0,
        }

    print(
        f"\nRetrieved chunks: {len(results)}"
    )

    # ========================================================
    # STEP 2: CONFIDENCE
    # ========================================================

    retrieval_confidence = (
        calculate_retrieval_confidence(
            results
        )
    )

    evidence_confidence = (
        calculate_evidence_confidence(
            results
        )
    )

    confidence = (
        calculate_overall_confidence(
            retrieval_confidence,
            evidence_confidence,
        )
    )

    print(
        f"\nRetrieval confidence: "
        f"{retrieval_confidence:.2f}"
    )

    print(
        f"Evidence confidence: "
        f"{evidence_confidence:.2f}"
    )

    print(
        f"Overall confidence: "
        f"{confidence['overall_confidence']:.2f}"
    )

    # ========================================================
    # STEP 3: IMPORTANT
    # CHECK WHETHER DOCUMENT REALLY ANSWERS QUESTION
    # ========================================================

    print(
        "\nChecking document relevance..."
    )

    document_relevant = check_context_relevance(
        query,
        results
    )

    # ========================================================
    # CASE 1: QUESTION IS IN DOCUMENT
    # ========================================================

    if document_relevant:

        print(
            "Document relevance: YES"
        )

        print(
            "Answer mode: RAG"
        )

        prompt = build_rag_prompt(
            query,
            results
        )

        answer = ask_groq(
            prompt
        )

        sources = build_sources(
            results
        )

        display_sources(
            results
        )

        return {
            "answer": answer,
            "mode": "rag",
            "sources": sources,
            "confidence": confidence,
            "retrieved_chunks": len(results),
        }

    # ========================================================
    # CASE 2: QUESTION IS OUTSIDE DOCUMENT
    # ========================================================

    print(
        "Document relevance: NO"
    )

    print(
        "Answer mode: GENERAL KNOWLEDGE"
    )

    general_prompt = build_general_prompt(
        query
    )

    answer = ask_groq(
        general_prompt
    )

    # VERY IMPORTANT:
    # Do NOT return document results.
    # Do NOT return document sources.

    return {
        "answer": answer,
        "mode": "general",
        "sources": [],
        "confidence": {
            "retrieval_confidence": 0.0,
            "evidence_confidence": 0.0,
            "overall_confidence": 0.0,
        },
        "retrieved_chunks": 0,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        print(
            "\nLoading documents..."
        )

        documents = load_pdfs(
            PDF_PATHS
        )

        print(
            f"Total pages: {len(documents)}"
        )

        print(
            "\nCreating chunks..."
        )

        chunks = split_documents(
            documents,
            min_chunk_size=100,
        )

        print(
            f"Total chunks: {len(chunks)}"
        )

        print(
            "\nLoading embedding model..."
        )

        embedding_model = (
            create_embedding_model()
        )

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

        print(
            "\nCreating BM25 index..."
        )

        bm25 = create_bm25_retriever(
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

        result = answer_question(
            question,
            vector_store,
            bm25,
            chunks,
            top_k=TOP_K,
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "FINAL ANSWER"
        )

        print(
            "=" * 60
        )

        print(
            result["answer"]
        )

        print(
            "\nMode:",
            result["mode"]
        )

        print(
            "\nRetrieved chunks:",
            result["retrieved_chunks"]
        )

    except Exception as e:

        print(
            "\n" + "=" * 60
        )

        print(
            "RAG PIPELINE ERROR"
        )

        print(
            "=" * 60
        )

        print(
            f"\nError: {str(e)}"
        )

        raise