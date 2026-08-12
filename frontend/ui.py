import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/ask"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-top: 10px;
        line-height: 1.7;
    }

    .source-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Hybrid RAG AI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ask questions about your Machine Learning and Deep Learning documents."
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.success("RAG API configured")

    st.markdown("### Architecture")

    st.markdown(
        """
        **PDF Documents**
        
        ↓
        
        **Chunking**
        
        ↓
        
        **Embeddings + ChromaDB**
        
        +
        
        **BM25**
        
        ↓
        
        **Hybrid Search / RRF**
        
        ↓
        
        **Confidence Scoring**
        
        ↓
        
        **Groq LLM**
        """
    )

    st.divider()

    st.markdown("### 🔎 Retrieval")

    st.write("Vector Search: ChromaDB")
    st.write("Keyword Search: BM25")
    st.write("Fusion: Reciprocal Rank Fusion")
    st.write("Top-K: 5")


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("💬 Ask a Question")

question = st.text_area(
    "Enter your question",
    placeholder="Example: What is deep learning?",
    height=120,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Question",
    type="primary",
    use_container_width=True,
)


# ============================================================
# API REQUEST
# ============================================================

if ask_button:

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("🔎 Searching documents and generating answer..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "question": question.strip()
                    },
                    timeout=120,
                )

                # ------------------------------------------------
                # HTTP ERROR
                # ------------------------------------------------

                if response.status_code != 200:

                    st.error(
                        f"API request failed: "
                        f"{response.status_code}"
                    )

                    st.code(
                        response.text
                    )

                    st.stop()

                # ------------------------------------------------
                # PARSE RESPONSE
                # ------------------------------------------------

                data = response.json()

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the FastAPI server."
                )

                st.info(
                    "Make sure Docker/FastAPI is running on "
                    "http://127.0.0.1:8000"
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "❌ Request timed out."
                )

                st.info(
                    "The RAG pipeline may still be processing "
                    "the request."
                )

                st.stop()

            except requests.exceptions.RequestException as e:

                st.error(
                    f"❌ Request error: {str(e)}"
                )

                st.stop()

            except ValueError:

                st.error(
                    "❌ API returned an invalid JSON response."
                )

                st.stop()


        # ========================================================
        # ANSWER
        # ========================================================

        st.divider()

        st.subheader("🧠 Answer")

        answer = data.get(
            "answer",
            "No answer returned.",
        )

        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )


        # ========================================================
        # CONFIDENCE
        # ========================================================

        confidence = data.get(
            "confidence",
            {},
        )

        st.divider()

        st.subheader("📊 Confidence")

        retrieval_confidence = confidence.get(
            "retrieval_confidence",
            0,
        )

        evidence_confidence = confidence.get(
            "evidence_confidence",
            0,
        )

        overall_confidence = confidence.get(
            "overall_confidence",
            0,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Retrieval",
                f"{retrieval_confidence:.2f}",
            )

            st.progress(
                min(
                    max(
                        float(retrieval_confidence),
                        0.0,
                    ),
                    1.0,
                )
            )

        with col2:

            st.metric(
                "Evidence",
                f"{evidence_confidence:.2f}",
            )

            st.progress(
                min(
                    max(
                        float(evidence_confidence),
                        0.0,
                    ),
                    1.0,
                )
            )

        with col3:

            st.metric(
                "Overall",
                f"{overall_confidence:.2f}",
            )

            st.progress(
                min(
                    max(
                        float(overall_confidence),
                        0.0,
                    ),
                    1.0,
                )
            )


        # ========================================================
        # RETRIEVED CHUNKS
        # ========================================================

        st.divider()

        retrieved_chunks = data.get(
            "retrieved_chunks",
            0,
        )

        st.subheader(
            f"📚 Retrieved Chunks ({retrieved_chunks})"
        )


        # ========================================================
        # SOURCES
        # ========================================================

        sources = data.get(
            "sources",
            [],
        )

        if sources:

            for source in sources:

                rank = source.get(
                    "rank",
                    "-",
                )

                source_name = source.get(
                    "source",
                    "Unknown",
                )

                page = source.get(
                    "page",
                    "Unknown",
                )

                dense_score = source.get(
                    "dense_score",
                    0,
                )

                bm25_score = source.get(
                    "bm25_score",
                    0,
                )

                rrf_score = source.get(
                    "rrf_score",
                    0,
                )

                with st.expander(
                    f"📄 Source {rank}: "
                    f"{source_name} — Page {page}"
                ):

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "Dense Score",
                            f"{float(dense_score):.4f}",
                        )

                    with col2:

                        st.metric(
                            "BM25 Score",
                            f"{float(bm25_score):.4f}",
                        )

                    with col3:

                        st.metric(
                            "RRF Score",
                            f"{float(rrf_score):.4f}",
                        )

                    st.caption(
                        f"Source: {source_name}"
                    )

                    st.caption(
                        f"Page: {page}"
                    )

        else:

            st.info(
                "No source information returned."
            )


        # ========================================================
        # RAW API RESPONSE
        # ========================================================

        with st.expander("🔧 View API Response"):

            st.json(data)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hybrid RAG AI • FastAPI + ChromaDB + BM25 + Groq"
)