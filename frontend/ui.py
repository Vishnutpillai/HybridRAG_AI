import os
import html
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hybrid RAG AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DEFAULT API URL
# ============================================================

DEFAULT_API_URL = os.getenv(
    "RAG_API_URL",
    "http://localhost:8000"
)


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(content: str) -> None:
    """
    Render raw HTML safely using Streamlit.
    """

    flattened = "\n".join(
        line.strip()
        for line in content.strip("\n").splitlines()
    )

    st.markdown(
        flattened,
        unsafe_allow_html=True
    )


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    /* ==================================================
       MAIN APP
       ================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(99,102,241,.13),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 10%,
                rgba(14,165,233,.10),
                transparent 28%
            ),
            #0b1020;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #11182d,
            #0d1325
        );

        border-right: 1px solid rgba(255,255,255,.08);
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ==================================================
       HERO
       ================================================== */

    .hero {
        padding: 30px 34px;
        border-radius: 24px;

        background:
            linear-gradient(
                135deg,
                rgba(99,102,241,.22),
                rgba(14,165,233,.08)
            ),
            rgba(17,24,39,.72);

        border: 1px solid rgba(148,163,184,.18);

        box-shadow:
            0 20px 60px rgba(0,0,0,.22);

        margin-bottom: 24px;
    }

    .hero-badge {
        display: inline-block;

        padding: 6px 12px;

        border-radius: 999px;

        background: rgba(99,102,241,.16);

        border: 1px solid rgba(129,140,248,.25);

        color: #c7d2fe;

        font-size: 12px;

        font-weight: 700;

        letter-spacing: .5px;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-top: 12px;
        color: #f8fafc;
    }

    .hero-description {
        color: #aebbd0;
        font-size: 16px;
        line-height: 1.7;
        margin-top: 10px;
    }


    /* ==================================================
       SECTION TITLES
       ================================================== */

    .section-title {
        color: #f8fafc;
        font-size: 20px;
        font-weight: 750;
        margin: 14px 0;
    }


    /* ==================================================
       ANSWER BOX
       ================================================== */

    .answer-box {
        background:
            linear-gradient(
                145deg,
                rgba(30,41,59,.72),
                rgba(15,23,42,.82)
            );

        border: 1px solid rgba(129,140,248,.22);

        border-radius: 18px;

        padding: 22px;

        color: #e5e7eb;

        line-height: 1.7;

        font-size: 15px;

        box-shadow:
            0 10px 35px rgba(0,0,0,.15);
    }


    /* ==================================================
       GENERAL ANSWER
       ================================================== */

    .general-answer {
        background:
            linear-gradient(
                145deg,
                rgba(14,116,144,.18),
                rgba(15,23,42,.85)
            );

        border: 1px solid rgba(14,165,233,.30);

        border-radius: 18px;

        padding: 22px;

        color: #e5e7eb;

        line-height: 1.7;

        font-size: 15px;
    }


    /* ==================================================
       INFO BOX
       ================================================== */

    .info-box {
        background: rgba(14,165,233,.10);

        border: 1px solid rgba(14,165,233,.25);

        border-radius: 14px;

        padding: 14px 16px;

        margin: 10px 0;

        color: #bae6fd;

        font-size: 14px;
    }


    /* ==================================================
       UPLOAD BOX
       ================================================== */

    .upload-box {
        background:
            linear-gradient(
                145deg,
                rgba(99,102,241,.12),
                rgba(14,165,233,.08)
            );

        border: 1px solid rgba(129,140,248,.25);

        border-radius: 15px;

        padding: 15px;

        margin-bottom: 12px;

        color: #dbeafe;

        font-size: 13px;

        line-height: 1.6;
    }


    /* ==================================================
       SOURCE CARD
       ================================================== */

    .source-card {
        background: rgba(15,23,42,.72);

        border: 1px solid rgba(148,163,184,.12);

        border-radius: 14px;

        padding: 14px 16px;

        margin: 8px 0;
    }

    .source-title {
        color: #e2e8f0;

        font-weight: 700;

        font-size: 14px;
    }

    .source-meta {
        color: #94a3b8;

        font-size: 12px;

        margin-top: 5px;
    }


    /* ==================================================
       STATUS
       ================================================== */

    .status {
        padding: 10px 12px;

        border-radius: 12px;

        background: rgba(34,197,94,.10);

        border: 1px solid rgba(34,197,94,.20);

        color: #86efac;

        font-size: 13px;

        font-weight: 650;
    }

    .muted {
        color: #94a3b8;
        font-size: 13px;
    }


    /* ==================================================
       FOOTER
       ================================================== */

    .footer {
        text-align: center;

        color: #64748b;

        font-size: 12px;

        padding-top: 30px;

        line-height: 1.8;
    }


    /* ==================================================
       TEXT AREA
       ================================================== */

    div[data-testid="stTextArea"] textarea {
        background: rgba(15,23,42,.85) !important;

        color: #f8fafc !important;

        border: 1px solid rgba(148,163,184,.18);

        border-radius: 14px;
    }


    /* ==================================================
       BUTTON
       ================================================== */

    .stButton > button {
        width: 100%;

        border-radius: 12px;

        border: 0;

        padding: 12px 18px;

        font-weight: 750;

        color: white;

        background:
            linear-gradient(
                90deg,
                #6366f1,
                #0ea5e9
            );
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                90deg,
                #4f46e5,
                #0284c7
            );

        color: white;
    }


    /* ==================================================
       METRICS
       ================================================== */

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,.65);

        border: 1px solid rgba(148,163,184,.12);

        padding: 15px;

        border-radius: 15px;
    }

    </style>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "answer_data" not in st.session_state:
    st.session_state.answer_data = None


# ============================================================
# FUNCTIONS
# ============================================================

def ask_api(question, api_url):

    response = requests.post(
        api_url.rstrip("/") + "/ask",
        json={
            "question": question
        },
        timeout=180
    )

    response.raise_for_status()

    return response.json()


def upload_pdf(uploaded_file, api_url):

    """
    Upload PDF to FastAPI /v1/ingest endpoint.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/pdf"
        )
    }

    response = requests.post(
        api_url.rstrip("/") + "/v1/ingest",
        files=files,
        timeout=300
    )

    response.raise_for_status()

    return response.json()


def score_percent(value):

    try:

        return max(
            0,
            min(
                100,
                round(float(value) * 100)
            )
        )

    except (TypeError, ValueError):

        return 0


def source_name(path):

    if not path:

        return "Unknown document"

    return (
        str(path)
        .replace("\\", "/")
        .split("/")[-1]
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## ⚡ Hybrid RAG")

    st.caption(
        "AI-powered document assistant"
    )

    st.divider()


    # ========================================================
    # BACKEND
    # ========================================================

    st.markdown("### 🔌 Backend")

    api_url = st.text_input(
        "FastAPI URL",
        value=DEFAULT_API_URL,
        help=(
            "Docker Compose: http://backend:8000 | "
            "Local: http://localhost:8000"
        )
    )


    # ========================================================
    # CHECK API
    # ========================================================

    if st.button("🔎 Check API"):

        try:

            r = requests.get(
                api_url.rstrip("/") + "/",
                timeout=8
            )

            if r.ok:

                st.success(
                    "API is online"
                )

            else:

                st.warning(
                    f"API returned {r.status_code}"
                )

        except requests.RequestException:

            st.error(
                "API is not reachable"
            )


    st.divider()


    # ========================================================
    # PDF UPLOAD
    # ========================================================

    st.markdown("### 📤 Upload PDF")

    render_html(
        """
        <div class="upload-box">

        Upload a PDF The backend will process and index the
        document for RAG retrieval.

        </div>
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )


    if uploaded_file is not None:

        st.caption(
            f"📄 {uploaded_file.name}"
        )

        st.caption(
            f"Size: {uploaded_file.size / (1024 * 1024):.2f} MB"
        )


        if st.button(
            "⬆️ Upload & Index PDF",
            use_container_width=True
        ):

            with st.spinner(
                "Uploading and indexing PDF..."
            ):

                try:

                    upload_result = upload_pdf(
                        uploaded_file,
                        api_url
                    )

                    st.success(
                        "✅ PDF uploaded and indexed successfully!"
                    )


                    # Show backend response
                    with st.expander(
                        "📋 Upload details"
                    ):

                        st.json(
                            upload_result
                        )


                except requests.exceptions.ConnectionError:

                    st.error(
                        """
                        ❌ Could not connect to FastAPI.

                        Make sure your backend is running.
                        """
                    )


                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ Upload timed out. Large PDFs may take longer to process."
                    )


                except requests.exceptions.HTTPError as e:

                    try:

                        detail = e.response.json()

                    except Exception:

                        detail = (
                            e.response.text
                            if e.response is not None
                            else ""
                        )

                    st.error(
                        f"❌ Upload failed: {detail}"
                    )


                except Exception as e:

                    st.error(
                        f"❌ Unexpected upload error: {e}"
                    )


    st.divider()


    # ========================================================
    # RETRIEVED CHUNKS
    # ========================================================

    retrieved_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1
    )


    st.divider()


    # ========================================================
    # PIPELINE
    # ========================================================

    st.markdown("### 🧩 Pipeline")

    render_html(
        """
        <div class="status">
            ● RAG API configured
        </div>

        <br>

        <div class="muted">

        📄 PDF Documents
        <br><br>

        📤 PDF Upload & Indexing
        <br><br>

        ✂️ Recursive Chunking
        <br><br>

        🧠 Embeddings + ChromaDB
        <br><br>

        🔎 BM25 Retrieval
        <br><br>

        🔀 Hybrid Search / RRF
        <br><br>

        📊 Confidence Scoring
        <br><br>

        🤖 Groq LLM
        <br><br>

        🌐 General Knowledge Fallback

        </div>
        """
    )


    st.divider()


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    st.markdown("### 📚 Knowledge Base")

    st.caption(
        "Machine Learning + Deep Learning"
    )

    st.caption(
        "1,127 pages • 3,107 chunks"
    )


    st.divider()


    # ========================================================
    # CLEAR CONVERSATION
    # ========================================================

    if st.button(
        "🗑️ Clear conversation"
    ):

        st.session_state.history = []

        st.session_state.answer_data = None

        st.rerun()


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-badge">
            HYBRID RETRIEVAL • GROQ • RAG
        </div>

        <div class="hero-title">
            ⚡ Hybrid RAG AI Assistant
        </div>

        <p class="hero-description">

            Ask questions about your Machine Learning
            and Deep Learning knowledge base.

            Document questions use hybrid retrieval
            with semantic search and BM25.

            You can also upload new PDF documents
            directly through the interface.

        </p>

    </div>
    """
)


# ============================================================
# QUESTION AREA
# ============================================================

left, right = st.columns(
    [2.8, 1.2],
    gap="large"
)


# ============================================================
# QUESTION INPUT
# ============================================================

with left:

    render_html(
        '<div class="section-title">💬 Ask a question</div>'
    )

    question = st.text_area(
        "Question",

        placeholder=(
            "Example: What is deep learning?\n"
            "Example: Explain gradient descent.\n"
            "Example: What is overfitting?"
        ),

        height=125,

        label_visibility="collapsed"
    )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

with right:

    render_html(
        '<div class="section-title">💡 Try one</div>'
    )

    examples = [

        "What is deep learning?",

        "What is gradient descent?",

        "What is overfitting?",

        "Explain convolutional neural networks.",

        "What is the capital of France?",

        "Who is the Prime Minister of India?"

    ]


    selected = st.selectbox(
        "Example questions",

        ["Select a question..."] + examples,

        label_visibility="collapsed"
    )


    if selected != "Select a question...":

        question = selected


    st.caption(
        "Questions related to your PDFs use RAG. "
        "Other questions use general LLM knowledge."
    )


# ============================================================
# ASK BUTTON
# ============================================================

ask_clicked = st.button(
    "🔍 Ask Question",
    use_container_width=True
)


# ============================================================
# API CALL
# ============================================================

if ask_clicked:

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Analyzing question..."
        ):

            try:

                data = ask_api(
                    question.strip(),
                    api_url
                )


                st.session_state.answer_data = data


                st.session_state.history.append(
                    {
                        "question": question.strip(),

                        "answer": data.get(
                            "answer",
                            ""
                        )
                    }
                )


            except requests.exceptions.ConnectionError:

                st.error(
                    f"""
                    ❌ Could not connect to FastAPI.

                    Backend URL:

                    `{api_url}`

                    Make sure Docker Compose is running.
                    """
                )


            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ Request timed out. Please try again."
                )


            except requests.exceptions.HTTPError as e:

                try:

                    detail = e.response.json()

                except Exception:

                    detail = (
                        e.response.text
                        if e.response is not None
                        else ""
                    )

                st.error(
                    f"API error: {detail}"
                )


            except Exception as e:

                st.error(
                    f"Unexpected error: {e}"
                )


# ============================================================
# RESULTS
# ============================================================

data = st.session_state.answer_data


if data:

    st.divider()


    # ========================================================
    # OUT OF SCOPE
    # ========================================================

    out_of_scope = data.get(
        "out_of_scope",
        False
    )


    answer = data.get(
        "answer",
        "No answer returned."
    )


    # ========================================================
    # ANSWER
    # ========================================================

    render_html(
        '<div class="section-title">🧠 Answer</div>'
    )


    safe_answer = html.escape(
        str(answer)
    ).replace(
        "\n",
        "<br>"
    )


    if out_of_scope:

        render_html(
            """
            <div class="info-box">

                🌐 This question is outside the
                document knowledge base.

                The answer below is generated
                using general LLM knowledge.

            </div>
            """
        )


        render_html(
            f"""
            <div class="general-answer">

                {safe_answer}

            </div>
            """
        )


    else:

        render_html(
            f"""
            <div class="answer-box">

                {safe_answer}

            </div>
            """
        )


    # ========================================================
    # GENERAL KNOWLEDGE
    # ========================================================

    if out_of_scope:

        render_html(
            '<div class="section-title">🌐 General Knowledge</div>'
        )


        st.caption(
            "No document sources are displayed because "
            "this question was outside the indexed knowledge base."
        )


    # ========================================================
    # DOCUMENT QUESTION
    # ========================================================

    else:

        confidence = data.get(
            "confidence",
            {}
        )


        retrieval = confidence.get(
            "retrieval_confidence",
            0
        )


        evidence = confidence.get(
            "evidence_confidence",
            0
        )


        overall = confidence.get(
            "overall_confidence",
            0
        )


        chunks = data.get(
            "retrieved_chunks",
            retrieved_k
        )


        # ====================================================
        # RETRIEVAL QUALITY
        # ====================================================

        render_html(
            '<div class="section-title">📊 Retrieval quality</div>'
        )


        m1, m2, m3, m4 = st.columns(4)


        with m1:

            st.metric(
                "Overall Confidence",
                f"{score_percent(overall)}%"
            )


        with m2:

            st.metric(
                "Retrieval",
                f"{score_percent(retrieval)}%"
            )


        with m3:

            st.metric(
                "Evidence",
                f"{score_percent(evidence)}%"
            )


        with m4:

            st.metric(
                "Retrieved Chunks",
                chunks
            )


        # ====================================================
        # SOURCES
        # ====================================================

        render_html(
            '<div class="section-title">📚 Sources</div>'
        )


        sources = data.get(
            "sources",
            []
        )


        if sources:

            for item in sources:

                rank = item.get(
                    "rank",
                    "?"
                )


                source = source_name(
                    item.get("source")
                )


                page = item.get(
                    "page",
                    "?"
                )


                dense = item.get(
                    "dense_score",
                    0
                )


                bm25 = item.get(
                    "bm25_score",
                    0
                )


                rrf = item.get(
                    "rrf_score",
                    0
                )


                render_html(
                    f"""
                    <div class="source-card">

                        <div class="source-title">

                            #{rank}
                            &nbsp;
                            {html.escape(str(source))}

                        </div>

                        <div class="source-meta">

                            📄 Page {page}

                            &nbsp; • &nbsp;

                            🧠 Dense {float(dense):.4f}

                            &nbsp; • &nbsp;

                            🔎 BM25 {float(bm25):.4f}

                            &nbsp; • &nbsp;

                            🔀 RRF {float(rrf):.4f}

                        </div>

                    </div>
                    """
                )


        else:

            st.caption(
                "No source metadata returned."
            )


    # ========================================================
    # API RESPONSE
    # ========================================================

    with st.expander(
        "🧾 View API response"
    ):

        st.json(data)


# ============================================================
# HISTORY
# ============================================================

if st.session_state.history:

    st.divider()


    render_html(
        '<div class="section-title">🕘 Recent questions</div>'
    )


    for item in reversed(
        st.session_state.history[-5:]
    ):

        with st.expander(
            item["question"]
        ):

            st.write(
                item["answer"]
            )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">

        <strong>
            Hybrid RAG AI Assistant
        </strong>

        <br>

        Semantic Search • BM25 • RRF •
        Groq • General Knowledge Fallback

        <br><br>

        Built with Python • FastAPI •
        Streamlit • ChromaDB • Docker

    </div>
    """
)