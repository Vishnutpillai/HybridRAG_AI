import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Hybrid RAG AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")


st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,.13), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(14,165,233,.10), transparent 28%),
        #0b1020;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11182d, #0d1325);
    border-right: 1px solid rgba(255,255,255,.08);
}
.block-container { max-width: 1250px; padding-top: 2rem; }

.hero {
    padding: 30px 34px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(99,102,241,.22), rgba(14,165,233,.08)),
                rgba(17,24,39,.72);
    border: 1px solid rgba(148,163,184,.18);
    box-shadow: 0 20px 60px rgba(0,0,0,.22);
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
.hero h1 { font-size: 42px; margin: 12px 0 0; color: #f8fafc; }
.hero p { color: #aebbd0; font-size: 16px; margin-top: 10px; }

.section-title {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 750;
    margin: 12px 0;
}
.answer-box {
    background: linear-gradient(145deg, rgba(30,41,59,.72), rgba(15,23,42,.82));
    border: 1px solid rgba(129,140,248,.22);
    border-radius: 18px;
    padding: 22px;
    color: #e5e7eb;
    line-height: 1.7;
    font-size: 15px;
}
.source-card {
    background: rgba(15,23,42,.72);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 14px;
    padding: 14px 16px;
    margin: 8px 0;
}
.source-title { color: #e2e8f0; font-weight: 700; font-size: 14px; }
.source-meta { color: #94a3b8; font-size: 12px; margin-top: 5px; }

.status {
    padding: 10px 12px;
    border-radius: 12px;
    background: rgba(34,197,94,.10);
    border: 1px solid rgba(34,197,94,.20);
    color: #86efac;
    font-size: 13px;
    font-weight: 650;
}
.muted { color: #94a3b8; font-size: 13px; }
.footer { text-align: center; color: #64748b; font-size: 12px; padding-top: 30px; }

div[data-testid="stTextArea"] textarea {
    background: rgba(15,23,42,.85);
    color: #f8fafc;
    border: 1px solid rgba(148,163,184,.18);
    border-radius: 14px;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 0;
    padding: 12px 18px;
    font-weight: 750;
    color: white;
    background: linear-gradient(90deg, #6366f1, #0ea5e9);
}
div[data-testid="stMetric"] {
    background: rgba(15,23,42,.65);
    border: 1px solid rgba(148,163,184,.12);
    padding: 15px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "answer_data" not in st.session_state:
    st.session_state.answer_data = None


def ask_api(question, api_url):
    response = requests.post(
        api_url.rstrip("/") + "/ask",
        json={"question": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def score_percent(value):
    try:
        return max(0, min(100, round(float(value) * 100)))
    except (TypeError, ValueError):
        return 0


def source_name(path):
    if not path:
        return "Unknown document"
    return str(path).replace("\\", "/").split("/")[-1]


# ---------------- Sidebar ----------------

with st.sidebar:
    st.markdown("## ⚡ Hybrid RAG")
    st.caption("AI-powered document assistant")
    st.divider()

    st.markdown("### 🔌 Backend")
    api_url = st.text_input(
        "FastAPI URL",
        value=DEFAULT_API_URL,
        help="Example: http://localhost:8000",
    )

    if st.button("🔎 Check API"):
        try:
            r = requests.get(api_url.rstrip("/") + "/", timeout=8)
            if r.ok:
                st.success("API is online")
            else:
                st.warning(f"API returned {r.status_code}")
        except requests.RequestException:
            st.error("API is not reachable")

    st.divider()

    st.markdown("### 🧩 Pipeline")
    st.markdown("""
    <div class="status">● RAG API configured</div>
    <br>
    <div class="muted">
    📄 PDF Documents<br><br>
    ✂️ Recursive Chunking<br><br>
    🧠 Embeddings + ChromaDB<br><br>
    🔎 BM25 Retrieval<br><br>
    🔀 Hybrid Search / RRF<br><br>
    📊 Confidence Scoring<br><br>
    🤖 Groq LLM
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    st.caption("Machine Learning + Deep Learning")
    st.caption("1,127 pages • 3,107 chunks")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.answer_data = None
        st.rerun()


# ---------------- Hero ----------------

st.markdown("""
<div class="hero">
    <div class="hero-badge">HYBRID RETRIEVAL • GROQ • RAG</div>
    <h1>⚡ Hybrid RAG AI Assistant</h1>
    <p>
        Ask questions about your Machine Learning and Deep Learning
        knowledge base and receive grounded answers with confidence
        scores and document-level sources.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- Question ----------------

left, right = st.columns([2.8, 1.2], gap="large")

with left:
    st.markdown('<div class="section-title">💬 Ask a question</div>',
                unsafe_allow_html=True)
    question = st.text_area(
        "Question",
        placeholder="Example: What is deep learning?\nExample: Explain gradient descent.",
        height=125,
        label_visibility="collapsed",
    )

with right:
    st.markdown('<div class="section-title">💡 Try one</div>',
                unsafe_allow_html=True)
    examples = [
        "What is deep learning?",
        "What is gradient descent?",
        "What is overfitting?",
        "Explain convolutional neural networks.",
    ]
    selected = st.selectbox(
        "Example questions",
        ["Select a question..."] + examples,
        label_visibility="collapsed",
    )
    if selected != "Select a question...":
        question = selected
    st.caption("Answers use retrieved document context.")

ask_clicked = st.button("🔍 Ask Question", use_container_width=True)


# ---------------- API call ----------------

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating an answer..."):
            try:
                data = ask_api(question.strip(), api_url)
                st.session_state.answer_data = data
                st.session_state.history.append({
                    "question": question.strip(),
                    "answer": data.get("answer", ""),
                })
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Could not connect to FastAPI at {api_url}. "
                    "Start the backend first."
                )
            except requests.exceptions.Timeout:
                st.error("⏱️ The request timed out. Please try again.")
            except requests.exceptions.HTTPError as e:
                try:
                    detail = e.response.json()
                except Exception:
                    detail = e.response.text if e.response is not None else ""
                st.error(f"API error: {detail}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ---------------- Results ----------------

data = st.session_state.answer_data

if data:
    st.divider()

    st.markdown('<div class="section-title">🧠 Answer</div>',
                unsafe_allow_html=True)

    answer = data.get("answer", "No answer returned.")
    st.markdown(
        f'<div class="answer-box">{answer}</div>',
        unsafe_allow_html=True,
    )

    confidence = data.get("confidence", {})
    retrieval = confidence.get("retrieval_confidence", 0)
    evidence = confidence.get("evidence_confidence", 0)
    overall = confidence.get("overall_confidence", 0)
    chunks = data.get("retrieved_chunks", 0)

    st.markdown('<div class="section-title">📊 Retrieval quality</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Overall Confidence", f"{score_percent(overall)}%")
    with m2:
        st.metric("Retrieval", f"{score_percent(retrieval)}%")
    with m3:
        st.metric("Evidence", f"{score_percent(evidence)}%")
    with m4:
        st.metric("Retrieved Chunks", chunks)

    st.markdown('<div class="section-title">📚 Sources</div>',
                unsafe_allow_html=True)

    sources = data.get("sources", [])
    if sources:
        for item in sources:
            rank = item.get("rank", "?")
            source = source_name(item.get("source"))
            page = item.get("page", "?")
            dense = item.get("dense_score", 0)
            bm25 = item.get("bm25_score", 0)
            rrf = item.get("rrf_score", 0)

            st.markdown(f"""
            <div class="source-card">
                <div class="source-title">#{rank} &nbsp; {source}</div>
                <div class="source-meta">
                    📄 Page {page}
                    &nbsp; • &nbsp; 🧠 Dense {float(dense):.4f}
                    &nbsp; • &nbsp; 🔎 BM25 {float(bm25):.4f}
                    &nbsp; • &nbsp; 🔀 RRF {float(rrf):.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("No source metadata returned.")

    with st.expander("🧾 View API response"):
        st.json(data)


# ---------------- History ----------------

if st.session_state.history:
    st.divider()
    st.markdown('<div class="section-title">🕘 Recent questions</div>',
                unsafe_allow_html=True)

    for item in reversed(st.session_state.history[-5:]):
        with st.expander(item["question"]):
            st.write(item["answer"])


st.markdown("""
<div class="footer">
    Hybrid RAG AI Assistant • Semantic Search + BM25 + RRF + Groq
    <br><br>
    Built with Python • FastAPI • Streamlit • ChromaDB • Docker
</div>
""", unsafe_allow_html=True)
