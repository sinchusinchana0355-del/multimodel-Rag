import streamlit as st
import time
from pypdf import PdfReader

from embeddings import get_jina_embeddings
from vision import describe_image
from chunking import chunk_text
from retriever import FAISSRetriever
from reranker import simple_rerank
from llm import ask_llm


# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="VisionDoc AI",
    page_icon="🧠",
    layout="wide"
)


# ---------------- CUSTOM DARK THEME ---------------- #
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.main-title {
    font-size: 44px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 30px;
}

.panel {
    padding: 20px;
    border-radius: 18px;
    background-color: #1f2937;
    border: 1px solid #374151;
    margin-bottom: 20px;
}

.section-header {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #f3f4f6;
}

.stButton>button {
    background-color: #7c3aed;
    color: white;
    border-radius: 14px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
    border: none;
}
.stButton>button:hover {
    background-color: #5b21b6;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ---------------- #
st.markdown(
    "<div class='main-title'>VisionDoc AI – Intelligent Multimodal RAG</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='subtitle'>AI-powered semantic search across documents and images</div>",
    unsafe_allow_html=True
)


# ---------------- SESSION STATE ---------------- #
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.header("⚙️ System Configuration")

    groq_key = st.text_input("🔑 Vision Model API Key", type="password")
    jina_key = st.text_input("📦 Embedding Engine Key", type="password")

    model = st.selectbox(
        "🤖 Language Brain",
        ["llama-3.1-8b-instant", "openai/gpt-oss-120b"]
    )

    filter_type = st.radio(
        "📂 Search Mode",
        ["all", "text", "image"],
        horizontal=True
    )

    st.divider()


# ---------------- UPLOAD SECTION ---------------- #
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📄 Knowledge Document Upload</div>", unsafe_allow_html=True)
    txt_file = st.file_uploader("Supported: TXT, PDF", type=["txt", "pdf"])
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🖼️ Visual Data Upload</div>", unsafe_allow_html=True)
    img_file = st.file_uploader("Supported: PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------- PROCESSING ---------------- #
if txt_file and groq_key and jina_key:

    with st.spinner("🔄 Processing knowledge sources..."):

        # Extract text
        if txt_file.name.endswith(".pdf"):
            reader = PdfReader(txt_file)
            raw_text = "\n".join(
                [p.extract_text() for p in reader.pages if p.extract_text()]
            )
        else:
            raw_text = txt_file.read().decode("utf-8")

        chunks = chunk_text(raw_text)
        metadata = [{"type": "text"} for _ in chunks]

        # Process image if uploaded
        if img_file:
            image_bytes = img_file.read()
            vision_text = describe_image(image_bytes, groq_key)

            if vision_text:
                chunks.append("Image description: " + vision_text)
                metadata.append({"type": "image"})

        embeddings = get_jina_embeddings(chunks, jina_key)
        retriever = FAISSRetriever(embeddings, metadata)


    # ---------------- QUERY INTERFACE ---------------- #
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💬 Ask Your Question</div>", unsafe_allow_html=True)

    query = st.text_input(
        "Type your query below",
        placeholder="Example: What insights can be derived from the document?"
    )

    run = st.button("🔍 Analyze & Generate Insight", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run and query:

        start = time.time()

        query_emb = get_jina_embeddings([query], jina_key)

        f = None if filter_type == "all" else filter_type
        ids = retriever.search(query_emb, top_k=5, filter_type=f)

        retrieved_docs = [chunks[i] for i in ids]
        reranked = simple_rerank(query, retrieved_docs)

        context = "\n\n".join(reranked[:3])

        answer = ask_llm(context, query, groq_key, model)

        latency = round(time.time() - start, 2)

        st.session_state.history.append((query, answer))


        # ---------------- ANSWER PANEL ---------------- #
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🧠 Generated Answer</div>", unsafe_allow_html=True)
        st.write(answer)
        st.markdown("</div>", unsafe_allow_html=True)


        # ---------------- PERFORMANCE PANEL ---------------- #
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>⚡ Performance Metrics</div>", unsafe_allow_html=True)
        st.metric("Response Time (seconds)", latency)
        st.markdown("</div>", unsafe_allow_html=True)


        with st.expander("📚 Retrieved Context"):
            st.text(context)

        with st.expander("🕘 Recent Query History"):
            for q, a in st.session_state.history[-5:]:
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
                st.divider()

else:
    st.info("Upload a document and provide valid API keys to begin.")


# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown(
    "<center>Built by Sinchana | Multimodal RAG Project 2026 🚀</center>",
    unsafe_allow_html=True
)
