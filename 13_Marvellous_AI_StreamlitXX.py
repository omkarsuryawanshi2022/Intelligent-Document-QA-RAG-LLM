import os
import io
import re
import json
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ollama

from gtts import gTTS
from pypdf import PdfReader
from streamlit_mic_recorder import speech_to_text

from rag_engineX import RAGEngine


# =====================================================
# APP CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="OS AI Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_TITLE = "OS AI Agent"
OLLAMA_MODEL = "llama3.2:1b"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 50
DEFAULT_TOP_K = 5


# =====================================================
# CHATGPT-LIKE STYLING
# =====================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     "Helvetica Neue", Arial, sans-serif;
    }

    .stApp {
        background: #ffffff;
        color: #0d0d0d;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;
        max-width: 900px;
    }

    h1, h2, h3 {
        color: #0d0d0d;
        letter-spacing: -0.01em;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #f9f9f9;
        border-right: 1px solid #ececec;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.75rem;
    }
    section[data-testid="stSidebar"] h1 {
        font-size: 1.05rem !important;
        font-weight: 600;
        color: #0d0d0d;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        padding: 8px 12px;
        margin: 2px 0;
        border-radius: 10px;
        transition: background 0.15s ease;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: #ececec;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] input {
        display: none;
    }

    /* ---- Hero greeting ---- */
    .hero-greeting {
        text-align: center;
        margin-top: 4rem;
        margin-bottom: 2rem;
    }
    .hero-greeting h1 {
        font-size: 2.4rem;
        font-weight: 500;
        color: #0d0d0d;
        margin: 0;
        letter-spacing: -0.02em;
    }

    /* ---- Composer ---- */
    div[data-testid="stChatInput"] {
        border-radius: 999px !important;
        border: 1px solid #e5e5e5 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        background: #ffffff;
        padding: 6px 10px;
    }
    div[data-testid="stChatInput"] textarea {
        border: none !important;
        background: transparent !important;
        padding: 10px 4px !important;
        font-size: 1rem;
    }
    div[data-testid="stChatInput"] button {
        border-radius: 999px !important;
        background: #0d0d0d;
        color: white;
        border: none;
    }

    /* ---- Chat bubbles ---- */
    div[data-testid="stChatMessage"] {
        background: transparent;
        border: none;
        padding: 0.5rem 0;
    }
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarUser"]
    ) div[data-testid="stMarkdownContainer"] {
        background: #f4f4f4;
        padding: 12px 16px;
        border-radius: 20px;
        display: inline-block;
        max-width: 100%;
    }
    div[data-testid="stChatMessage"]:has(
        div[data-testid="stChatMessageAvatarAssistant"]
    ) div[data-testid="stMarkdownContainer"] {
        background: transparent;
        padding: 4px 0;
    }

    /* ---- Mic button (pill) ---- */
    div[data-testid="column"] .stButton > button {
        border-radius: 999px;
        border: 1px solid #e5e5e5;
        background: #ffffff;
        color: #0d0d0d;
        font-weight: 500;
        min-height: 44px;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: #f4f4f4;
        border-color: #d4d4d4;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid #e5e5e5;
        background: #ffffff;
        font-size: 0.85rem;
        color: #555;
        margin-right: 6px;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #ececec;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 8px 0;
    }

    .small-note {
        color: #8e8e8e;
        font-size: 0.8rem;
    }

    [data-testid="stMetric"] {
        background: #fafafa;
        padding: 14px;
        border: 1px solid #ececec;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"] {
        color: #8e8e8e !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #0d0d0d !important;
        font-weight: 600;
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid #e5e5e5;
        background: #ffffff;
        color: #0d0d0d;
        font-weight: 500;
    }
    .stButton > button:hover {
        background: #f4f4f4;
        border-color: #d4d4d4;
    }
    .stButton > button[kind="primary"] {
        background: #0d0d0d;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #262626;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #ececec;
        border-radius: 12px;
        background: #ffffff;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* typing indicator dots */
    @keyframes pulseDot {
        0%, 80%, 100% { opacity: 0.2; }
        40% { opacity: 1; }
    }
    .typing-dots span {
        display: inline-block;
        width: 6px;
        height: 6px;
        margin: 0 2px;
        border-radius: 50%;
        background: #0d0d0d;
        animation: pulseDot 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# SESSION STATE
# =====================================================

def initialize_session_state():
    defaults = {
        "messages": [],
        "active_document_ids": [],
        "active_document_names": [],
        "last_answer": "",
        "last_sources": [],
        "conversation_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "last_uploaded_files": [],
        "pending_question": None,   # queued question to render instantly
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# =====================================================
# INITIALIZE RAG ENGINE
# =====================================================

@st.cache_resource(show_spinner="Loading PDF search engine...")
def get_rag_engine():
    return RAGEngine(
        persist_directory=str(BASE_DIR / "chroma_db"),
        collection_name="OS_ai_Agent_v2",
        chunk_size=900,
        chunk_overlap=180,
    )


try:
    rag = get_rag_engine()
except Exception as exc:
    st.error(
        "RAG engine initialize nahi hua. "
        "Packages aur ChromaDB configuration check karein."
    )
    st.exception(exc)
    st.stop()


# =====================================================
# OLLAMA
# =====================================================

def check_ollama():
    try:
        ollama.list()
        return True
    except Exception:
        return False


def ollama_response(prompt, system_prompt=None, stream=False):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=stream,
        options={
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 512,
        },
    )


def generate_text(prompt, system_prompt=None):
    response = ollama_response(prompt=prompt, system_prompt=system_prompt, stream=False)
    return response["message"]["content"].strip()


# =====================================================
# CONVERSATION MEMORY
# =====================================================

def build_conversation_history(limit=10):
    history = st.session_state.messages[-limit:]
    lines = []
    for message in history:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    return "\n".join(lines)


def add_message(role, content, sources=None):
    item = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if sources:
        item["sources"] = sources
    st.session_state.messages.append(item)


# =====================================================
# SOURCE CITATIONS
# =====================================================

def format_sources(matches):
    sources = []
    seen = set()
    for match in matches:
        filename = match.get("filename", "Unknown PDF")
        page = match.get("page", 0)
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "filename": filename,
                "page": page,
                "score": round(float(match.get("score", 0)), 3),
                "text": match.get("text", ""),
            }
        )
    return sources


def render_sources(sources):
    if not sources:
        st.info("Is answer ke liye koi source passage retrieve nahi hua.")
        return

    st.markdown("#### 📚 Sources")
    for source in sources:
        st.markdown(
            f"""
            <div class="source-card">
                <b>📄 {source['filename']}</b><br>
                <span class="small-note">Page {source['page']} · relevance {source['score']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"View extracted text — Page {source['page']}"):
            st.write(source["text"])


# =====================================================
# RAG RETRIEVAL (shared)
# =====================================================

def retrieve_matches(question, document_ids):
    all_matches = []

    if not document_ids:
        all_matches.extend(rag.search(question=question, top_k=DEFAULT_TOP_K))
    else:
        per_document_k = max(2, DEFAULT_TOP_K // len(document_ids))
        for document_id in document_ids:
            all_matches.extend(
                rag.search(question=question, document_id=document_id, top_k=per_document_k)
            )
        global_matches = rag.search(question=question, top_k=DEFAULT_TOP_K)
        selected_ids = set(document_ids)
        for match in global_matches:
            if match["document_id"] in selected_ids:
                all_matches.append(match)

    unique_matches = []
    seen = set()
    for match in all_matches:
        key = (match["document_id"], match["page"], match["text"])
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    unique_matches.sort(key=lambda item: item.get("score", 0), reverse=True)
    return unique_matches[:8]


def build_rag_prompt(question, context):
    history = build_conversation_history(limit=8)
    prompt = f"""
You are OS AI Agent, a helpful PDF question-answering assistant.

Answer the user's question using the retrieved PDF context.

STRICT RULES:
1. Prioritize facts present in the PDF context.
2. Do not invent facts or page references.
3. If the context does not contain the answer, clearly say so.
4. Cite facts as [filename, Page N].
5. Use conversation history to resolve follow-ups.

Recent conversation:
{history}

Current question:
{question}

Retrieved PDF context:
{context}

Answer:
"""
    system_prompt = (
        "You are a careful retrieval-augmented AI Agent. "
        "Accuracy and honest source attribution matter most."
    )
    return prompt, system_prompt


def build_general_prompt(question):
    """
    Fast-path prompt for general questions that don't need PDFs.
    """
    history = build_conversation_history(limit=6)
    prompt = f"""
You are OS AI Agent, a helpful assistant.

Answer the user's question clearly and concisely.
Use conversation history to resolve follow-up questions.
Do not invent facts. If you are unsure, say so.

Recent conversation:
{history}

Current question:
{question}

Answer:
"""
    system_prompt = "You are a helpful, accurate assistant."
    return prompt, system_prompt


# =====================================================
# ROUTING: does this question need PDFs?
# =====================================================

PDF_HINTS = re.compile(
    r"\b(pdf|document|file|page|section|chapter|paragraph|"
    r"uploaded|indexed|text above|according to|summarize|summary|"
    r"extract|quote|cite|report|notes?|article)\b",
    re.IGNORECASE,
)

GENERAL_HINTS = re.compile(
    r"^(hi|hello|hey|namaste|who are you|what can you do|"
    r"tell me about|give me info|what is|explain|describe|"
    r"define|how to|why does|when did|where is|who is)\b",
    re.IGNORECASE,
)


def should_use_pdfs(question, document_ids):
    """
    Cheap heuristic — decides whether to hit the RAG path or
    the fast general-answer path. No LLM call.

    - If PDFs are explicitly selected → use RAG (user opted in).
    - If the question mentions PDF-specific words → use RAG.
    - Otherwise → general answer (fast).
    """
    if document_ids:
        return True
    if PDF_HINTS.search(question):
        return True
    if GENERAL_HINTS.search(question):
        return False
    # default: general
    return False


# =====================================================
# RAG QUESTION ANSWERING — STREAMING (reuse matches)
# =====================================================

def stream_answer_pdf_question(question, document_ids, matches=None):
    if matches is None:
        matches = retrieve_matches(question, document_ids)

    if not matches:
        yield (
            "Mujhe indexed PDFs mein is question ka relevant content nahi mila. "
            "Please PDF check karein ya question ko specific banayein."
        )
        return

    context = rag.build_context(matches, max_characters=10000)
    prompt, system_prompt = build_rag_prompt(question, context)

    try:
        stream = ollama_response(prompt=prompt, system_prompt=system_prompt, stream=True)
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as exc:
        yield (
            "\n\nOllama se response generate nahi hua. "
            f"Model `{OLLAMA_MODEL}` installed hai? ({exc})"
        )


def stream_general_answer(question):
    prompt, system_prompt = build_general_prompt(question)
    try:
        stream = ollama_response(prompt=prompt, system_prompt=system_prompt, stream=True)
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as exc:
        yield (
            "\n\nOllama se response generate nahi hua. "
            f"Model `{OLLAMA_MODEL}` installed hai? ({exc})"
        )


# =====================================================
# PDF SUMMARY / COMPARISON
# =====================================================

def summarize_pdf(document_id):
    document = rag.documents_metadata.get(document_id)
    if not document:
        return "Document metadata nahi mila."

    pages = rag.extract_pdf_pages(document["source_path"])
    sections = [f"[Page {p['page']}]\n{p['text'][:2500]}" for p in pages if p["text"]]
    text = "\n\n".join(sections)[:24000]

    prompt = f"""
Summarize the following PDF content.

Provide:
1. Short overview
2. Main topics
3. Important concepts
4. Key facts
5. Useful takeaways

PDF: {document['filename']}

Content:
{text}
"""
    return generate_text(
        prompt,
        system_prompt="You are an accurate document summarization assistant.",
    )


def compare_pdfs(document_id_1, document_id_2):
    document_1 = rag.documents_metadata.get(document_id_1)
    document_2 = rag.documents_metadata.get(document_id_2)
    if not document_1 or not document_2:
        return "Dono documents available nahi hain."

    query = "main topics concepts definitions important facts objectives and conclusions"
    matches_1 = rag.search(query, document_id=document_id_1, top_k=8)
    matches_2 = rag.search(query, document_id=document_id_2, top_k=8)
    context_1 = rag.build_context(matches_1, max_characters=9000)
    context_2 = rag.build_context(matches_2, max_characters=9000)

    prompt = f"""
Compare these two PDFs.

PDF A: {document_1['filename']}
{context_1}

PDF B: {document_2['filename']}
{context_2}

Produce:
1. Overview of PDF A
2. Overview of PDF B
3. Similarities
4. Differences
5. Unique topics in each PDF
6. Practical conclusion
"""
    return generate_text(
        prompt,
        system_prompt="You compare documents using evidence, not assumptions.",
    )


# =====================================================
# TEXT TO SPEECH
# =====================================================

def create_tts_audio(text, language="en"):
    text = text.strip()
    if not text:
        return None
    text = text[:4500]
    audio_buffer = io.BytesIO()
    gTTS(text=text, lang=language, slow=False).write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;
                    padding:6px 4px 16px 4px;">
            <div style="width:28px;height:28px;border-radius:50%;
                        background:#0d0d0d;display:flex;
                        align-items:center;justify-content:center;
                        color:white;font-size:14px;">✦</div>
            <div style="font-weight:600;font-size:1rem;">OS AI Agent</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home & Chat",
            "📤 Upload & Index",
            "📝 PDF Summary",
            "⚖️ Compare PDFs",
            "📊 PDF Analytics",
            "🕘 Chat History",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        '<div class="small-note" style="padding:4px 0 6px 0;">SYSTEM STATUS</div>',
        unsafe_allow_html=True,
    )

    if check_ollama():
        st.markdown(
            '<div style="font-size:0.82rem;color:#0d0d0d;">🟢 Ollama reachable</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-size:0.82rem;color:#b91c1c;">🔴 Ollama not reachable</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="small-note">Model · {OLLAMA_MODEL}</div>'
        f'<div class="small-note">Chunks · {rag.collection.count()}</div>',
        unsafe_allow_html=True,
    )

    if st.button("🗑️ Clear current chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_answer = ""
        st.session_state.last_sources = []
        st.rerun()


# =====================================================
# DOCUMENT SELECTOR
# =====================================================

documents = rag.list_documents()
document_options = {item["filename"]: item["document_id"] for item in documents}

if documents:
    with st.sidebar.expander("📚 PDFs to search", expanded=False):
        selected_names = st.multiselect(
            "Search only selected PDFs",
            options=list(document_options.keys()),
            default=[
                name for name in st.session_state.active_document_names
                if name in document_options
            ],
            help="Agar koi PDF select nahi hai, to general answer + full knowledge base search hoga.",
        )
        st.session_state.active_document_names = selected_names
        st.session_state.active_document_ids = [
            document_options[n] for n in selected_names
        ]


# =====================================================
# PAGE: HOME & CHAT  (instant feedback + mic button)
# =====================================================

if page == "🏠 Home & Chat":

    # ---- Hero greeting when empty ----
    if not st.session_state.messages and not st.session_state.pending_question:
        st.markdown(
            '<div class="hero-greeting"><h1>What\'s on your mind today?</h1></div>',
            unsafe_allow_html=True,
        )

    # ---- Render existing conversation ----
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    render_sources(message["sources"])

    # ---- Mic button + composer row ----
    # Chat input is rendered by st.chat_input; mic sits just above it
    # so it looks like the ChatGPT "mic" button in the composer area.
    mic_col_1, mic_col_2, mic_col_3 = st.columns([1, 4, 1])

    with mic_col_3:
        # We use session_state flag so the speech widget only renders
        # when the user actually clicks the mic button.
        if st.button("🎤", key="open_mic", help="Voice input",
                     use_container_width=True):
            st.session_state["mic_open"] = not st.session_state.get("mic_open", False)

    if st.session_state.get("mic_open", False):
        try:
            voice_text = speech_to_text(
                language="en",
                start_prompt="🎙️ Start recording",
                stop_prompt="⏹️ Stop recording",
                just_once=True,
                use_container_width=True,
                key="voice_input",
            )
        except Exception:
            voice_text = None
            st.caption("Voice input component available nahi hua.")

        if voice_text:
            st.session_state["pending_question"] = voice_text
            st.session_state["mic_open"] = False

    # ---- Prompt composer ----
    question = st.chat_input("Ask anything…")

    if question:
        st.session_state["pending_question"] = question

    # ---- Handle pending question: instant feedback + stream ----
    pending = st.session_state.get("pending_question")

    if pending:
        st.session_state["pending_question"] = None
        pending = pending.strip()

        if pending:
            # 1. Show user bubble IMMEDIATELY
            add_message("user", pending)
            with st.chat_message("user"):
                st.markdown(pending)

            # 2. Decide fast path vs RAG path
            use_rag = should_use_pdfs(
                pending,
                st.session_state.active_document_ids,
            )

            with st.chat_message("assistant"):
                matches = []
                sources = []

                if use_rag:
                    # RAG path: retrieve first (fast), show sources,
                    # then stream tokens from the model.
                    with st.spinner("Searching your PDFs…"):
                        matches = retrieve_matches(
                            pending,
                            st.session_state.active_document_ids,
                        )
                        sources = format_sources(matches)

                    if sources:
                        with st.expander("📚 Source citations", expanded=False):
                            render_sources(sources)

                    stream = stream_answer_pdf_question(
                        pending,
                        st.session_state.active_document_ids,
                        matches=matches,
                    )
                else:
                    # Fast general path: start streaming right away.
                    stream = stream_general_answer(pending)

                # Small typing indicator while waiting for first token
                placeholder = st.empty()
                placeholder.markdown(
                    '<div class="typing-dots">'
                    "<span></span><span></span><span></span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                # Stream and swap in the first token immediately
                answer = st.write_stream(stream)
                placeholder.empty()

                if isinstance(answer, list):
                    answer = "".join(answer)
                answer = (answer or "").strip()

                st.session_state.last_answer = answer
                st.session_state.last_sources = sources
                add_message("assistant", answer, sources=sources)

            st.rerun()

    # ---- TTS controls ----
    if st.session_state.last_answer:
        with st.expander("🔊 Listen to the latest answer", expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                tts_language = st.selectbox(
                    "Language", options=["English", "Hindi"],
                    key="tts_language", label_visibility="collapsed",
                )
            with col2:
                if st.button("Generate speech", key="generate_tts", use_container_width=True):
                    try:
                        audio_bytes = create_tts_audio(
                            st.session_state.last_answer,
                            language=("hi" if tts_language == "Hindi" else "en"),
                        )
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                    except Exception as exc:
                        st.error("Text-to-speech failed.")
                        st.caption(str(exc))


# =====================================================
# PAGE: UPLOAD & INDEX
# =====================================================

elif page == "📤 Upload & Index":

    st.subheader("📤 Upload PDF documents")
    st.caption("Upload PDFs to extract text and index them for search.")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum recommended size: {MAX_UPLOAD_MB} MB per file.",
    )

    force_reindex = st.checkbox("Re-index existing documents", value=False)

    if st.button("🚀 Index PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Pehle kam se kam ek PDF select karein.")
        else:
            progress = st.progress(0)
            successful = 0
            failures = []

            for index, uploaded_file in enumerate(uploaded_files):
                try:
                    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
                        raise ValueError(f"File {MAX_UPLOAD_MB} MB limit se badi hai.")
                    destination = UPLOAD_DIR / uploaded_file.name
                    destination.write_bytes(uploaded_file.getvalue())
                    with st.spinner(f"Indexing {uploaded_file.name}..."):
                        result = rag.add_pdf(str(destination), replace_existing=force_reindex)
                    successful += 1
                    st.success(f"{uploaded_file.name}: {result['message']}")
                    if not result.get("already_indexed", False):
                        st.write({
                            "Pages": result.get("page_count"),
                            "Words": result.get("word_count"),
                            "Chunks": result.get("chunk_count"),
                            "Empty pages": result.get("empty_pages"),
                        })
                except Exception as exc:
                    failures.append((uploaded_file.name, str(exc)))
                    st.error(f"Could not index {uploaded_file.name}: {exc}")
                progress.progress((index + 1) / len(uploaded_files))

            st.success(f"Completed: {successful} PDF(s) processed.")
            if failures:
                st.warning(f"{len(failures)} file(s) could not be indexed.")
            st.rerun()

    st.divider()
    st.subheader("📚 Indexed documents")

    current_documents = rag.list_documents()
    if current_documents:
        dataframe = pd.DataFrame([
            {
                "Filename": doc["filename"],
                "Pages": doc.get("page_count", 0),
                "Words": doc.get("word_count", 0),
                "Chunks": doc.get("chunk_count", 0),
                "Indexed at": doc.get("indexed_at", ""),
                "Document ID": doc["document_id"],
            }
            for doc in current_documents
        ])
        st.dataframe(dataframe, use_container_width=True, hide_index=True)

        delete_names = {
            f"{doc['filename']} — {doc['document_id']}": doc["document_id"]
            for doc in current_documents
        }
        selected_to_delete = st.selectbox(
            "Select document to delete",
            options=["Select a document"] + list(delete_names.keys()),
        )
        if st.button("🗑️ Delete selected document"):
            if selected_to_delete != "Select a document":
                rag.delete_document(delete_names[selected_to_delete])
                st.success("Document deleted.")
                st.rerun()
    else:
        st.info("Abhi tak koi PDF index nahi hua.")


# =====================================================
# PAGE: PDF SUMMARY
# =====================================================

elif page == "📝 PDF Summary":
    st.subheader("📝 PDF Summary")

    if not documents:
        st.info("Summary generate karne se pehle PDF index karein.")
    else:
        selected_summary_name = st.selectbox(
            "Select PDF", options=list(document_options.keys()), key="summary_pdf"
        )
        summary_document_id = document_options[selected_summary_name]

        if st.button("Generate summary", type="primary"):
            try:
                with st.spinner("Generating PDF summary..."):
                    summary = summarize_pdf(summary_document_id)
                st.markdown(summary)
                st.download_button(
                    "⬇️ Download summary",
                    data=summary,
                    file_name=Path(selected_summary_name).stem + "_summary.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Summary generation failed: {exc}")


# =====================================================
# PAGE: COMPARE PDFs
# =====================================================

elif page == "⚖️ Compare PDFs":
    st.subheader("⚖️ Compare two PDFs")

    if len(documents) < 2:
        st.info("Comparison ke liye kam se kam 2 PDFs index karein.")
    else:
        comparison_names = list(document_options.keys())
        col_a, col_b = st.columns(2)
        with col_a:
            pdf_a_name = st.selectbox("PDF A", options=comparison_names, key="compare_a")
        with col_b:
            pdf_b_name = st.selectbox(
                "PDF B", options=comparison_names, index=1, key="compare_b"
            )

        if pdf_a_name == pdf_b_name:
            st.warning("Please select two different PDFs.")
        elif st.button("Compare PDFs", type="primary"):
            try:
                with st.spinner("Comparing documents..."):
                    comparison = compare_pdfs(
                        document_options[pdf_a_name],
                        document_options[pdf_b_name],
                    )
                st.markdown(comparison)
                st.download_button(
                    "⬇️ Download comparison",
                    data=comparison,
                    file_name="pdf_comparison.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")


# =====================================================
# PAGE: ANALYTICS
# =====================================================

elif page == "📊 PDF Analytics":
    st.subheader("📊 PDF analytics dashboard")

    current_documents = rag.list_documents()
    if not current_documents:
        st.info("Analytics dekhne ke liye PDF upload aur index karein.")
    else:
        analytics_names = {
            f"{doc['filename']} — {doc['document_id']}": doc["document_id"]
            for doc in current_documents
        }
        selected_analytics_name = st.selectbox(
            "Select document", options=list(analytics_names.keys())
        )
        selected_analytics_id = analytics_names[selected_analytics_name]

        try:
            analytics = rag.get_pdf_analytics(selected_analytics_id)
            if not analytics:
                st.warning("Analytics metadata available nahi hai.")
                st.stop()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pages", analytics["page_count"])
            m2.metric("Words", f"{analytics['word_count']:,}")
            m3.metric("Chunks", analytics["chunk_count"])
            m4.metric("Avg. words/page", analytics["average_words_per_page"])

            st.divider()
            st.markdown("### 📄 Page-wise word count")
            page_df = pd.DataFrame([
                {
                    "Page": p["page"],
                    "Words": p["word_count"],
                    "Characters": p["character_count"],
                }
                for p in analytics["pages"]
            ])
            st.dataframe(page_df, use_container_width=True, hide_index=True)
            st.bar_chart(page_df.set_index("Page")["Words"])

            st.markdown("### 🔤 Top keywords")
            keywords = analytics["top_keywords"]
            if keywords:
                keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Frequency"])
                st.dataframe(keyword_df, use_container_width=True, hide_index=True)
                st.bar_chart(keyword_df.set_index("Keyword")["Frequency"])
            else:
                st.info("Keyword statistics ke liye sufficient text nahi hai.")

            st.markdown("### ℹ️ Document information")
            st.json({
                "filename": analytics["filename"],
                "document_id": analytics["document_id"],
                "indexed_at": analytics.get("indexed_at"),
                "empty_pages": analytics.get("empty_pages"),
                "character_count": analytics.get("character_count"),
                "source_path": analytics.get("source_path"),
            })

            csv_bytes = page_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download page analytics CSV",
                data=csv_bytes,
                file_name="pdf_page_analytics.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Analytics generate nahi hui: {exc}")


# =====================================================
# PAGE: CHAT HISTORY
# =====================================================

elif page == "🕘 Chat History":
    st.subheader("🕘 Current session chat history")
    messages = st.session_state.messages

    if not messages:
        st.info("Abhi current session mein chat history nahi hai.")
    else:
        st.caption(f"Messages: {len(messages)}")
        for index, message in enumerate(messages):
            role_label = "You" if message["role"] == "user" else "AI Agent"
            with st.expander(
                f"{index + 1}. {role_label} — {message.get('timestamp', '')}"
            ):
                st.markdown(message["content"])
                if message.get("sources"):
                    render_sources(message["sources"])

        export_data = json.dumps(messages, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇️ Export chat history JSON",
            data=export_data,
            file_name=f"chat_history_{st.session_state.conversation_id}.json",
            mime="application/json",
        )

        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.session_state.last_answer = ""
            st.session_state.last_sources = []
            st.rerun()

    st.warning(
        "Chat history current Streamlit session mein rakhi jaati hai. "
        "Zaroori chats ko JSON mein export karein."
    )


# =====================================================
# PAGE: SETTINGS
# =====================================================

elif page == "⚙️ Settings":
    st.subheader("⚙️ Settings")

    st.markdown("### Ollama")
    st.write(f"Selected model: `{OLLAMA_MODEL}`")
    if st.button("Check Ollama connection"):
        if check_ollama():
            st.success("Ollama server reachable hai.")
        else:
            st.error("Ollama server reachable nahi hai. `ollama serve` check karein.")

    st.markdown("### PDF search engine")
    st.write(f"Embedding model: `{rag.embedding_model_name}`")
    st.write(f"ChromaDB collection: `{rag.collection_name}`")
    st.write(f"Indexed chunks: `{rag.collection.count()}`")

    st.markdown("### Conversation memory")
    st.write(
        "Conversation memory recent messages ko follow-up questions resolve "
        "karne ke liye use karti hai. Current session tak limited."
    )

    st.markdown("### Voice")
    st.write(
        "Voice input mic button se open hota hai. "
        "Text-to-speech gTTS use karta hai aur internet required hai."
    )

    st.markdown("### Data storage")
    st.write(f"PDF uploads: `{UPLOAD_DIR}`")
    st.write(f"Vector database: `{BASE_DIR / 'chroma_db'}`")
    st.warning("Database files manually delete karne se pehle backup lein.")


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div class="small-note" style="text-align:center;padding:2rem 0 1rem 0;">
        OS AI Agent · Answers depend on extracted PDF text and retrieved passages.<br>
        Always verify important facts against the original document.
    </div>
    """,
    unsafe_allow_html=True,
)