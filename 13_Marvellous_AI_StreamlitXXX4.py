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
from openai import OpenAI
from gtts import gTTS
from pypdf import PdfReader
from streamlit_mic_recorder import speech_to_text

# =====================================================
# CONFIGURATION & PROVIDER
# =====================================================
st.set_page_config(
    page_title="OS AI Agent",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "OS AI Agent"

# Groq Configuration (uses OpenAI-compatible API)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = OpenAI(
    api_key=GROQ_API_KEY if GROQ_API_KEY else "dummy",
    base_url="https://api.groq.com/openai/v1",
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = 50
DEFAULT_TOP_K = 5

# =====================================================
# STYLING (ChatGPT-like, Light Theme)
# =====================================================
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }
    .stApp { background: #ffffff; color: #0d0d0d; }
    .block-container { padding-top: 1rem; padding-bottom: 6rem; max-width: 900px; }
    h1, h2, h3 { color: #0d0d0d; letter-spacing: -0.01em; }

    section[data-testid="stSidebar"] { background: #f9f9f9; border-right: 1px solid #ececec; }
    section[data-testid="stSidebar"] > div { padding-top: 0.75rem; }
    section[data-testid="stSidebar"] h1 { font-size: 1.05rem !important; font-weight: 600; }
    
    .hero-greeting { text-align: center; margin-top: 4rem; margin-bottom: 2rem; }
    .hero-greeting h1 { font-size: 2.4rem; font-weight: 500; margin: 0; letter-spacing: -0.02em; }

    div[data-testid="stChatInput"] {
        border-radius: 999px !important; border: 1px solid #e5e5e5 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); background: #ffffff; padding: 6px 10px;
    }
    div[data-testid="stChatInput"] textarea {
        border: none !important; background: transparent !important;
        padding: 10px 4px !important; font-size: 1rem;
    }
    div[data-testid="stChatInput"] button {
        border-radius: 999px !important; background: #0d0d0d; color: white; border: none;
    }

    div[data-testid="stChatMessage"] { background: transparent; border: none; padding: 0.5rem 0; }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stMarkdownContainer"] {
        background: #f4f4f4; padding: 12px 16px; border-radius: 20px; display: inline-block; max-width: 100%;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stMarkdownContainer"] {
        background: transparent; padding: 4px 0;
    }

    div[data-testid="column"] .stButton > button {
        border-radius: 999px; border: 1px solid #e5e5e5; background: #ffffff;
        color: #0d0d0d; font-weight: 500; min-height: 44px;
    }
    div[data-testid="column"] .stButton > button:hover { background: #f4f4f4; }

    .source-card { background: #ffffff; border: 1px solid #ececec; border-radius: 12px; padding: 12px 14px; margin: 8px 0; }
    .small-note { color: #8e8e8e; font-size: 0.8rem; }
    [data-testid="stMetric"] { background: #fafafa; padding: 14px; border: 1px solid #ececec; border-radius: 12px; }
    
    @keyframes pulseDot { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }
    .typing-dots span {
        display: inline-block; width: 6px; height: 6px; margin: 0 2px;
        border-radius: 50%; background: #0d0d0d; animation: pulseDot 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
    
    #MainMenu { visibility: hidden; } footer { visibility: hidden; }
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
        "pending_question": None,
        "mic_open": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# =====================================================
# LAZY LOAD RAG ENGINE (Memory Optimization)
# =====================================================
@st.cache_resource(show_spinner="Loading PDF engine (First time only)...")
def get_rag_engine():
    """
    Lazy loads the RAG Engine. This prevents the app from crashing 
    on startup due to heavy model loading (sentence-transformers).
    """
    from rag_engineX import RAGEngine
    return RAGEngine(
        persist_directory=str(BASE_DIR / "chroma_db"),
        collection_name="OS_ai_Agent_v2",
        chunk_size=900,
        chunk_overlap=180,
    )

# =====================================================
# LLM HELPERS
# =====================================================
def check_llm():
    if not GROQ_API_KEY:
        return False, "GROQ_API_KEY missing"
    return True, f"Groq configured ({LLM_MODEL})"

def build_messages(prompt, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages

def llm_stream(prompt, system_prompt=None):
    messages = build_messages(prompt, system_prompt)
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as exc:
        yield f"\n\nResponse generate nahi hua. Provider error: {exc}"

def llm_text(prompt, system_prompt=None):
    messages = build_messages(prompt, system_prompt)
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"Error: {exc}"

# =====================================================
# CONVERSATION & UTILS
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
        sources.append({
            "filename": filename,
            "page": page,
            "score": round(float(match.get("score", 0)), 3),
            "text": match.get("text", ""),
        })
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
# RAG RETRIEVAL (Lazy loads engine on first use)
# =====================================================
def retrieve_matches(question, document_ids):
    rag = get_rag_engine() # Loads only when this function is called
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

# =====================================================
# PROMPTS & ROUTING
# =====================================================
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

Recent conversation:
{history}
Current question:
{question}
Retrieved PDF context:
{context}
Answer:
"""
    return prompt, "You are a careful retrieval-augmented AI Agent."

def build_general_prompt(question):
    history = build_conversation_history(limit=6)
    prompt = f"""
You are OS AI Agent, a helpful assistant.
Answer the user's question clearly and concisely.
Do not invent facts. If you are unsure, say so.

Recent conversation:
{history}
Current question:
{question}
Answer:
"""
    return prompt, "You are a helpful, accurate assistant."

PDF_HINTS = re.compile(r"\b(pdf|document|file|page|section|chapter|paragraph|uploaded|indexed|summarize|summary|extract|quote|cite)\b", re.IGNORECASE)
GENERAL_HINTS = re.compile(r"^(hi|hello|hey|namaste|who are you|what can you do|tell me about|give me info|what is|explain|describe|define|how to)\b", re.IGNORECASE)

def should_use_pdfs(question, document_ids):
    if document_ids: return True
    if PDF_HINTS.search(question): return True
    if GENERAL_HINTS.search(question): return False
    return False

# =====================================================
# STREAMING GENERATORS
# =====================================================
def stream_answer_pdf_question(question, document_ids, matches=None):
    if matches is None:
        matches = retrieve_matches(question, document_ids)
    if not matches:
        yield "Mujhe indexed PDFs mein is question ka relevant content nahi mila."
        return
    rag = get_rag_engine()
    context = rag.build_context(matches, max_characters=10000)
    prompt, system_prompt = build_rag_prompt(question, context)
    yield from llm_stream(prompt, system_prompt)

def stream_general_answer(question):
    prompt, system_prompt = build_general_prompt(question)
    yield from llm_stream(prompt, system_prompt)

# =====================================================
# TEXT TO SPEECH
# =====================================================
def create_tts_audio(text, language="en"):
    text = text.strip()
    if not text: return None
    audio_buffer = io.BytesIO()
    gTTS(text=text[:4500], lang=language, slow=False).write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.getvalue()

# =====================================================
# SIDEBAR (Lazy loads document list only when needed)
# =====================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;padding:6px 4px 16px 4px;">
            <div style="width:28px;height:28px;border-radius:50%;background:#0d0d0d;
                        display:flex;align-items:center;justify-content:center;color:white;font-size:14px;">✦</div>
            <div style="font-weight:600;font-size:1rem;">OS AI Agent</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏠 Home & Chat", "📤 Upload & Index", "📝 PDF Summary", "⚖️ Compare PDFs", "📊 PDF Analytics", "🕘 Chat History", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<div class="small-note" style="padding:4px 0 6px 0;">SYSTEM STATUS</div>', unsafe_allow_html=True)

    llm_ok, llm_msg = check_llm()
    if llm_ok:
        st.markdown(f'<div style="font-size:0.82rem;color:#0d0d0d;">🟢 {llm_msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:0.82rem;color:#b91c1c;">🔴 {llm_msg}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="small-note">Provider · {MODEL_PROVIDER}</div><div class="small-note">Model · {LLM_MODEL}</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear current chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_answer = ""
        st.session_state.last_sources = []
        st.rerun()

# Document selector in sidebar (Lazy loaded)
if page == "🏠 Home & Chat":
    with st.sidebar.expander("📚 PDFs to search", expanded=False):
        # Only load documents if the user wants to select them
        try:
            rag = get_rag_engine()
            documents = rag.list_documents()
            document_options = {item["filename"]: item["document_id"] for item in documents}
            if documents:
                selected_names = st.multiselect(
                    "Search only selected PDFs",
                    options=list(document_options.keys()),
                    default=[n for n in st.session_state.active_document_names if n in document_options],
                )
                st.session_state.active_document_names = selected_names
                st.session_state.active_document_ids = [document_options[n] for n in selected_names]
            else:
                st.info("No PDFs indexed yet.")
        except Exception:
            st.info("PDF Engine loading... please wait.")

# =====================================================
# PAGE: HOME & CHAT
# =====================================================
if page == "🏠 Home & Chat":
    if not st.session_state.messages and not st.session_state.pending_question:
        st.markdown('<div class="hero-greeting"><h1>What\'s on your mind today?</h1></div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    render_sources(message["sources"])

    # Mic button row
    mic_col_1, mic_col_2, mic_col_3 = st.columns([1, 4, 1])
    with mic_col_3:
        if st.button("🎤", key="open_mic", help="Voice input", use_container_width=True):
            st.session_state["mic_open"] = not st.session_state.get("mic_open", False)

    if st.session_state.get("mic_open", False):
        try:
            voice_text = speech_to_text(
                language="en", start_prompt="🎙️ Start recording", stop_prompt="⏹️ Stop recording",
                just_once=True, use_container_width=True, key="voice_input",
            )
        except Exception:
            voice_text = None
            st.caption("Voice input component available nahi hua.")
        if voice_text:
            st.session_state["pending_question"] = voice_text
            st.session_state["mic_open"] = False

    question = st.chat_input("Ask anything…")
    if question:
        st.session_state["pending_question"] = question

    pending = st.session_state.get("pending_question")
    if pending:
        st.session_state["pending_question"] = None
        pending = pending.strip()
        if pending:
            add_message("user", pending)
            with st.chat_message("user"):
                st.markdown(pending)

            use_rag = should_use_pdfs(pending, st.session_state.active_document_ids)

            with st.chat_message("assistant"):
                matches = []
                sources = []

                if use_rag:
                    with st.spinner("Searching your PDFs…"):
                        matches = retrieve_matches(pending, st.session_state.active_document_ids)
                        sources = format_sources(matches)
                    if sources:
                        with st.expander("📚 Source citations", expanded=False):
                            render_sources(sources)
                    stream = stream_answer_pdf_question(pending, st.session_state.active_document_ids, matches=matches)
                else:
                    stream = stream_general_answer(pending)

                placeholder = st.empty()
                placeholder.markdown('<div class="typing-dots"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
                answer = st.write_stream(stream)
                placeholder.empty()

                if isinstance(answer, list):
                    answer = "".join(answer)
                answer = (answer or "").strip()

                st.session_state.last_answer = answer
                st.session_state.last_sources = sources
                add_message("assistant", answer, sources=sources)

            st.rerun()

    if st.session_state.last_answer:
        with st.expander("🔊 Listen to the latest answer", expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                tts_language = st.selectbox("Language", ["English", "Hindi"], key="tts_language", label_visibility="collapsed")
            with col2:
                if st.button("Generate speech", key="generate_tts", use_container_width=True):
                    try:
                        audio_bytes = create_tts_audio(st.session_state.last_answer, language=("hi" if tts_language == "Hindi" else "en"))
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                    except Exception as exc:
                        st.error("Text-to-speech failed.")

# =====================================================
# PAGE: UPLOAD & INDEX
# =====================================================
elif page == "📤 Upload & Index":
    st.subheader("📤 Upload PDF documents")
    st.caption("Upload PDFs to extract text and index them for search.")

    uploaded_files = st.file_uploader("Choose one or more PDF files", type=["pdf"], accept_multiple_files=True)
    force_reindex = st.checkbox("Re-index existing documents", value=False)

    if st.button("🚀 Index PDFs", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Pehle kam se kam ek PDF select karein.")
        else:
            rag = get_rag_engine()
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
    try:
        rag = get_rag_engine()
        current_documents = rag.list_documents()
        if current_documents:
            dataframe = pd.DataFrame([{
                "Filename": doc["filename"], "Pages": doc.get("page_count", 0),
                "Words": doc.get("word_count", 0), "Chunks": doc.get("chunk_count", 0),
                "Indexed at": doc.get("indexed_at", ""), "Document ID": doc["document_id"],
            } for doc in current_documents])
            st.dataframe(dataframe, use_container_width=True, hide_index=True)
            delete_names = {f"{d['filename']} — {d['document_id']}": d["document_id"] for d in current_documents}
            selected_to_delete = st.selectbox("Select document to delete", ["Select a document"] + list(delete_names.keys()))
            if st.button("🗑️ Delete selected document"):
                if selected_to_delete != "Select a document":
                    rag.delete_document(delete_names[selected_to_delete])
                    st.success("Document deleted.")
                    st.rerun()
        else:
            st.info("Abhi tak koi PDF index nahi hua.")
    except Exception:
        st.info("Loading documents...")

# =====================================================
# PAGE: PDF SUMMARY
# =====================================================
elif page == "📝 PDF Summary":
    st.subheader("📝 PDF Summary")
    try:
        rag = get_rag_engine()
        documents = rag.list_documents()
        if not documents:
            st.info("Summary generate karne se pehle PDF index karein.")
        else:
            document_options = {item["filename"]: item["document_id"] for item in documents}
            selected_summary_name = st.selectbox("Select PDF", options=list(document_options.keys()), key="summary_pdf")
            summary_document_id = document_options[selected_summary_name]
            if st.button("Generate summary", type="primary"):
                with st.spinner("Generating PDF summary..."):
                    document = rag.documents_metadata.get(summary_document_id)
                    if document:
                        pages = rag.extract_pdf_pages(document["source_path"])
                        sections = [f"[Page {p['page']}]\n{p['text'][:2500]}" for p in pages if p["text"]]
                        text = "\n\n".join(sections)[:24000]
                        prompt = f"Summarize the following PDF content.\n\nPDF: {document['filename']}\n\nContent:\n{text}"
                        summary = llm_text(prompt, "You are an accurate document summarization assistant.")
                        st.markdown(summary)
                        st.download_button("⬇️ Download summary", data=summary, file_name=Path(selected_summary_name).stem + "_summary.txt", mime="text/plain")
                    else:
                        st.error("Document metadata nahi mila.")
    except Exception as e:
        st.error(f"Error loading PDFs: {e}")

# =====================================================
# PAGE: COMPARE PDFs
# =====================================================
elif page == "⚖️ Compare PDFs":
    st.subheader("⚖️ Compare two PDFs")
    try:
        rag = get_rag_engine()
        documents = rag.list_documents()
        if len(documents) < 2:
            st.info("Comparison ke liye kam se kam 2 PDFs index karein.")
        else:
            document_options = {item["filename"]: item["document_id"] for item in documents}
            comparison_names = list(document_options.keys())
            col_a, col_b = st.columns(2)
            with col_a:
                pdf_a_name = st.selectbox("PDF A", options=comparison_names, key="compare_a")
            with col_b:
                pdf_b_name = st.selectbox("PDF B", options=comparison_names, index=1, key="compare_b")
            if pdf_a_name == pdf_b_name:
                st.warning("Please select two different PDFs.")
            elif st.button("Compare PDFs", type="primary"):
                with st.spinner("Comparing documents..."):
                    doc1 = rag.documents_metadata.get(document_options[pdf_a_name])
                    doc2 = rag.documents_metadata.get(document_options[pdf_b_name])
                    if doc1 and doc2:
                        query = "main topics concepts definitions important facts objectives and conclusions"
                        ctx1 = rag.build_context(rag.search(query, document_id=document_options[pdf_a_name], top_k=8), 9000)
                        ctx2 = rag.build_context(rag.search(query, document_id=document_options[pdf_b_name], top_k=8), 9000)
                        prompt = f"Compare these two PDFs.\n\nPDF A: {doc1['filename']}\n{ctx1}\n\nPDF B: {doc2['filename']}\n{ctx2}"
                        comparison = llm_text(prompt, "You compare documents using evidence, not assumptions.")
                        st.markdown(comparison)
                        st.download_button("⬇️ Download comparison", data=comparison, file_name="pdf_comparison.txt", mime="text/plain")
    except Exception as e:
        st.error(f"Error loading PDFs: {e}")

# =====================================================
# PAGE: ANALYTICS
# =====================================================
elif page == "📊 PDF Analytics":
    st.subheader("📊 PDF analytics dashboard")
    try:
        rag = get_rag_engine()
        current_documents = rag.list_documents()
        if not current_documents:
            st.info("Analytics dekhne ke liye PDF upload aur index karein.")
        else:
            analytics_names = {f"{d['filename']} — {d['document_id']}": d["document_id"] for d in current_documents}
            selected_analytics_name = st.selectbox("Select document", options=list(analytics_names.keys()))
            selected_analytics_id = analytics_names[selected_analytics_name]
            
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
            page_df = pd.DataFrame([{
                "Page": p["page"], "Words": p["word_count"], "Characters": p["character_count"]
            } for p in analytics["pages"]])
            st.dataframe(page_df, use_container_width=True, hide_index=True)
            st.bar_chart(page_df.set_index("Page")["Words"])
    except Exception as e:
        st.error(f"Error: {e}")

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
            with st.expander(f"{index + 1}. {role_label} — {message.get('timestamp', '')}"):
                st.markdown(message["content"])
                if message.get("sources"):
                    render_sources(message["sources"])
        
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.session_state.last_answer = ""
            st.session_state.last_sources = []
            st.rerun()

# =====================================================
# PAGE: SETTINGS
# =====================================================
elif page == "⚙️ Settings":
    st.subheader("⚙️ Settings")
    st.markdown("### LLM Provider")
    st.write(f"Active provider: `{MODEL_PROVIDER}`")
    st.write(f"Model: `{LLM_MODEL}`")
    llm_ok, llm_msg = check_llm()
    if llm_ok:
        st.success(llm_msg)
    else:
        st.error(llm_msg)
        st.info("Set `GROQ_API_KEY` in Render Environment variables.")
        
    st.markdown("### PDF search engine")
    st.info("PDF engine is lazy-loaded to save memory. It will start when you upload or search a PDF.")
    
    st.markdown("### Data storage")
    st.write(f"PDF uploads: `{UPLOAD_DIR}`")
    st.write(f"Vector database: `{BASE_DIR / 'chroma_db'}`")

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