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
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_TITLE = "OS AI Agent"
OLLAMA_MODEL = "llama3.2:1b"

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploaded_pdfs"
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MAX_UPLOAD_MB = 50
DEFAULT_TOP_K = 5


# =====================================================
# STYLING
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #172033;
    }

    .block-container {
        padding-top: 1.4rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        color: #172554;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 18px;
        background: linear-gradient(
            115deg,
            #dbeafe,
            #ede9fe
        );
        border: 1px solid #dbeafe;
        margin-bottom: 1rem;
    }

    .hero p {
        color: #334155;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
    }

    .small-note {
        color: #64748b;
        font-size: 0.88rem;
    }

    [data-testid="stMetric"] {
        background: white;
        padding: 12px;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
    }
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
        "conversation_id": datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        ),
        "last_uploaded_files": [],
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
        persist_directory=str(
            BASE_DIR / "chroma_db"
        ),
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


def ollama_response(
    prompt,
    system_prompt=None,
    stream=False,
):
    """
    Generate response using the local Ollama model.
    """

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    return ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=stream,
        options={
            "temperature": 0.2,
            "num_ctx": 8192,
        },
    )


def generate_text(prompt, system_prompt=None):
    """
    Non-streaming helper for summaries and comparisons.
    """

    response = ollama_response(
        prompt=prompt,
        system_prompt=system_prompt,
        stream=False,
    )

    return response["message"]["content"].strip()


# =====================================================
# CONVERSATION MEMORY
# =====================================================

def build_conversation_history(limit=10):
    """
    Keep the latest conversation turns in the prompt.

    This is session-level conversation memory.
    It persists across reruns in the same Streamlit session,
    but is not permanent cross-session storage.
    """

    history = st.session_state.messages[-limit:]

    lines = []

    for message in history:
        role = (
            "User"
            if message["role"] == "user"
            else "Assistant"
        )

        content = message["content"]

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)


def contextualize_question(question):
    """
    Rewrite a follow-up into a standalone question
    using recent chat history.
    """

    history = build_conversation_history(
        limit=8
    )

    if not history:
        return question

    prompt = f"""
Rewrite the latest user question as a standalone
search query using the conversation history.

Rules:
- Preserve the original intent.
- Resolve pronouns such as it, they, this, and that.
- Do not answer the question.
- Do not invent facts.
- Return only the rewritten question.

Conversation:
{history}

Latest question:
{question}

Standalone search question:
"""

    try:
        rewritten = generate_text(
            prompt,
            system_prompt=(
                "You rewrite follow-up questions for "
                "retrieval. Return only one question."
            ),
        )

        if rewritten and len(rewritten) < 600:
            return rewritten

    except Exception:
        pass

    return question


def add_message(role, content, sources=None):
    item = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
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
        filename = match.get(
            "filename",
            "Unknown PDF",
        )

        page = match.get(
            "page",
            0,
        )

        key = (filename, page)

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "filename": filename,
                "page": page,
                "score": round(
                    float(match.get("score", 0)),
                    3,
                ),
                "text": match.get("text", ""),
            }
        )

    return sources


def render_sources(sources):
    if not sources:
        st.info(
            "Is answer ke liye koi source passage "
            "retrieve nahi hua."
        )
        return

    st.markdown("#### 📚 Source citations")

    for source in sources:
        st.markdown(
            f"""
            <div class="source-card">
                <b>📄 {source['filename']}</b><br>
                Page: <b>{source['page']}</b><br>
                Similarity score: {source['score']}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            f"View extracted text — Page {source['page']}"
        ):
            st.write(source["text"])


# =====================================================
# RAG QUESTION ANSWERING
# =====================================================

def answer_pdf_question(question, document_ids):
    """
    Retrieve PDF passages and generate a grounded answer.
    """

    standalone_question = contextualize_question(
        question
    )

    all_matches = []

    if not document_ids:
        matches = rag.search(
            question=standalone_question,
            top_k=DEFAULT_TOP_K,
        )

        all_matches.extend(matches)

    else:
        # Search each selected document separately.
        # This helps comparison and multi-PDF questions.
        per_document_k = max(
            2,
            DEFAULT_TOP_K // len(document_ids),
        )

        for document_id in document_ids:
            matches = rag.search(
                question=standalone_question,
                document_id=document_id,
                top_k=per_document_k,
            )

            all_matches.extend(matches)

        # Also search globally to avoid missing a useful
        # passage from a selected PDF.
        global_matches = rag.search(
            question=standalone_question,
            top_k=DEFAULT_TOP_K,
        )

        selected_ids = set(document_ids)

        for match in global_matches:
            if match["document_id"] in selected_ids:
                all_matches.append(match)

    # Deduplicate passages.
    unique_matches = []
    seen = set()

    for match in all_matches:
        key = (
            match["document_id"],
            match["page"],
            match["text"],
        )

        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    unique_matches.sort(
        key=lambda item: item.get(
            "score",
            0,
        ),
        reverse=True,
    )

    matches = unique_matches[:8]

    if not matches:
        return (
            "Mujhe indexed PDFs mein is question ka "
            "relevant content nahi mila. Please PDF "
            "check karein ya question ko specific banayein.",
            [],
        )

    context = rag.build_context(
        matches,
        max_characters=14000,
    )

    history = build_conversation_history(
        limit=8
    )

    prompt = f"""
You are OS AI Agent, a helpful PDF question-answering assistant.

Answer the user's question using the retrieved PDF context.

STRICT RULES:
1. Prioritize facts present in the PDF context.
2. Do not invent facts or page references.
3. If the context does not contain the answer, clearly say so.
4. If you use a fact from a source, cite it as [filename, Page N].
5. If multiple PDFs support the answer, cite each relevant PDF.
6. Explain clearly with headings and bullet points when useful.
7. Use conversation history to resolve follow-up questions.
8. Do not claim that a source supports a fact unless it actually does.
9. Distinguish your general explanation from PDF-derived facts.

Recent conversation:
{history}

Current question:
{question}

Standalone retrieval question:
{standalone_question}

Retrieved PDF context:
{context}

Answer:
"""

    system_prompt = """
You are a careful retrieval-augmented AI Agent.
Accuracy and honest source attribution are more important
than producing an answer when evidence is missing.
"""

    try:
        answer = generate_text(
            prompt,
            system_prompt=system_prompt,
        )

    except Exception as exc:
        answer = (
            "Ollama se response generate nahi hua. "
            "Check karein ki Ollama running hai aur "
            f"model `{OLLAMA_MODEL}` installed hai.\n\n"
            f"Technical detail: {exc}"
        )

    sources = format_sources(matches)

    return answer, sources


# =====================================================
# PDF SUMMARY
# =====================================================

def summarize_pdf(document_id):
    document = rag.documents_metadata.get(
        document_id
    )

    if not document:
        return "Document metadata nahi mila."

    pages = rag.extract_pdf_pages(
        document["source_path"]
    )

    # Keep a bounded amount of extracted content.
    # For very long PDFs, summarize representative sections.
    sections = []

    for page in pages:
        if page["text"]:
            sections.append(
                f"[Page {page['page']}]\n"
                f"{page['text'][:2500]}"
            )

    text = "\n\n".join(sections)

    text = text[:24000]

    prompt = f"""
Summarize the following PDF content.

Provide:
1. Short overview
2. Main topics
3. Important concepts
4. Key facts
5. Useful takeaways

Cite page numbers only when the supplied text identifies them.
Do not invent content.

PDF: {document['filename']}

Content:
{text}
"""

    return generate_text(
        prompt,
        system_prompt=(
            "You are an accurate document summarization assistant."
        ),
    )


# =====================================================
# PDF COMPARISON
# =====================================================

def compare_pdfs(document_id_1, document_id_2):
    document_1 = rag.documents_metadata.get(
        document_id_1
    )

    document_2 = rag.documents_metadata.get(
        document_id_2
    )

    if not document_1 or not document_2:
        return "Dono documents available nahi hain."

    # Use retrieval to keep comparison bounded.
    query = (
        "main topics concepts definitions important "
        "facts objectives and conclusions"
    )

    matches_1 = rag.search(
        query,
        document_id=document_id_1,
        top_k=8,
    )

    matches_2 = rag.search(
        query,
        document_id=document_id_2,
        top_k=8,
    )

    context_1 = rag.build_context(
        matches_1,
        max_characters=9000,
    )

    context_2 = rag.build_context(
        matches_2,
        max_characters=9000,
    )

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

Use only evidence in the supplied content.
Cite filename and page number when supported by the context.
If the retrieved content is insufficient, state the limitation.
"""

    return generate_text(
        prompt,
        system_prompt=(
            "You compare documents using evidence, not assumptions."
        ),
    )


# =====================================================
# TEXT TO SPEECH
# =====================================================

def create_tts_audio(text, language="en"):
    """
    Generate audio using Google Text-to-Speech.
    Requires internet access.
    """

    text = text.strip()

    if not text:
        return None

    # Avoid extremely large requests.
    text = text[:4500]

    audio_buffer = io.BytesIO()

    tts = gTTS(
        text=text,
        lang=language,
        slow=False,
    )

    tts.write_to_fp(audio_buffer)

    audio_buffer.seek(0)

    return audio_buffer.getvalue()


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("📚 OS AI Agent")

    st.caption(
        "PDF learning • AI chat • Voice • Analytics"
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
    )

    st.divider()

    st.markdown("### System status")

    ollama_available = check_ollama()

    if ollama_available:
        st.success("Ollama is reachable")
    else:
        st.warning(
            "Ollama is not reachable"
        )

    st.caption(
        f"Model: {OLLAMA_MODEL}"
    )

    st.caption(
        f"Indexed chunks: {rag.collection.count()}"
    )

    if st.button(
        "🗑️ Clear current chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_answer = ""
        st.session_state.last_sources = []

        st.rerun()


# =====================================================
# HEADER
# =====================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>📚 {APP_TITLE}</h1>
        <p>
        Ask questions from PDFs, summarize documents,
        compare files, remember conversation context,
        and explore document analytics.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# DOCUMENT SELECTOR
# =====================================================

documents = rag.list_documents()

document_options = {
    item["filename"]: item["document_id"]
    for item in documents
}

if documents:
    with st.expander(
        "📚 Select PDFs for questions",
        expanded=True,
    ):
        selected_names = st.multiselect(
            "Search only selected PDFs",
            options=list(document_options.keys()),
            default=[
                name
                for name in st.session_state.active_document_names
                if name in document_options
            ],
            help=(
                "Agar koi PDF select nahi hai, "
                "to search poori indexed knowledge base mein hogi."
            ),
        )

        selected_document_ids = [
            document_options[name]
            for name in selected_names
        ]

        st.session_state.active_document_names = (
            selected_names
        )

        st.session_state.active_document_ids = (
            selected_document_ids
        )

else:
    st.info(
        "Abhi koi PDF indexed nahi hai. "
        "Pehle Upload & Index tab mein PDF add karein."
    )


# =====================================================
# HOME & CHAT
# =====================================================

if page == "🏠 Home & Chat":

    left, right = st.columns(
        [3, 1]
    )

    with left:
        st.subheader("💬 Ask your PDF")

    with right:
        st.metric(
            "Indexed PDFs",
            len(documents),
        )

    # Render conversation history.
    for index, message in enumerate(
        st.session_state.messages
    ):
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

            if message.get("sources"):
                with st.expander(
                    "📚 Sources",
                    expanded=False,
                ):
                    render_sources(
                        message["sources"]
                    )

    st.markdown("### 🎙️ Voice input")

    try:
        voice_text = speech_to_text(
            language="en",
            start_prompt="🎙️ Start recording",
            stop_prompt="⏹️ Stop recording",
            just_once=True,
            use_container_width=True,
            key="voice_input",
        )

    except Exception as exc:
        voice_text = None

        st.caption(
            "Voice input component available nahi hua. "
            "Browser microphone permission check karein."
        )

    if voice_text:
        st.session_state["voice_question"] = (
            voice_text
        )

    voice_question = st.session_state.pop(
        "voice_question",
        "",
    )

    question = st.chat_input(
        "Ask something about your PDF..."
    )

    if voice_question and not question:
        question = voice_question

    if question:
        question = question.strip()

        if not question:
            st.stop()

        add_message(
            "user",
            question,
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching PDFs and generating answer..."
            ):
                answer, sources = answer_pdf_question(
                    question,
                    st.session_state.active_document_ids,
                )

            st.markdown(answer)

            if sources:
                with st.expander(
                    "📚 Source citations",
                    expanded=True,
                ):
                    render_sources(sources)

            st.session_state.last_answer = answer
            st.session_state.last_sources = sources

            add_message(
                "assistant",
                answer,
                sources=sources,
            )

            st.rerun()

    # Audio player for the latest answer.
    if st.session_state.last_answer:
        st.divider()

        st.subheader("🔊 Listen to the latest answer")

        tts_language = st.selectbox(
            "Audio language",
            options=[
                "English",
                "Hindi",
            ],
            key="tts_language",
        )

        language_code = (
            "hi"
            if tts_language == "Hindi"
            else "en"
        )

        if st.button(
            "Generate speech",
            key="generate_tts",
        ):
            try:
                with st.spinner(
                    "Generating audio..."
                ):
                    audio_bytes = create_tts_audio(
                        st.session_state.last_answer,
                        language=language_code,
                    )

                if audio_bytes:
                    st.audio(
                        audio_bytes,
                        format="audio/mp3",
                    )

            except Exception as exc:
                st.error(
                    "Text-to-speech failed. "
                    "Internet connection aur gTTS package check karein."
                )
                st.caption(str(exc))


# =====================================================
# UPLOAD & INDEX
# =====================================================

elif page == "📤 Upload & Index":

    st.subheader("📤 Upload PDF documents")

    st.write(
        "PDF upload karke uska text extract aur index karein."
    )

    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            f"Maximum recommended size: {MAX_UPLOAD_MB} MB per file."
        ),
    )

    force_reindex = st.checkbox(
        "Re-index existing documents",
        value=False,
        help=(
            "Use this if you want to replace the existing "
            "indexed content for the same document."
        ),
    )

    if st.button(
        "🚀 Index PDFs",
        type="primary",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning(
                "Pehle kam se kam ek PDF select karein."
            )

        else:
            progress = st.progress(0)

            successful = 0
            failures = []

            for index, uploaded_file in enumerate(
                uploaded_files
            ):
                try:
                    if (
                        uploaded_file.size
                        > MAX_UPLOAD_MB * 1024 * 1024
                    ):
                        raise ValueError(
                            f"File {MAX_UPLOAD_MB} MB limit se badi hai."
                        )

                    destination = (
                        UPLOAD_DIR / uploaded_file.name
                    )

                    # Save the uploaded PDF.
                    destination.write_bytes(
                        uploaded_file.getvalue()
                    )

                    with st.spinner(
                        f"Indexing {uploaded_file.name}..."
                    ):
                        result = rag.add_pdf(
                            str(destination),
                            replace_existing=force_reindex,
                        )

                    successful += 1

                    st.success(
                        f"{uploaded_file.name}: "
                        f"{result['message']}"
                    )

                    if not result.get(
                        "already_indexed",
                        False,
                    ):
                        st.write(
                            {
                                "Pages": result.get(
                                    "page_count"
                                ),
                                "Words": result.get(
                                    "word_count"
                                ),
                                "Chunks": result.get(
                                    "chunk_count"
                                ),
                                "Empty pages": result.get(
                                    "empty_pages"
                                ),
                            }
                        )

                except Exception as exc:
                    failures.append(
                        (
                            uploaded_file.name,
                            str(exc),
                        )
                    )

                    st.error(
                        f"Could not index {uploaded_file.name}: {exc}"
                    )

                progress.progress(
                    (index + 1) / len(uploaded_files)
                )

            st.success(
                f"Completed: {successful} PDF(s) processed."
            )

            if failures:
                st.warning(
                    f"{len(failures)} file(s) could not be indexed."
                )

            st.rerun()

    st.divider()

    st.subheader("📚 Indexed documents")

    current_documents = rag.list_documents()

    if current_documents:
        dataframe = pd.DataFrame(
            [
                {
                    "Filename": doc["filename"],
                    "Pages": doc.get("page_count", 0),
                    "Words": doc.get("word_count", 0),
                    "Chunks": doc.get("chunk_count", 0),
                    "Indexed at": doc.get("indexed_at", ""),
                    "Document ID": doc["document_id"],
                }
                for doc in current_documents
            ]
        )

        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )

        delete_names = {
            f"{doc['filename']} — {doc['document_id']}":
            doc["document_id"]
            for doc in current_documents
        }

        selected_to_delete = st.selectbox(
            "Select document to delete",
            options=[
                "Select a document"
            ] + list(delete_names.keys()),
        )

        if st.button(
            "🗑️ Delete selected document",
        ):
            if selected_to_delete != "Select a document":
                document_id = delete_names[
                    selected_to_delete
                ]

                rag.delete_document(
                    document_id
                )

                st.success(
                    "Document deleted from the active RAG collection."
                )

                st.rerun()

    else:
        st.info(
            "Abhi tak koi PDF index nahi hua."
        )


# =====================================================
# PDF SUMMARY
# =====================================================

elif page == "📝 PDF Summary":

    st.subheader("📝 PDF Summary")

    if not documents:
        st.info(
            "Summary generate karne se pehle PDF index karein."
        )

    else:
        selected_summary_name = st.selectbox(
            "Select PDF",
            options=list(document_options.keys()),
            key="summary_pdf",
        )

        summary_document_id = document_options[
            selected_summary_name
        ]

        if st.button(
            "Generate summary",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Generating PDF summary..."
                ):
                    summary = summarize_pdf(
                        summary_document_id
                    )

                st.markdown(summary)

                st.download_button(
                    "⬇️ Download summary",
                    data=summary,
                    file_name=(
                        Path(selected_summary_name).stem
                        + "_summary.txt"
                    ),
                    mime="text/plain",
                )

            except Exception as exc:
                st.error(
                    f"Summary generation failed: {exc}"
                )


# =====================================================
# PDF COMPARISON
# =====================================================

elif page == "⚖️ Compare PDFs":

    st.subheader("⚖️ Compare two PDFs")

    if len(documents) < 2:
        st.info(
            "Comparison ke liye kam se kam 2 PDFs index karein."
        )

    else:
        comparison_names = list(
            document_options.keys()
        )

        col_a, col_b = st.columns(2)

        with col_a:
            pdf_a_name = st.selectbox(
                "PDF A",
                options=comparison_names,
                key="compare_a",
            )

        with col_b:
            pdf_b_name = st.selectbox(
                "PDF B",
                options=comparison_names,
                index=1,
                key="compare_b",
            )

        if pdf_a_name == pdf_b_name:
            st.warning(
                "Please select two different PDFs."
            )

        elif st.button(
            "Compare PDFs",
            type="primary",
        ):
            try:
                with st.spinner(
                    "Comparing documents..."
                ):
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
                st.error(
                    f"Comparison failed: {exc}"
                )


# =====================================================
# PDF ANALYTICS
# =====================================================

elif page == "📊 PDF Analytics":

    st.subheader("📊 PDF analytics dashboard")

    current_documents = rag.list_documents()

    if not current_documents:
        st.info(
            "Analytics dekhne ke liye PDF upload aur index karein."
        )

    else:
        analytics_names = {
            f"{doc['filename']} — {doc['document_id']}":
            doc["document_id"]
            for doc in current_documents
        }

        selected_analytics_name = st.selectbox(
            "Select document",
            options=list(analytics_names.keys()),
        )

        selected_analytics_id = analytics_names[
            selected_analytics_name
        ]

        try:
            analytics = rag.get_pdf_analytics(
                selected_analytics_id
            )

            if not analytics:
                st.warning(
                    "Analytics metadata available nahi hai."
                )
                st.stop()

            metric_1, metric_2, metric_3, metric_4 = (
                st.columns(4)
            )

            metric_1.metric(
                "Pages",
                analytics["page_count"],
            )

            metric_2.metric(
                "Words",
                f"{analytics['word_count']:,}",
            )

            metric_3.metric(
                "Chunks",
                analytics["chunk_count"],
            )

            metric_4.metric(
                "Avg. words/page",
                analytics["average_words_per_page"],
            )

            st.divider()

            st.markdown("### 📄 Page-wise word count")

            page_df = pd.DataFrame(
                [
                    {
                        "Page": page_data["page"],
                        "Words": page_data["word_count"],
                        "Characters": page_data["character_count"],
                    }
                    for page_data in analytics["pages"]
                ]
            )

            st.dataframe(
                page_df,
                use_container_width=True,
                hide_index=True,
            )

            chart_df = page_df.set_index(
                "Page"
            )

            st.bar_chart(
                chart_df["Words"]
            )

            st.markdown("### 🔤 Top keywords")

            keywords = analytics["top_keywords"]

            if keywords:
                keyword_df = pd.DataFrame(
                    keywords,
                    columns=[
                        "Keyword",
                        "Frequency",
                    ],
                )

                st.dataframe(
                    keyword_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.bar_chart(
                    keyword_df.set_index(
                        "Keyword"
                    )["Frequency"]
                )

            else:
                st.info(
                    "Document mein keyword statistics generate karne ke liye "
                    "sufficient extractable text nahi hai."
                )

            st.markdown("### ℹ️ Document information")

            st.json(
                {
                    "filename": analytics["filename"],
                    "document_id": analytics["document_id"],
                    "indexed_at": analytics.get("indexed_at"),
                    "empty_pages": analytics.get("empty_pages"),
                    "character_count": analytics.get("character_count"),
                    "source_path": analytics.get("source_path"),
                }
            )

            csv_bytes = page_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download page analytics CSV",
                data=csv_bytes,
                file_name="pdf_page_analytics.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(
                f"Analytics generate nahi hui: {exc}"
            )


# =====================================================
# CHAT HISTORY
# =====================================================

elif page == "🕘 Chat History":

    st.subheader("🕘 Current session chat history")

    messages = st.session_state.messages

    if not messages:
        st.info(
            "Abhi current session mein chat history nahi hai."
        )

    else:
        st.caption(
            f"Messages: {len(messages)}"
        )

        for index, message in enumerate(messages):
            role_label = (
                "You"
                if message["role"] == "user"
                else "AI Agent"
            )

            with st.expander(
                f"{index + 1}. {role_label} — "
                f"{message.get('timestamp', '')}"
            ):
                st.markdown(
                    message["content"]
                )

                if message.get("sources"):
                    render_sources(
                        message["sources"]
                    )

        export_data = json.dumps(
            messages,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            "⬇️ Export chat history JSON",
            data=export_data,
            file_name=(
                f"chat_history_"
                f"{st.session_state.conversation_id}.json"
            ),
            mime="application/json",
        )

        if st.button(
            "Clear chat history",
        ):
            st.session_state.messages = []
            st.session_state.last_answer = ""
            st.session_state.last_sources = []

            st.rerun()

    st.warning(
        "Chat history current Streamlit session mein rakhi jaati hai. "
        "Browser/session reset hone par permanent chat history guarantee "
        "nahi hai. Zaroori chats ko JSON mein export karein."
    )


# =====================================================
# SETTINGS
# =====================================================

elif page == "⚙️ Settings":

    st.subheader("⚙️ Settings")

    st.markdown("### Ollama")

    st.write(
        f"Selected model: `{OLLAMA_MODEL}`"
    )

    if st.button(
        "Check Ollama connection",
    ):
        if check_ollama():
            st.success(
                "Ollama server reachable hai."
            )
        else:
            st.error(
                "Ollama server reachable nahi hai. "
                "Terminal mein `ollama serve` check karein."
            )

    st.markdown("### PDF search engine")

    st.write(
        f"Embedding model: "
        "`{rag.embedding_model_name}`"
    )

    st.write(
        f"ChromaDB collection: `{rag.collection_name}`"
    )

    st.write(
        f"Indexed chunks: `{rag.collection.count()}`"
    )

    st.markdown("### Conversation memory")

    st.write(
        "Conversation memory recent messages ko follow-up "
        "questions resolve karne ke liye use karti hai."
    )

    st.write(
        "Memory current session tak limited hai."
    )

    st.markdown("### Voice")

    st.write(
        "Voice input browser microphone aur speech-recognition "
        "service par depend karta hai."
    )

    st.write(
        "Text-to-speech gTTS use karta hai aur internet connection "
        "required hai."
    )

    st.markdown("### Data storage")

    st.write(
        f"PDF uploads: `{UPLOAD_DIR}`"
    )

    st.write(
        f"Vector database: `{BASE_DIR / 'chroma_db'}`"
    )

    st.warning(
        "Database files manually delete karne se pehle backup lein. "
        "Current collection ke indexed records delete karne par "
        "unhe dobara index karna pad sakta hai."
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="small-note">
    OS AI Agent • Streamlit • Ollama • ChromaDB
    <br>
    Answers depend on the extracted PDF text and retrieved passages.
    Always verify important facts against the original document.
    </div>
    """,
    unsafe_allow_html=True,
)