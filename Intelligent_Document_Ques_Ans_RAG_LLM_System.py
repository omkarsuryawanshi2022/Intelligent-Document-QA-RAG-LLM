# ============================================================
# Project Name : Intelligent Document Question Answering System
#                using RAG and Large Language Models
#
# Author      : Omkar Nanaso Suryawanshi
# Date        : 23/10/2026
#
# Description :
# This project allows the user to upload a PDF document and ask
# questions based on its content. The system uses the RAG architecture,
# FAISS vector database, Sentence Transformers embeddings, and a
# local LLM through Ollama to generate context-aware answers.
# ============================================================


import streamlit as st
import numpy as np
import PyPDF2 
import faiss
import requests
from streamlit_mic_recorder import speech_to_text

from sentence_transformers import SentenceTransformer

# ============================================================
# Function Name : extract_text_from_pdf
# Description   : Extracts readable text from an uploaded PDF file.
# ============================================================

def extract_text_from_pdf(pdf_file):
    text = ""

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for page_number, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()

            if page_text:
                text = text + page_text + "\n"

    except Exception as e:
        st.error(f"Error while reading PDF : {e}")

    return text

# ============================================================
# Function Name : split_text_into_chunks
# Description   : Splits large PDF text into overlapping chunks.
# ============================================================

def split_text_into_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap
    
    return chunks

# ============================================================
# Function Name : load_embedding_model
# Description   : Loads the Sentence Transformer model once.
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# Function Name : create_vector_database
# Description   : Converts chunks into embeddings and stores them in FAISS.
# ============================================================


def create_vector_database(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# ============================================================
# Function Name : search_relevant_chunks
# Description   : Retrieves matching PDF chunks for a user query.
# ============================================================


def search_relevant_chunks(question, chunks, index, embedding_model, top_k=3):
    question_embedding = embedding_model.encode([question])

    question_embedding = np.array(question_embedding).astype("float32")

    distances, indices = index.search(question_embedding, top_k)

    relevant_chunks = []

    for i in indices[0]:
        if i < len(chunks):
            relevant_chunks.append(chunks[i])

    return relevant_chunks

# ============================================================
# Function Name : ask_llm
# Description   : Answers questions using retrieved PDF context.
# Project       : Marvellous Infosystems - PDF RAG System
# ============================================================

def ask_llm(question, context):

    prompt = f"""
You are an intelligent document question-answering assistant
created for Marvellous Infosystems.

Instructions:
1. Answer only using the provided PDF context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I could not find the answer in the uploaded PDF."
4. Give a clear and simple explanation.
5. Use bullet points when appropriate.
6. Do not invent facts or information.

Context:
{context}

Question:
{question}

Answer:
"""

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        answer = result.get("response", "").strip()

        if not answer:
            return "The LLM returned an empty response."

        return answer

    except requests.exceptions.ConnectionError:
        return (
            "Ollama is not running. "
            "Please start Ollama and try again."
        )

    except requests.exceptions.Timeout:
        return (
            "The Ollama request timed out. "
            "Please try again."
        )

    except requests.exceptions.HTTPError as e:
        return (
            f"Ollama HTTP error: {e}. "
            f"Details: {response.text}"
        )

    except requests.exceptions.RequestException as e:
        return f"Error communicating with Ollama: {e}"

    except ValueError:
        return "Ollama returned an invalid JSON response."


# ============================================================
# Function Name : get_voice_question
# Description   : Captures a spoken question using the microphone.
# Returns       : Recognized question as text, or None.
# ============================================================

def get_voice_question():

    spoken_text = speech_to_text(
        language="en",
        use_container_width=True,
        just_once=True,
        key="pdf_voice_input"
    )

    if spoken_text and spoken_text.strip():
        return spoken_text.strip()

    return None


# ============================================================
# Function Name : show_project_flow
# Description   : Displays the RAG project flow in the Streamlit UI.
# ============================================================


def show_project_flow():
    st.markdown("""
    ### Complete project flow

    '''text
    PDF Upload

    Extract Text from PDF

    Split text into chunks

    Convert Chunks into Embeddings

    Store Embeddings in FAISS Vector Database

    User asks Question

    Convert Question into Embeddings

    Search similar chunks

    Prepare context

    Send context + Question to LLM

    Generate Final Answer
    '''
""")

# ============================================================
# STREAMLIT WEB APPLICATION - UI ONLY
# Intelligent Document Question Answering System using RAG
# ============================================================

# ============================================================
# Intelligent Document Question Answering System
# UI ONLY - Existing RAG Functions Remain Unchanged
# ============================================================

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Document Question Answering System | Document AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - STYLING ONLY
# ============================================================

st.markdown("""
<style>

/* Application background */
.stApp {
    background-color: #f5f7fb;
    color: #1e293b;
}

/* Main content */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Headings */
h1, h2, h3 {
    color: #163b65 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    border-bottom: 1px solid #dbe3ed;
}

.stTabs [data-baseweb="tab"] {
    color: #334155;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #2563eb !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.55rem 1rem;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Input fields */
.stTextInput input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px dashed #94a3b8;
    border-radius: 10px;
    padding: 12px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #2563eb !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Footer */
.footer-text {
    color: #64748b;
    font-size: 13px;
}

/* Remove Streamlit's default toolbar and footer */
[data-testid="stToolbar"] {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🤖 I Don't Just Code, I Create Intelligence")

st.subheader("Intelligent Document Question Answering System")

st.caption(
    "Ask questions about your PDF documents using "
    "Retrieval-Augmented Generation (RAG) and Large Language Models."
)

st.divider()

# ============================================================
# PROJECT INFORMATION
# ============================================================

with st.container(border=True):

    st.markdown("### Document Intelligence Platform")

    st.write(
        "Upload a PDF, retrieve relevant information, and generate "
        "answers using a locally running language model."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Developer:** Omkar Nanaso Suryawanshi")
        st.markdown("**Organization:** OS TechLabs")

    with col2:
        st.markdown("**Technology:** Python, Streamlit, FAISS, Ollama")
        st.markdown("**Tagline:** Beyond Code, Towards Intelligence.!")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 Project Dashboard")

    st.caption("Intelligent Document QA using RAG")

    st.divider()

    st.subheader("About Project")

    st.write(
        "A document question-answering application that retrieves "
        "relevant PDF content and generates answers using an LLM."
    )

    st.subheader("Technologies")

    st.markdown("""
    - Python
    - Streamlit
    - PyPDF2
    - Sentence Transformers
    - FAISS
    - Ollama
    - Llama 3
    """)

    st.subheader("Concepts")

    st.markdown("""
    - PDF Processing
    - Text Chunking
    - Text Embeddings
    - Vector Search
    - Semantic Retrieval
    - RAG
    - Large Language Models
    - Prompt Engineering
    """)

    st.divider()

    st.subheader("Local LLM")

    st.code("ollama run llama3.2:1b", language="bash")

    st.caption(
        "Ensure Ollama is running and the model name matches "
        "your existing ask_llm() function."
    )


# ============================================================
# NAVIGATION TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📄 Upload & Ask",
    "🔄 Project Flow",
    "🧠 Technical Concepts"
])


# ============================================================
# TAB 1: PDF UPLOAD AND QUESTION ANSWERING
# Existing RAG functionality preserved
# ============================================================

with tab1:

    st.header("Upload PDF Document")

    st.write(
        "Select a PDF document to extract its text and ask questions "
        "about its contents."
    )

    uploaded_file = st.file_uploader(
        "Choose your PDF file",
        type=["pdf"],
        help="Upload a text-based PDF document."
    )

    # Existing function
    embedding_model = load_embedding_model()

    if uploaded_file is not None:

        st.success("PDF uploaded successfully!")

        # ----------------------------------------------------
        # Extract PDF Text
        # ----------------------------------------------------

        with st.spinner("Extracting text from PDF..."):

            pdf_text = extract_text_from_pdf(uploaded_file)

        if not pdf_text.strip():

            st.error(
                "No readable text was found. "
                "Please upload a PDF containing extractable text."
            )

        else:

            # ------------------------------------------------
            # Text Preview
            # ------------------------------------------------

            st.subheader("Extracted Text")

            with st.expander("Preview extracted text"):

                st.text_area(
                    "First 2,000 characters",
                    value=pdf_text[:2000],
                    height=200,
                    key="pdf_preview"
                )

            # ------------------------------------------------
            # Split Text into Chunks
            # ------------------------------------------------

            with st.spinner("Splitting document into chunks..."):

                chunks = split_text_into_chunks(pdf_text)

            # ------------------------------------------------
            # Document Statistics
            # ------------------------------------------------

            st.subheader("Document Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Characters",
                    f"{len(pdf_text):,}"
                )

            with col2:
                st.metric(
                    "Total Chunks",
                    f"{len(chunks):,}"
                )

            with col3:
                st.metric(
                    "Chunk Size",
                    "500 characters"
                )

            # ------------------------------------------------
            # Create FAISS Vector Database
            # ------------------------------------------------

            with st.spinner(
                "Creating embeddings and vector database..."
            ):

                index, embeddings = create_vector_database(
                    chunks,
                    embedding_model
                )

            st.success("Vector database created successfully!")

            st.divider()

            # ------------------------------------------------
            # Ask Questions
            # ------------------------------------------------

            st.subheader("Ask a Question")

            # ------------------------------------------------
            # Voice input: shown only after a readable PDF has
            # been uploaded and its FAISS index has been created.
            # The recognized speech is placed into the same text
            # input used for manually typed questions.
            # ------------------------------------------------

            question_col, mic_col = st.columns([11, 1])

            with mic_col:
                spoken_question = speech_to_text(
                    language="en",
                    start_prompt="🎤",
                    stop_prompt="⏹️",
                    just_once=True,
                    use_container_width=False,
                    key="pdf_voice_input"
                )

            # Update the text-input value before creating that widget.
            if spoken_question and spoken_question.strip():
                st.session_state["document_question"] = spoken_question.strip()

            with question_col:
                question = st.text_input(
                    "Enter your question about the uploaded PDF",
                    placeholder="Example: What are the main topics in this document?",
                    key="document_question"
                )

            if question.strip():

                # --------------------------------------------
                # Retrieve Relevant Chunks
                # --------------------------------------------

                with st.spinner("Searching relevant PDF content..."):

                    relevant_chunks = search_relevant_chunks(
                        question,
                        chunks,
                        index,
                        embedding_model,
                        top_k=3
                    )

                context = "\n\n".join(relevant_chunks)

                # --------------------------------------------
                # Display Retrieved Context
                # --------------------------------------------

                with st.expander("View retrieved context"):

                    st.write(context)

                # --------------------------------------------
                # Generate Answer
                # --------------------------------------------

                with st.spinner("Generating answer using local LLM..."):

                    answer = ask_llm(
                        question,
                        context
                    )

                # --------------------------------------------
                # Display Answer
                # --------------------------------------------

                st.subheader("Answer")

                st.success(answer)


# ============================================================
# TAB 2: PROJECT FLOW
# Existing function preserved
# ============================================================

with tab2:

    st.header("RAG Project Architecture")

    st.write(
        "The application follows a retrieval-augmented generation "
        "workflow to answer questions using information from PDF documents."
    )

    show_project_flow()


# ============================================================
# TAB 3: TECHNICAL CONCEPTS
# ============================================================

with tab3:

    st.header("Technical Concepts")

    st.write(
        "Explore the main technologies used in this project."
    )

    with st.expander("1. PDF Text Extraction", expanded=True):

        st.write(
            "PyPDF2 extracts readable text from PDF documents."
        )

    with st.expander("2. Text Chunking"):

        st.write(
            "Large documents are divided into smaller text chunks "
            "to support efficient retrieval."
        )

    with st.expander("3. Embeddings"):

        st.write(
            "Sentence Transformers converts text into numerical "
            "vectors that represent semantic information."
        )

    with st.expander("4. FAISS Vector Database"):

        st.write(
            "FAISS indexes and searches vectors to identify "
            "document chunks relevant to a user's question."
        )

    with st.expander("5. Semantic Search"):

        st.write(
            "The question embedding is compared with document "
            "embeddings to retrieve relevant content."
        )

    with st.expander("6. Retrieval-Augmented Generation (RAG)"):

        st.write(
            "RAG retrieves relevant information and supplies it "
            "as context to the language model before answer generation."
        )

    with st.expander("7. Large Language Models"):

        st.write(
            "Ollama runs the configured language model locally "
            "to generate answers from the provided prompt."
        )

    with st.expander("8. Prompt Engineering"):

        st.write(
            "Structured instructions, retrieved context, and the "
            "user's question guide the model's response."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "OS TechLabs | Beyond Code, Towards Intelligence."
)

st.caption(
    "Developed by Omkar Nanaso Suryawanshi | "
    "Python • RAG • FAISS • Ollama • Streamlit"
)


