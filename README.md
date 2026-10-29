# 📚 Intelligent Document Question Answering System — RAG & LLM

An AI-powered document question-answering application built using **Retrieval-Augmented Generation (RAG)** and Large Language Models (LLMs).

The application allows users to upload PDF documents, extract text, generate semantic embeddings, retrieve relevant document chunks using FAISS, and generate context-aware answers using a locally running LLM through Ollama.

## 🚀 Features

- **PDF Upload & Processing:** Upload PDF documents and extract readable text using PyPDF2.
- **Text Chunking:** Split document text into overlapping chunks for retrieval.
- **Semantic Embeddings:** Convert document chunks and user questions into numerical vectors using Sentence Transformers.
- **FAISS Vector Search:** Retrieve relevant document chunks using L2 distance.
- **Local LLM Integration:** Generate answers using Ollama and the `llama3.2:1b` model.
- **Voice Input:** Ask questions using microphone-based speech input.
- **Context-Aware Answers:** Generate answers using retrieved PDF content and structured prompts.
- **Interactive Streamlit UI:** Explore document statistics, extracted text, retrieved context, and generated answers.
- **RAG Architecture:** Combine information retrieval with LLM-based answer generation.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web application interface |
| PyPDF2 | PDF text extraction |
| NumPy | Numerical operations |
| Sentence Transformers | Text embeddings |
| FAISS CPU | Vector indexing and search |
| Ollama | Local LLM inference |
| Llama 3.2 1B | Language model |
| Requests | Communication with Ollama API |
| Streamlit Mic Recorder | Voice input |

## ⚙️ Prerequisites

Before running the application, ensure you have:

- Python installed.
- Git installed if cloning the repository.
- Ollama installed and running.
- Sufficient system resources to run the embedding model and local LLM.

### Install Ollama

Download Ollama from its official website:

https://ollama.com/

### Download the LLM

Open Command Prompt or a terminal and run:

```bash
ollama pull llama3.2:1b
```

Verify that the model is available:

```bash
ollama list
```

Ollama normally exposes its local API at:

```text
http://localhost:11434
```

The application uses the `/api/generate` endpoint.

## 📥 Installation

### 1. Clone the repository

Replace `<your-repository-url>` with your actual GitHub repository URL.

```bash
git clone <your-repository-url>
```

Navigate to the project directory:

```bash
cd <your-repository-folder>
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Run the Application

Ensure Ollama is installed and the required model is available.

From the project directory, run:

```bash
streamlit run program.py
```

The application should open in your browser.

If it does not open automatically, use the local address displayed in the terminal.

## 🧠 How It Works

The application follows this RAG workflow:

1. Upload a PDF document.
2. Extract readable text from the document.
3. Split the extracted text into overlapping chunks.
4. Generate embeddings using Sentence Transformers.
5. Store the embeddings in a FAISS vector index.
6. Enter a question using text input or the microphone.
7. Generate an embedding for the question.
8. Retrieve the three nearest document chunks using L2 distance.
9. Pass the retrieved context and question to the local LLM.
10. Display the generated answer in the Streamlit interface.

## 📂 Project Structure

```text
your-project/
│
├── program.py
├── requirements.txt
└── README.md
```

## 🔒 Privacy

The application sends prompts to the locally running Ollama service rather than a cloud-hosted LLM API.

However, microphone speech recognition may involve external processing depending on the library's implementation. Therefore, the complete application should not be described as entirely offline without verifying the speech-recognition behavior.

## ⚠️ Limitations

- Text extraction works with PDFs containing extractable text. Scanned PDFs may require OCR.
- Answer quality depends on the extracted text, retrieved chunks, and language model.
- The application instructs the LLM to answer using the provided PDF context, but responses are not guaranteed to be completely free of hallucinations.
- Ollama and the required model must be installed separately.
- Performance depends on the available hardware and document size.

## 👨‍💻 Author

**Omkar Nanaso Suryawanshi**

Developed as an AI and Generative AI project to explore document processing, vector search, Retrieval-Augmented Generation, and local Large Language Models.

**Technologies:** Python · Streamlit · RAG · FAISS · Sentence Transformers · Ollama

**Tagline:** Beyond Code, Towards Intelligence.

---

⭐ If you find this project useful, consider starring the repository.