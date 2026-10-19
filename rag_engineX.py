import re
import json
import hashlib
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from google import genai



class RAGEngine:
    """
    OS AI Agent RAG Engine (Lightweight for 512MB RAM)
    
    Features:
    - PDF extraction with page numbers
    - Overlapping text chunks
    - Persistent ChromaDB storage with LRU cache (memory safe)
    - Document-specific search
    - Uses Google Gemini API for embeddings (No torch/sentence-transformers)
    """

    def __init__(
        self,
        persist_directory="chroma_db",
        collection_name="OS_ai_Agent_v2",
        chunk_size=900,
        chunk_overlap=180,
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # 1. LRU Cache ke saath ChromaDB Client banayein
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                chroma_segment_cache_policy="LRU",
                chroma_memory_limit_bytes=200 * 1024 * 1024,  # 200MB limit
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "OS AI Agent PDF knowledge base"},
        )

        # 2. Gemini API Client set karein
        self.embedding_model_name = "text-embedding-004"
        gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        if not gemini_api_key:
            print("WARNING: GEMINI_API_KEY nahi mili! PDF search kaam nahi karega.")
            
        self.genai_client = genai.Client(api_key=gemini_api_key)

        self.metadata_file = Path(self.persist_directory) / "documents_metadata.json"
        self.documents_metadata = self._load_metadata()

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    def _load_metadata(self):
        if not self.metadata_file.exists():
            return {}
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_metadata(self):
        temporary_file = self.metadata_file.with_suffix(".tmp")
        with open(temporary_file, "w", encoding="utf-8") as file:
            json.dump(self.documents_metadata, file, indent=2, ensure_ascii=False)
        temporary_file.replace(self.metadata_file)

    # --------------------------------------------------
    # Embeddings via Gemini API
    # --------------------------------------------------
    def _get_embeddings(self, texts):
        """Generate embeddings for a list of strings using Gemini API."""
        if not texts:
            return []
            
        # Gemini API batch embedding
        response = self.genai_client.models.embed_content(
            model=self.embedding_model_name,
            contents=texts,
        )
        # response.embeddings ek list return karta hai
        return [emb.values for emb in response.embeddings]

    # --------------------------------------------------
    # PDF extraction
    # --------------------------------------------------
    @staticmethod
    def _clean_text(text):
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def extract_pdf_pages(self, pdf_path):
        reader = PdfReader(pdf_path)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(
                    "PDF password-protected hai. Pehle PDF unlock karein."
                ) from exc

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = self._clean_text(text)
            pages.append({
                "page": page_number,
                "text": text,
                "character_count": len(text),
                "word_count": len(text.split()),
            })
        return pages

    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------
    def _split_text(self, text):
        text = text.strip()
        if not text:
            return []

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                )
                if boundary > start + int(self.chunk_size * 0.55):
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            next_start = end - self.chunk_overlap
            start = max(next_start, start + 1)

        return chunks

    # --------------------------------------------------
    # Index a PDF
    # --------------------------------------------------
    def add_pdf(self, pdf_path, document_id=None, replace_existing=False):
        pdf_path = str(Path(pdf_path).resolve())
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        filename = Path(pdf_path).name

        hasher = hashlib.sha256()
        with open(pdf_path, "rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                hasher.update(block)
        file_hash = hasher.hexdigest()

        if document_id is None:
            document_id = file_hash[:20]

        if document_id in self.documents_metadata:
            old_metadata = self.documents_metadata[document_id]
            if old_metadata.get("file_hash") == file_hash and not replace_existing:
                return {
                    "success": True, "already_indexed": True,
                    "document_id": document_id, "filename": filename,
                    "message": "This PDF is already indexed.",
                }
            if replace_existing:
                self.delete_document(document_id)

        pages = self.extract_pdf_pages(pdf_path)
        chunks = []
        metadatas = []
        ids = []
        page_count = len(pages)
        total_words = 0
        empty_pages = 0

        for page_data in pages:
            page_number = page_data["page"]
            page_text = page_data["text"]
            total_words += page_data["word_count"]

            if not page_text:
                empty_pages += 1
                continue

            page_chunks = self._split_text(page_text)
            for chunk_index, chunk in enumerate(page_chunks):
                chunk_id = f"{document_id}_p{page_number}_c{chunk_index}"
                chunks.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "document_id": document_id,
                    "filename": filename,
                    "page": page_number,
                    "chunk_index": chunk_index,
                    "source_path": pdf_path,
                })

        if not chunks:
            raise ValueError(
                "PDF se searchable text extract nahi hua. "
                "Agar PDF scanned hai, to OCR required ho sakta hai."
            )

        # Gemini API se embeddings generate karein (Batch size 100 to avoid API limits)
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeddings = self._get_embeddings(batch_chunks)
            embeddings.extend(batch_embeddings)

        # ChromaDB mein insert karein
        for start in range(0, len(chunks), batch_size):
            stop = start + batch_size
            self.collection.upsert(
                ids=ids[start:stop],
                documents=chunks[start:stop],
                metadatas=metadatas[start:stop],
                embeddings=embeddings[start:stop],
            )

        self.documents_metadata[document_id] = {
            "document_id": document_id,
            "filename": filename,
            "source_path": pdf_path,
            "file_hash": file_hash,
            "page_count": page_count,
            "word_count": total_words,
            "character_count": sum(p["character_count"] for p in pages),
            "chunk_count": len(chunks),
            "empty_pages": empty_pages,
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save_metadata()

        return {
            "success": True, "already_indexed": False,
            "document_id": document_id, "filename": filename,
            "page_count": page_count, "word_count": total_words,
            "chunk_count": len(chunks), "empty_pages": empty_pages,
            "message": "PDF indexed successfully.",
        }

    # --------------------------------------------------
    # Search
    # --------------------------------------------------
    def search(self, question, document_id=None, top_k=5):
        question = question.strip()
        if not question or self.collection.count() == 0:
            return []

        # Question ka embedding nikalne ke liye Gemini API use karein
        query_embeddings = self._get_embeddings([question])
        if not query_embeddings:
            return []
        query_embedding = query_embeddings[0]

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(max(int(top_k), 1), max(self.collection.count(), 1)),
            "include": ["documents", "metadatas", "distances"],
        }

        if document_id:
            query_args["where"] = {"document_id": document_id}

        results = self.collection.query(**query_args)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            matches.append({
                "text": text,
                "page": int(metadata.get("page", 0)),
                "filename": metadata.get("filename", "Unknown PDF"),
                "document_id": metadata.get("document_id", ""),
                "score": score,
                "distance": float(distance),
            })
        return matches

    # --------------------------------------------------
    # Build answer context
    # --------------------------------------------------
    def build_context(self, matches, max_characters=10000):
        if not matches:
            return "No relevant PDF passages were found."

        context_parts = []
        current_length = 0
        for index, match in enumerate(matches, start=1):
            passage = (
                f"[Source {index}: {match['filename']}, Page {match['page']}]\n"
                f"{match['text']}\n"
            )
            if current_length + len(passage) > max_characters:
                break
            context_parts.append(passage)
            current_length += len(passage)

        return "\n".join(context_parts)

    # --------------------------------------------------
    # Document management
    # --------------------------------------------------
    def list_documents(self):
        return list(self.documents_metadata.values())

    def delete_document(self, document_id):
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception:
            pass
        self.documents_metadata.pop(document_id, None)
        self._save_metadata()

    def delete_all_documents(self):
        for item in self.list_documents():
            self.delete_document(item["document_id"])

    # --------------------------------------------------
    # PDF statistics
    # --------------------------------------------------
    def get_pdf_analytics(self, document_id):
        metadata = self.documents_metadata.get(document_id)
        if not metadata:
            return None

        pages = self.extract_pdf_pages(metadata["source_path"])
        word_counter = Counter()
        stop_words = {
            "the", "and", "for", "with", "that", "this", "are", "was", "were",
            "you", "your", "from", "have", "has", "had", "not", "but", "can",
            "will", "into", "its", "our", "their", "they", "them", "then",
            "than", "also", "which", "when", "where", "what", "who", "why",
            "how", "all", "any", "one", "two", "use", "using", "used", "per",
            "each", "such", "more", "most", "some", "may", "been", "being",
            "about", "between", "other", "is", "in", "on", "at", "of", "to",
            "a", "an", "or", "as", "by", "it", "be", "if", "we", "he", "she", "i",
        }

        for page in pages:
            words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b", page["text"].lower())
            word_counter.update(word for word in words if word not in stop_words)

        return {
            **metadata,
            "pages": pages,
            "top_keywords": word_counter.most_common(15),
            "average_words_per_page": (
                round(metadata["word_count"] / metadata["page_count"], 2)
                if metadata["page_count"] else 0
            ),
        }