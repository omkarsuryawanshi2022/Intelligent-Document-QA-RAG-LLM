import re
import json
import hashlib
from pathlib import Path
from collections import Counter
from datetime import datetime

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class RAGEngine:
    """
    OS AI Agent RAG Engine

    Features:
    - PDF extraction with page numbers
    - Overlapping text chunks
    - Persistent ChromaDB storage
    - Document-specific search
    - Accurate page citations
    - PDF statistics
    - Document comparison support
    """

    def __init__(
        self,
        persist_directory="chroma_db",
        collection_name="OS_ai_Agent_v2",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=900,
        chunk_overlap=180,
    ):
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "OS AI Agent PDF knowledge base"
            },
        )

        self.embedding_model_name = embedding_model
        self.embedding_model = None

        self.metadata_file = (
            Path(self.persist_directory) / "documents_metadata.json"
        )

        self.documents_metadata = self._load_metadata()

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def _load_metadata(self):
        if not self.metadata_file.exists():
            return {}

        try:
            with open(
                self.metadata_file,
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)

        except (json.JSONDecodeError, OSError):
            return {}

    def _save_metadata(self):
        temporary_file = self.metadata_file.with_suffix(".tmp")

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.documents_metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_file.replace(self.metadata_file)

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name
            )

        return self.embedding_model

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
        """
        Extract text page by page.
        The page number is the original PDF page number, starting at 1.
        """

        reader = PdfReader(pdf_path)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError(
                    "PDF password-protected hai. "
                    "Pehle PDF unlock karein."
                ) from exc

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            text = self._clean_text(text)

            pages.append(
                {
                    "page": page_number,
                    "text": text,
                    "character_count": len(text),
                    "word_count": len(text.split()),
                }
            )

        return pages

    # --------------------------------------------------
    # Chunking
    # --------------------------------------------------

    def _split_text(self, text):
        """
        Split text into overlapping chunks.
        """

        text = text.strip()

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(
                start + self.chunk_size,
                len(text),
            )

            # Prefer a natural sentence or paragraph boundary.
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                )

                if boundary > start + int(
                    self.chunk_size * 0.55
                ):
                    end = boundary + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= len(text):
                break

            next_start = end - self.chunk_overlap

            start = max(
                next_start,
                start + 1,
            )

        return chunks

    # --------------------------------------------------
    # Index a PDF
    # --------------------------------------------------

    def add_pdf(
        self,
        pdf_path,
        document_id=None,
        replace_existing=False,
    ):
        pdf_path = str(Path(pdf_path).resolve())

        if not Path(pdf_path).exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        filename = Path(pdf_path).name

        # Stable document ID based on file contents.
        hasher = hashlib.sha256()

        with open(pdf_path, "rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                hasher.update(block)

        file_hash = hasher.hexdigest()

        if document_id is None:
            document_id = file_hash[:20]

        if document_id in self.documents_metadata:
            old_metadata = self.documents_metadata[document_id]

            if (
                old_metadata.get("file_hash") == file_hash
                and not replace_existing
            ):
                return {
                    "success": True,
                    "already_indexed": True,
                    "document_id": document_id,
                    "filename": filename,
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

            for chunk_index, chunk in enumerate(
                page_chunks
            ):
                chunk_id = (
                    f"{document_id}_p{page_number}"
                    f"_c{chunk_index}"
                )

                chunks.append(chunk)

                ids.append(chunk_id)

                metadatas.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "page": page_number,
                        "chunk_index": chunk_index,
                        "source_path": pdf_path,
                    }
                )

        if not chunks:
            raise ValueError(
                "PDF se searchable text extract nahi hua. "
                "Agar PDF scanned hai, to OCR required ho sakta hai."
            )

        model = self._get_embedding_model()

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        # ChromaDB supports batches. Use smaller batches
        # for large PDFs.
        batch_size = 100

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
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
            "character_count": sum(
                page["character_count"]
                for page in pages
            ),
            "chunk_count": len(chunks),
            "empty_pages": empty_pages,
            "indexed_at": datetime.now().isoformat(
                timespec="seconds"
            ),
        }

        self._save_metadata()

        return {
            "success": True,
            "already_indexed": False,
            "document_id": document_id,
            "filename": filename,
            "page_count": page_count,
            "word_count": total_words,
            "chunk_count": len(chunks),
            "empty_pages": empty_pages,
            "message": "PDF indexed successfully.",
        }

    # --------------------------------------------------
    # Search
    # --------------------------------------------------

    def search(
        self,
        question,
        document_id=None,
        top_k=5,
    ):
        """
        Semantic search with source page metadata.

        Returns a list of dictionaries:
        {
            "text": "...",
            "page": 1,
            "filename": "...",
            "document_id": "...",
            "score": 0.78
        }
        """

        question = question.strip()

        if not question:
            return []

        if self.collection.count() == 0:
            return []

        model = self._get_embedding_model()

        query_embedding = model.encode(
            [question],
            normalize_embeddings=True,
        )[0].tolist()

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": min(
                max(int(top_k), 1),
                max(self.collection.count(), 1),
            ),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if document_id:
            query_args["where"] = {
                "document_id": document_id
            }

        results = self.collection.query(
            **query_args
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []

        for text, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            metadata = metadata or {}

            # Chroma's default distance is cosine distance
            # for collections configured with cosine space.
            # Similarity is an approximate ranking score,
            # not a probability.
            score = max(
                0.0,
                min(1.0, 1.0 - float(distance)),
            )

            matches.append(
                {
                    "text": text,
                    "page": int(
                        metadata.get("page", 0)
                    ),
                    "filename": metadata.get(
                        "filename",
                        "Unknown PDF",
                    ),
                    "document_id": metadata.get(
                        "document_id",
                        "",
                    ),
                    "score": score,
                    "distance": float(distance),
                }
            )

        return matches

    # --------------------------------------------------
    # Build answer context
    # --------------------------------------------------

    def build_context(
        self,
        matches,
        max_characters=14000,
    ):
        if not matches:
            return "No relevant PDF passages were found."

        context_parts = []
        current_length = 0

        for index, match in enumerate(
            matches,
            start=1,
        ):
            passage = (
                f"[Source {index}: "
                f"{match['filename']}, "
                f"Page {match['page']}]\n"
                f"{match['text']}\n"
            )

            if (
                current_length + len(passage)
                > max_characters
            ):
                break

            context_parts.append(passage)
            current_length += len(passage)

        return "\n".join(context_parts)

    # --------------------------------------------------
    # Document management
    # --------------------------------------------------

    def list_documents(self):
        return list(
            self.documents_metadata.values()
        )

    def delete_document(self, document_id):
        try:
            self.collection.delete(
                where={
                    "document_id": document_id
                }
            )

        except Exception:
            pass

        self.documents_metadata.pop(
            document_id,
            None,
        )

        self._save_metadata()

    def delete_all_documents(self):
        for item in self.list_documents():
            self.delete_document(
                item["document_id"]
            )

    # --------------------------------------------------
    # PDF statistics
    # --------------------------------------------------

    def get_pdf_analytics(self, document_id):
        metadata = self.documents_metadata.get(
            document_id
        )

        if not metadata:
            return None

        pages = self.extract_pdf_pages(
            metadata["source_path"]
        )

        word_counter = Counter()

        stop_words = {
            "the", "and", "for", "with", "that",
            "this", "are", "was", "were", "you",
            "your", "from", "have", "has", "had",
            "not", "but", "can", "will", "into",
            "its", "our", "their", "they", "them",
            "then", "than", "also", "which", "when",
            "where", "what", "who", "why", "how",
            "all", "any", "one", "two", "use",
            "using", "used", "per", "each", "such",
            "more", "most", "some", "may", "been",
            "being", "about", "between", "other",
            "is", "in", "on", "at", "of", "to",
            "a", "an", "or", "as", "by", "it",
            "be", "if", "we", "he", "she", "i",
        }

        for page in pages:
            words = re.findall(
                r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b",
                page["text"].lower(),
            )

            word_counter.update(
                word
                for word in words
                if word not in stop_words
            )

        return {
            **metadata,
            "pages": pages,
            "top_keywords": word_counter.most_common(15),
            "average_words_per_page": (
                round(
                    metadata["word_count"]
                    / metadata["page_count"],
                    2,
                )
                if metadata["page_count"]
                else 0
            ),
        }