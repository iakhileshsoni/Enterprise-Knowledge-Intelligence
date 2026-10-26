from __future__ import annotations

import traceback
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError, IngestionError
from app.core.logging import get_logger
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.normalizer import normalize_text
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.models.chunk import Chunk as ChunkModel
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)
settings = get_settings()

_MIME_TO_PARSER = {
    "application/pdf": PDFParser,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCXParser,
}


class IngestionPipeline:
    """Executes the complete document ingestion pipeline.

    Atomicity guarantee:
    - Chunks are created in the DB and indexed in FAISS/FTS5.
    - The document status is only set to ``ready`` if ALL steps succeed.
    - On failure, the document status is set to ``failed`` with an error message.
    - Partial chunks from a failed ingestion are cleaned up.
    - The FAISS index is saved to disk only on success.

    Re-indexing support:
    - Old chunks for the document are deleted before new ones are created.
    - The FAISS index maintains a mapping from vector_id → chunk_id; deleted
      mappings are removed (the vector remains but will never be returned due
      to auth filtering since the chunk record is gone).
    """

    def __init__(
        self,
        db: Session,
        vector_store: FAISSVectorStore,
        keyword_store: KeywordStore,
    ) -> None:
        self._db = db
        self._doc_repo = DocumentRepository(db)
        self._chunk_repo = ChunkRepository(db)
        self._vector_store = vector_store
        self._keyword_store = keyword_store

    def run(self, document_id: str) -> None:
        """Run the full ingestion pipeline for a document.

        Args:
            document_id: UUID of the document to ingest.
        """
        doc = self._doc_repo.get_by_id(document_id)
        if not doc:
            logger.error("Ingestion: document %s not found", document_id)
            return

        # Create ingestion job record
        job = self._doc_repo.create_ingestion_job(
            document_id=document_id,
            embedding_model=settings.embedding_model,
            chunker_version=StructureAwareChunker.VERSION,
        )

        self._doc_repo.update_status(document_id, "processing")
        self._doc_repo.update_job_status(job.id, "processing")

        try:
            self._execute(doc, job.id)
        except Exception as exc:
            error_detail = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Ingestion failed for document %s: %s\n%s",
                document_id,
                error_detail,
                traceback.format_exc(),
            )
            self._doc_repo.update_status(
                document_id,
                "failed",
                error_message="Ingestion failed. Check server logs for details.",
            )
            self._doc_repo.update_job_status(job.id, "failed", error_detail=error_detail)
            # Clean up any partial chunks from this failed job
            self._chunk_repo.delete_by_job(job.id)

    def _execute(self, doc, job_id: str) -> None:
        """Internal execution — separated so exceptions propagate cleanly."""
        # ── Step 1: Parser dispatch ────────────────────────────────────────────
        parser_class = _MIME_TO_PARSER.get(doc.mime_type)
        if not parser_class:
            raise IngestionError(f"No parser available for MIME type: {doc.mime_type}")

        parser = parser_class()
        logger.info("Parsing %s (%s)", doc.original_filename, doc.mime_type)
        parsed = parser.parse(doc.storage_path)

        if not parsed.blocks:
            raise IngestionError(
                "No text could be extracted from the document. "
                "It may be empty, image-only, or password-protected."
            )

        # ── Step 2: Clean up old chunks (for re-indexing) ─────────────────────
        old_chunks = self._chunk_repo.get_by_document(doc.id)
        if old_chunks:
            old_vector_ids = [c.vector_index_id for c in old_chunks if c.vector_index_id is not None]
            self._vector_store.delete(old_vector_ids)
            self._keyword_store.delete_document(doc.id)
            self._chunk_repo.delete_by_document(doc.id)
            logger.info(
                "Removed %d old chunks for document %s (re-indexing)",
                len(old_chunks),
                doc.id,
            )

        # ── Step 3: Structure-aware chunking ───────────────────────────────────
        chunker = StructureAwareChunker()
        chunks = chunker.chunk(parsed)
        if not chunks:
            raise IngestionError("Chunker produced zero chunks from the extracted text.")

        logger.info(
            "Chunked %s into %d chunks (size=%d, overlap=%d)",
            doc.original_filename,
            len(chunks),
            settings.chunk_size,
            settings.chunk_overlap,
        )

        # ── Step 4: Embed all chunks ───────────────────────────────────────────
        embedder = EmbeddingService()
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        # ── Step 5: Add to FAISS ───────────────────────────────────────────────
        # We need temporary chunk objects to get chunk UUIDs before adding to FAISS
        # Create DB chunk records first (without vector_index_id)
        chunk_models: list[ChunkModel] = []
        for i, chunk in enumerate(chunks):
            cm = ChunkModel(
                document_id=doc.id,
                ingestion_job_id=job_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                page_number=chunk.page_number,
                section_heading=chunk.section_heading,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                embedding_model=settings.embedding_model,
            )
            chunk_models.append(cm)

        created_chunks = self._chunk_repo.bulk_create(chunk_models)

        # ── Step 6: Index in FAISS + FTS5 ─────────────────────────────────────
        chunk_ids = [c.id for c in created_chunks]
        vector_ids = self._vector_store.add(embeddings, chunk_ids)

        # Update each chunk with its FAISS vector ID
        fts_records: list[tuple[str, str, str]] = []
        for chunk_model, vector_id in zip(created_chunks, vector_ids):
            self._chunk_repo.update_vector_index_id(chunk_model.id, vector_id)
            fts_records.append((chunk_model.id, doc.id, chunk_model.text))

        # ── Step 7: Index in FTS5 ──────────────────────────────────────────────
        self._keyword_store.add_batch(fts_records)

        # ── Step 8: Save FAISS index to disk ──────────────────────────────────
        self._vector_store.save()

        # ── Step 9: Mark success ──────────────────────────────────────────────
        self._doc_repo.update_status(doc.id, "ready")
        self._doc_repo.update_job_status(
            job_id, "completed", chunk_count=len(created_chunks)
        )
        logger.info(
            "Ingestion complete: document=%s chunks=%d",
            doc.id,
            len(created_chunks),
        )
