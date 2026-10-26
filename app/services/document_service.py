from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    IngestionError,
    PermissionDeniedError,
)
from app.core.logging import get_logger
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.embedder import EmbeddingService
from app.ingestion.parsers.docx_parser import DOCXParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.pipeline import IngestionPipeline
from app.models.chunk import Chunk as ChunkModel
from app.models.document import Document
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore
from app.security.file_validator import validate_upload
from app.security.path_safety import safe_filename, safe_upload_path
from app.utils.hashing import sha256_bytes

logger = get_logger(__name__)
settings = get_settings()


class DocumentService:
    """Orchestrates document upload, validation, ingestion, and deletion."""

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

    def handle_upload(
        self,
        owner_id: str,
        filename: str,
        content: bytes,
        declared_mime: str | None = None,
    ) -> Document:
        """Validate, save, and register the document.  Returns immediately.

        The actual ingestion (parsing, chunking, embedding) is done by
        ``run_ingestion()`` which is called as a FastAPI BackgroundTask.

        Raises:
            UnsupportedFileTypeError, FileTooLargeError: Validation failures.
            DuplicateDocumentError: Same file already exists.
        """
        # Step 1: Validate content
        mime_type = validate_upload(filename, content, declared_mime)

        # Step 2: Hash for dedup check
        file_hash = sha256_bytes(content)
        existing = self._doc_repo.get_by_hash(file_hash)
        if existing:
            raise DuplicateDocumentError(
                f"Document already exists (id={existing.id}). "
                "Use re-indexing if you want to update its chunks."
            )

        # Step 3: Safe filename + save to disk
        safe_name = safe_filename(filename)
        dest = safe_upload_path(settings.upload_dir, safe_name)
        dest.write_bytes(content)
        logger.info("File saved: %s (%d bytes)", dest, len(content))

        # Step 4: Create DB record
        doc = self._doc_repo.create(
            owner_id=owner_id,
            filename=safe_name,
            original_filename=filename,
            file_hash=file_hash,
            mime_type=mime_type,
            file_size_bytes=len(content),
            storage_path=str(dest),
        )
        return doc

    def run_ingestion(self, document_id: str) -> None:
        """Execute the full ingestion pipeline for a document.

        This is designed to run in a FastAPI BackgroundTask (separate thread).
        All database interactions use the session from __init__.

        On any error: marks document status=failed and logs the error without
        exposing internal details to users.
        """
        pipeline = IngestionPipeline(
            db=self._db,
            vector_store=self._vector_store,
            keyword_store=self._keyword_store,
        )
        pipeline.run(document_id)

    def get_document(self, document_id: str, user_id: str, user_role: str) -> Document:
        """Fetch document metadata with authorization check."""
        doc = self._doc_repo.get_by_id(document_id)
        if not doc or doc.status == "deleted":
            raise DocumentNotFoundError()
        if user_role != "admin" and not self._doc_repo.user_can_read(document_id, user_id):
            raise DocumentNotFoundError()  # Don't reveal existence to unauthorized users
        return doc

    def list_documents(
        self, user_id: str, user_role: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[Document], int]:
        """List documents the user has access to."""
        if user_role == "admin":
            from sqlalchemy import select, func
            from app.models.document import Document as DocModel
            stmt = select(DocModel).where(DocModel.status != "deleted")
            total_stmt = select(func.count()).select_from(stmt.subquery())
            total = self._db.execute(total_stmt).scalar_one()
            docs = list(
                self._db.execute(
                    stmt.order_by(DocModel.created_at.desc())
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                ).scalars().all()
            )
            return docs, total
        return self._doc_repo.list_for_user(user_id, page, page_size)

    def delete_document(self, document_id: str, user_id: str, user_role: str) -> None:
        """Soft-delete a document and schedule index cleanup."""
        doc = self.get_document(document_id, user_id, user_role)

        # Only owner or admin can delete
        if user_role != "admin" and doc.owner_id != user_id:
            raise PermissionDeniedError("Only the document owner or an admin can delete this document.")

        # Remove from vector and keyword stores
        chunks = self._chunk_repo.get_by_document(document_id)
        vector_ids = [c.vector_index_id for c in chunks if c.vector_index_id is not None]
        if vector_ids:
            self._vector_store.delete(vector_ids)

        self._keyword_store.delete_document(document_id)
        self._chunk_repo.delete_by_document(document_id)

        # Soft delete the document record
        self._doc_repo.soft_delete(document_id)
        self._vector_store.save()

        logger.info("Document %s deleted by user %s", document_id, user_id)

    def trigger_reindex(self, document_id: str, user_id: str, user_role: str) -> str:
        """Trigger re-ingestion of an already-ingested document.

        Returns the new ingestion job ID.
        """
        doc = self.get_document(document_id, user_id, user_role)
        if user_role != "admin" and doc.owner_id != user_id:
            raise PermissionDeniedError("Only the document owner or an admin can re-index.")

        # Create a new ingestion job — the pipeline handles cleanup of old chunks
        job = self._doc_repo.create_ingestion_job(
            document_id=document_id,
            embedding_model=settings.embedding_model,
            chunker_version=StructureAwareChunker.VERSION,
        )
        self._doc_repo.update_status(document_id, "pending")
        return job.id
