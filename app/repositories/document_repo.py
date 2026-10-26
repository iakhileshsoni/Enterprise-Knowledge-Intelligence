from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import Document, DocumentPermission, IngestionJob


class DocumentRepository:
    """Data-access layer for Document, DocumentPermission, and IngestionJob records."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Document CRUD ─────────────────────────────────────────────────────────

    def create(
        self,
        owner_id: str,
        filename: str,
        original_filename: str,
        file_hash: str,
        mime_type: str,
        file_size_bytes: int,
        storage_path: str,
    ) -> Document:
        doc = Document(
            owner_id=owner_id,
            filename=filename,
            original_filename=original_filename,
            file_hash=file_hash,
            mime_type=mime_type,
            file_size_bytes=file_size_bytes,
            storage_path=storage_path,
            status="pending",
        )
        self._db.add(doc)
        self._db.commit()
        self._db.refresh(doc)
        return doc

    def get_by_id(self, document_id: str) -> Optional[Document]:
        return self._db.get(Document, document_id)

    def get_by_hash(self, file_hash: str) -> Optional[Document]:
        stmt = select(Document).where(
            Document.file_hash == file_hash,
            Document.status != "deleted",
        )
        return self._db.execute(stmt).scalar_one_or_none()

    def list_for_user(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Document], int]:
        """Return documents the user owns OR has explicit permission to read."""
        owned_stmt = select(Document.id).where(Document.owner_id == user_id)
        permitted_stmt = select(DocumentPermission.document_id).where(
            DocumentPermission.user_id == user_id
        )
        allowed_ids_stmt = owned_stmt.union(permitted_stmt)

        base = (
            select(Document)
            .where(Document.id.in_(allowed_ids_stmt))
            .where(Document.status != "deleted")
        )

        total = self._db.execute(
            select(func.count()).select_from(base.subquery())
        ).scalar_one()

        docs = (
            self._db.execute(
                base.order_by(Document.created_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            .scalars()
            .all()
        )
        return list(docs), total

    def update_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        doc = self.get_by_id(document_id)
        if doc:
            doc.status = status
            if error_message is not None:
                doc.error_message = error_message
            self._db.commit()

    def soft_delete(self, document_id: str) -> None:
        self.update_status(document_id, "deleted")

    def increment_version(self, document_id: str) -> int:
        doc = self.get_by_id(document_id)
        if doc:
            doc.version += 1
            self._db.commit()
            return doc.version
        return 1

    def user_can_read(self, document_id: str, user_id: str) -> bool:
        """Return True if the user owns or has explicit read permission."""
        doc = self.get_by_id(document_id)
        if not doc or doc.status == "deleted":
            return False
        if doc.owner_id == user_id:
            return True
        stmt = select(DocumentPermission).where(
            DocumentPermission.document_id == document_id,
            DocumentPermission.user_id == user_id,
        )
        return self._db.execute(stmt).scalar_one_or_none() is not None

    def get_authorized_document_ids(self, user_id: str, role: str) -> list[str]:
        """Return all document IDs the user is allowed to query.

        Admins can query all non-deleted documents.
        """
        if role == "admin":
            stmt = select(Document.id).where(Document.status == "ready")
            return list(self._db.execute(stmt).scalars().all())

        owned = select(Document.id).where(
            Document.owner_id == user_id,
            Document.status == "ready",
        )
        permitted_doc_ids = select(DocumentPermission.document_id).where(
            DocumentPermission.user_id == user_id
        )
        permitted = select(Document.id).where(
            Document.id.in_(permitted_doc_ids),
            Document.status == "ready",
        )
        all_ids_stmt = owned.union(permitted)
        return list(self._db.execute(all_ids_stmt).scalars().all())

    # ── DocumentPermission ────────────────────────────────────────────────────

    def grant_permission(
        self,
        document_id: str,
        user_id: str,
        granted_by: str,
        permission: str = "read",
    ) -> DocumentPermission:
        perm = DocumentPermission(
            document_id=document_id,
            user_id=user_id,
            permission=permission,
            granted_by=granted_by,
        )
        self._db.add(perm)
        self._db.commit()
        self._db.refresh(perm)
        return perm

    # ── IngestionJob ──────────────────────────────────────────────────────────

    def create_ingestion_job(
        self,
        document_id: str,
        embedding_model: str,
        chunker_version: str,
    ) -> IngestionJob:
        job = IngestionJob(
            document_id=document_id,
            status="pending",
            embedding_model=embedding_model,
            chunker_version=chunker_version,
        )
        self._db.add(job)
        self._db.commit()
        self._db.refresh(job)
        return job

    def update_job_status(
        self,
        job_id: str,
        status: str,
        chunk_count: Optional[int] = None,
        error_detail: Optional[str] = None,
    ) -> None:
        job = self._db.get(IngestionJob, job_id)
        if job:
            job.status = status
            if status == "processing":
                job.started_at = datetime.now(timezone.utc)
            elif status in ("completed", "failed"):
                job.completed_at = datetime.now(timezone.utc)
            if chunk_count is not None:
                job.chunk_count = chunk_count
            if error_detail is not None:
                job.error_detail = error_detail
            self._db.commit()

    def get_latest_job(self, document_id: str) -> Optional[IngestionJob]:
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.id.desc())
            .limit(1)
        )
        return self._db.execute(stmt).scalar_one_or_none()
