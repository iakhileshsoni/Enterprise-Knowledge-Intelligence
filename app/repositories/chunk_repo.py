from __future__ import annotations

from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models.chunk import Chunk


class ChunkRepository:
    """Data-access layer for Chunk records."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]:
        """Persist a batch of Chunk ORM objects in a single transaction."""
        self._db.add_all(chunks)
        self._db.commit()
        for c in chunks:
            self._db.refresh(c)
        return chunks

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        return self._db.get(Chunk, chunk_id)

    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(Chunk.id.in_(chunk_ids))
        return list(self._db.execute(stmt).scalars().all())

    def get_by_document(self, document_id: str) -> list[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index)
        )
        return list(self._db.execute(stmt).scalars().all())

    def get_by_vector_index_ids(
        self, vector_index_ids: list[int], authorized_doc_ids: list[str]
    ) -> list[Chunk]:
        """Fetch chunks by their FAISS integer IDs, filtered to authorized documents.

        This is the authorization gate: only chunks whose document_id appears in
        ``authorized_doc_ids`` are returned, regardless of what FAISS returned.
        """
        if not vector_index_ids or not authorized_doc_ids:
            return []
        stmt = select(Chunk).where(
            Chunk.vector_index_id.in_(vector_index_ids),
            Chunk.document_id.in_(authorized_doc_ids),
        )
        return list(self._db.execute(stmt).scalars().all())

    def delete_by_document(self, document_id: str) -> int:
        """Delete all chunks for a document. Returns number of deleted rows."""
        chunks = self.get_by_document(document_id)
        count = len(chunks)
        for c in chunks:
            self._db.delete(c)
        self._db.commit()
        return count

    def delete_by_job(self, job_id: str) -> int:
        """Delete chunks produced by a specific ingestion job (for failed re-index rollback)."""
        stmt = select(Chunk).where(Chunk.ingestion_job_id == job_id)
        chunks = list(self._db.execute(stmt).scalars().all())
        count = len(chunks)
        for c in chunks:
            self._db.delete(c)
        self._db.commit()
        return count

    def update_vector_index_id(self, chunk_id: str, vector_index_id: int) -> None:
        chunk = self.get_by_id(chunk_id)
        if chunk:
            chunk.vector_index_id = vector_index_id
            self._db.commit()

    def count_by_document(self, document_id: str) -> int:
        stmt = select(Chunk).where(Chunk.document_id == document_id)
        return len(self._db.execute(stmt).scalars().all())
