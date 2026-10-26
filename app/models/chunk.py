from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Chunk(Base):
    """A single text chunk derived from a document during ingestion.

    The ``vector_index_id`` is the integer ID assigned by FAISS when the chunk's
    embedding was added to the index.  It is the bridge between the FAISS index
    and the SQLite metadata.  Authorization filtering works by:

    1. Querying SQLite for chunk IDs the user is allowed to see.
    2. Filtering FAISS results to only those FAISS IDs.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ingestion_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ingestion_jobs.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_heading: Mapped[str | None] = mapped_column(String(512), nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Integer ID assigned by FAISS. NULL until embedded and indexed.
    vector_index_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id!r} doc={self.document_id!r} "
            f"idx={self.chunk_index} page={self.page_number}>"
        )
