from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    """Returned immediately after a file is accepted for ingestion."""
    document_id: str
    filename: str
    status: str
    message: str = "Document accepted for processing."


class DocumentStatusResponse(BaseModel):
    """Detailed ingestion status for a document."""
    document_id: str
    filename: str
    status: str
    version: int
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class DocumentResponse(BaseModel):
    """Full document metadata returned by GET /documents/{id}."""
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    original_filename: str
    mime_type: str
    file_size_bytes: int
    status: str
    version: int
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class ReindexResponse(BaseModel):
    """Returned when a re-index job is triggered."""
    document_id: str
    job_id: str
    message: str = "Re-indexing job has been queued."
