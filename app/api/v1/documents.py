from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, Query, UploadFile, File, Response
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db, get_keyword_store, get_vector_store
from app.models.user import User
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore
from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    ReindexResponse,
)
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


def _get_doc_service(
    db: Session = Depends(get_db),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
    keyword_store: KeywordStore = Depends(get_keyword_store),
) -> DocumentService:
    return DocumentService(db, vector_store, keyword_store)


@router.post("", response_model=DocumentUploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(_get_doc_service),
) -> DocumentUploadResponse:
    """Upload a PDF or DOCX document for ingestion.

    Returns immediately with 202 Accepted.  Ingestion runs as a background task.
    Poll ``GET /documents/{id}/status`` to track progress.
    """
    content = await file.read()
    doc = doc_service.handle_upload(
        owner_id=current_user.id,
        filename=file.filename or "unknown",
        content=content,
        declared_mime=file.content_type,
    )
    # Schedule ingestion as a background task
    background_tasks.add_task(doc_service.run_ingestion, doc.id)

    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.original_filename,
        status=doc.status,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(_get_doc_service),
) -> DocumentListResponse:
    """List all documents the current user has access to."""
    docs, total = doc_service.list_documents(
        current_user.id, current_user.role, page, page_size
    )
    return DocumentListResponse(
        items=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(_get_doc_service),
) -> DocumentResponse:
    """Get metadata for a specific document."""
    doc = doc_service.get_document(document_id, current_user.id, current_user.role)
    return DocumentResponse.model_validate(doc)


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(_get_doc_service),
    db: Session = Depends(get_db),
) -> DocumentStatusResponse:
    """Get ingestion status for a document."""
    from app.repositories.document_repo import DocumentRepository
    doc = doc_service.get_document(document_id, current_user.id, current_user.role)
    doc_repo = DocumentRepository(db)
    latest_job = doc_repo.get_latest_job(document_id)
    return DocumentStatusResponse(
        document_id=doc.id,
        filename=doc.original_filename,
        status=doc.status,
        version=doc.version,
        chunk_count=latest_job.chunk_count if latest_job else None,
        error_message=doc.error_message if doc.status == "failed" else None,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.delete("/{document_id}", status_code=204, response_class=Response)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(_get_doc_service),
) -> Response:
    """Soft-delete a document and remove its index entries."""
    doc_service.delete_document(document_id, current_user.id, current_user.role)
    return Response(status_code=204)


@router.post("/{document_id}/reindex", response_model=ReindexResponse)
def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    doc_service: DocumentService = Depends(_get_doc_service),
) -> ReindexResponse:
    """Trigger re-ingestion (re-chunking + re-embedding) of a document.

    Useful when the chunking strategy or embedding model changes.
    """
    job_id = doc_service.trigger_reindex(document_id, current_user.id, current_user.role)
    background_tasks.add_task(doc_service.run_ingestion, document_id)
    return ReindexResponse(document_id=document_id, job_id=job_id)
