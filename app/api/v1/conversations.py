from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request, Response
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user, get_db, get_keyword_store, get_vector_store
from app.models.user import User
from app.repositories.conversation_repo import ConversationRepository
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore
from app.schemas.qa import (
    AskRequest,
    AskResponse,
    ConversationAskRequest,
    ConversationCreate,
    ConversationDetailResponse,
    ConversationResponse,
    MessageResponse,
)
from app.services.qa_service import QAService

router = APIRouter(tags=["Q&A / Conversations"])


def _get_qa_service(
    db: Session = Depends(get_db),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
    keyword_store: KeywordStore = Depends(get_keyword_store),
) -> QAService:
    return QAService(db, vector_store, keyword_store)


# ── Stateless Ask ─────────────────────────────────────────────────────────────

@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    qa_service: QAService = Depends(_get_qa_service),
) -> AskResponse:
    """Ask a single question without maintaining conversation history.

    The response includes citations to source document chunks and an explicit
    ``abstain`` flag when insufficient evidence is available.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return await qa_service.answer(
        question=payload.question,
        user=current_user,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        request_id=request_id,
    )


# ── Conversations ─────────────────────────────────────────────────────────────

@router.post("/conversations", response_model=ConversationResponse, status_code=201)
def create_conversation(
    payload: ConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationResponse:
    """Create a new conversation session."""
    repo = ConversationRepository(db)
    conv = repo.create(user_id=current_user.id, title=payload.title)
    return ConversationResponse.model_validate(conv)


@router.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ConversationResponse]:
    """List the current user's conversations."""
    repo = ConversationRepository(db)
    convs, _ = repo.list_for_user(current_user.id, page, page_size)
    return [ConversationResponse.model_validate(c) for c in convs]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationDetailResponse:
    """Get a conversation with its full message history."""
    from app.core.exceptions import ConversationNotFoundError, PermissionDeniedError
    repo = ConversationRepository(db)
    conv = repo.get_by_id(conversation_id)
    if not conv:
        raise ConversationNotFoundError()
    if conv.user_id != current_user.id and current_user.role != "admin":
        raise ConversationNotFoundError()  # Don't reveal existence

    messages = repo.get_recent_messages(conversation_id, limit=100)
    return ConversationDetailResponse(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[MessageResponse.model_validate(m) for m in messages],
    )


@router.post("/conversations/{conversation_id}/messages", response_model=AskResponse)
async def ask_in_conversation(
    conversation_id: str,
    payload: ConversationAskRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    qa_service: QAService = Depends(_get_qa_service),
) -> AskResponse:
    """Ask a question within an existing conversation (stateful, uses history)."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return await qa_service.answer(
        question=payload.question,
        user=current_user,
        conversation_id=conversation_id,
        document_ids=payload.document_ids,
        top_k=payload.top_k,
        request_id=request_id,
    )


@router.delete("/conversations/{conversation_id}", status_code=204, response_class=Response)
def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Response:
    """Delete a conversation and all its messages."""
    from app.core.exceptions import ConversationNotFoundError, PermissionDeniedError
    repo = ConversationRepository(db)
    conv = repo.get_by_id(conversation_id)
    if not conv:
        raise ConversationNotFoundError()
    if conv.user_id != current_user.id and current_user.role != "admin":
        raise PermissionDeniedError()
    repo.delete(conversation_id)
    return Response(status_code=204)
