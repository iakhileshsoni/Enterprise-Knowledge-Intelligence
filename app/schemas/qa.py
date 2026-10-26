from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Citation ──────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single citation derived from retrieval metadata — never from LLM output."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    relevance_score: float
    excerpt: str = Field(..., max_length=300, description="Short excerpt from the source chunk.")


# ── Ask (stateless single-turn) ───────────────────────────────────────────────

class AskRequest(BaseModel):
    """Stateless single-turn question."""
    question: str = Field(..., min_length=3, max_length=2000)
    # Optional document scope filter — if empty, searches all authorized docs
    document_ids: Optional[list[str]] = None
    # Override retrieval k for this request
    top_k: Optional[int] = Field(None, ge=1, le=50)


class AskResponse(BaseModel):
    """Answer with citations and abstention flag."""
    request_id: str
    question: str
    answer: Optional[str]
    abstain: bool = False
    abstain_reason: Optional[str] = None
    citations: list[Citation] = []
    retrieval_count: int
    latency_ms: float


# ── Conversation messages ─────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    citations: Optional[list[Any]] = None
    latency_ms: Optional[float] = None
    created_at: datetime


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(None, max_length=256)


class ConversationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetailResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    messages: list[MessageResponse]


class ConversationAskRequest(BaseModel):
    """Turn within a conversation — includes conversation_id from the path."""
    question: str = Field(..., min_length=3, max_length=2000)
    document_ids: Optional[list[str]] = None
    top_k: Optional[int] = Field(None, ge=1, le=50)
