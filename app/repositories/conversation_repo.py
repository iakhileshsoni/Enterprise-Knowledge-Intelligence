from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.conversation import Conversation, Message


class ConversationRepository:
    """Data-access layer for Conversation and Message records."""

    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Conversation ──────────────────────────────────────────────────────────

    def create(self, user_id: str, title: Optional[str] = None) -> Conversation:
        conv = Conversation(user_id=user_id, title=title)
        self._db.add(conv)
        self._db.commit()
        self._db.refresh(conv)
        return conv

    def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        return self._db.get(Conversation, conversation_id)

    def list_for_user(
        self, user_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[Conversation], int]:
        base = select(Conversation).where(Conversation.user_id == user_id)
        total = self._db.execute(
            select(func.count()).select_from(base.subquery())
        ).scalar_one()
        convs = (
            self._db.execute(
                base.order_by(Conversation.updated_at.desc())
                .offset((page - 1) * page_size)
                .limit(page_size)
            )
            .scalars()
            .all()
        )
        return list(convs), total

    def user_owns(self, conversation_id: str, user_id: str) -> bool:
        conv = self.get_by_id(conversation_id)
        return conv is not None and conv.user_id == user_id

    def delete(self, conversation_id: str) -> None:
        conv = self.get_by_id(conversation_id)
        if conv:
            self._db.delete(conv)
            self._db.commit()

    # ── Message ───────────────────────────────────────────────────────────────

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[list] = None,
        retrieval_metadata: Optional[dict] = None,
        latency_ms: Optional[float] = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations,
            retrieval_metadata=retrieval_metadata,
            latency_ms=latency_ms,
        )
        self._db.add(msg)
        # Touch conversation.updated_at
        conv = self.get_by_id(conversation_id)
        if conv:
            from datetime import datetime, timezone
            conv.updated_at = datetime.now(timezone.utc)
        self._db.commit()
        self._db.refresh(msg)
        return msg

    def get_recent_messages(
        self, conversation_id: str, limit: int = 12
    ) -> list[Message]:
        """Return the most recent messages in chronological order."""
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(self._db.execute(stmt).scalars().all())
        return list(reversed(messages))  # chronological order

    def message_count(self, conversation_id: str) -> int:
        stmt = select(func.count()).where(Message.conversation_id == conversation_id)
        return self._db.execute(stmt).scalar_one()
