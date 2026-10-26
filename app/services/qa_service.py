from __future__ import annotations

import time
import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import (
    ConversationNotFoundError,
    InsufficientEvidenceError,
    PermissionDeniedError,
)
from app.core.logging import get_logger
from app.generation.citation_validator import CitationValidator
from app.generation.ollama_client import OllamaClient
from app.generation.prompt_builder import PromptBuilder
from app.models.user import User
from app.repositories.conversation_repo import ConversationRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store.faiss_store import FAISSVectorStore
from app.schemas.qa import AskResponse, Citation

logger = get_logger(__name__)
settings = get_settings()


class QAService:
    """Orchestrates the full question-answering pipeline.

    Pipeline:
    1. Authorization scope (authorized_doc_ids)
    2. Hybrid retrieval (vector + keyword, RRF fused)
    3. Optional reranking
    4. Evidence threshold check → abstain if below threshold
    5. Context construction (token-budgeted)
    6. LLM generation
    7. Citation validation
    8. Conversation persistence (if conversation_id provided)
    """

    def __init__(
        self,
        db: Session,
        vector_store: FAISSVectorStore,
        keyword_store: KeywordStore,
    ) -> None:
        self._db = db
        self._retriever = HybridRetriever(vector_store, keyword_store, db)
        self._reranker = Reranker()
        self._prompt_builder = PromptBuilder()
        self._citation_validator = CitationValidator()
        self._conversation_repo = ConversationRepository(db)
        self._ollama = OllamaClient()

    async def answer(
        self,
        question: str,
        user: User,
        conversation_id: Optional[str] = None,
        document_ids: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        request_id: Optional[str] = None,
    ) -> AskResponse:
        """Answer a user question using the RAG pipeline.

        Args:
            question: The user's question.
            user: The authenticated User ORM object.
            conversation_id: Optional — attach turn to this conversation.
            document_ids: Optional explicit document scope.
            top_k: Override retrieval_top_k for this request.
            request_id: Correlation ID for logging.

        Returns:
            AskResponse with answer, citations, and abstention flag.
        """
        request_id = request_id or str(uuid.uuid4())
        start_time = time.perf_counter()

        # ── Validate conversation ownership ────────────────────────────────────
        conversation_history: list[dict[str, str]] = []
        if conversation_id:
            conv = self._conversation_repo.get_by_id(conversation_id)
            if not conv:
                raise ConversationNotFoundError()
            if conv.user_id != user.id:
                raise PermissionDeniedError("You do not own this conversation.")
            recent = self._conversation_repo.get_recent_messages(
                conversation_id, limit=settings.max_conversation_turns * 2
            )
            conversation_history = [
                {"role": m.role, "content": m.content}
                for m in recent
                if m.role in ("user", "assistant")
            ]

        # ── Hybrid retrieval ───────────────────────────────────────────────────
        retrieved = self._retriever.retrieve(
            query=question,
            user_id=user.id,
            user_role=user.role,
            top_k=top_k or settings.retrieval_top_k,
            document_ids=document_ids,
        )

        # ── Optional reranking ─────────────────────────────────────────────────
        if settings.enable_reranker and retrieved:
            retrieved = self._reranker.rerank(question, retrieved)

        # ── Evidence threshold check ───────────────────────────────────────────
        retrieval_count = len(retrieved)
        top_vector_score = max(
            (r.vector_score for r in retrieved if r.vector_score is not None),
            default=0.0,
        )
        has_keyword_match = any(
            r.keyword_score is not None and r.keyword_score > 0 for r in retrieved
        )

        if not retrieved or (top_vector_score < settings.evidence_threshold and not has_keyword_match):
            logger.info(
                "Abstaining: no sufficient evidence (top_vector_score=%.4f, threshold=%.4f, has_keyword=%s)",
                top_vector_score,
                settings.evidence_threshold,
                has_keyword_match,
            )
            elapsed = (time.perf_counter() - start_time) * 1000
            response = AskResponse(
                request_id=request_id,
                question=question,
                answer=None,
                abstain=True,
                abstain_reason="No relevant information found in the available documents.",
                citations=[],
                retrieval_count=retrieval_count,
                latency_ms=round(elapsed, 2),
            )
            await self._persist_turn(
                conversation_id, user.id, question, response, conversation_history
            )
            return response

        # ── Context construction ───────────────────────────────────────────────
        context_text, source_map = self._prompt_builder.build_context_block(retrieved)
        messages = self._prompt_builder.build_prompt(
            question=question,
            context_text=context_text,
            conversation_history=conversation_history if conversation_id else None,
        )

        # ── LLM generation ─────────────────────────────────────────────────────
        llm_output = await self._ollama.chat(messages)

        # ── Citation validation ────────────────────────────────────────────────
        answer_text, citations = self._citation_validator.validate_and_extract(
            llm_output=llm_output,
            source_map=source_map,
            retrieved_chunks=retrieved,
        )

        # ── Abstention from LLM output ─────────────────────────────────────────
        abstain = answer_text is None
        elapsed = (time.perf_counter() - start_time) * 1000

        response = AskResponse(
            request_id=request_id,
            question=question,
            answer=answer_text,
            abstain=abstain,
            abstain_reason="Insufficient evidence in the document corpus." if abstain else None,
            citations=citations,
            retrieval_count=retrieval_count,
            latency_ms=round(elapsed, 2),
        )

        # ── Persist conversation turn ──────────────────────────────────────────
        await self._persist_turn(
            conversation_id, user.id, question, response, conversation_history
        )

        logger.info(
            "QA complete: request_id=%s retrieved=%d abstain=%s latency=%.1f ms",
            request_id,
            retrieval_count,
            abstain,
            elapsed,
        )
        return response

    async def _persist_turn(
        self,
        conversation_id: Optional[str],
        user_id: str,
        question: str,
        response: AskResponse,
        history: list[dict],
    ) -> None:
        """Save the Q&A turn to conversation history if conversation_id is set."""
        if not conversation_id:
            return
        try:
            self._conversation_repo.add_message(
                conversation_id=conversation_id,
                role="user",
                content=question,
            )
            self._conversation_repo.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response.answer or response.abstain_reason or "",
                citations=[c.model_dump() for c in response.citations],
                retrieval_metadata={"retrieval_count": response.retrieval_count},
                latency_ms=response.latency_ms,
            )
        except Exception as exc:
            logger.warning("Failed to persist conversation turn: %s", exc)
