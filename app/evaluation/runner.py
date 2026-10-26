from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.dataset import EvalDataset, EvalQuestion, load_dataset
from app.evaluation.metrics import compute_aggregate_metrics
from app.models.user import User
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)
settings = get_settings()


class EvalRunner:
    """Runs an evaluation dataset through the retrieval pipeline and computes metrics.

    This runner evaluates RETRIEVAL quality only (not LLM generation quality).
    Generation evaluation (faithfulness, citation correctness) requires human
    labelling or a stronger judge model and is deferred to a later phase.

    Evaluation flow per question:
    1. Run hybrid retrieval with the eval user's authorization scope.
    2. Compare retrieved chunk IDs against ground-truth relevant_chunk_ids.
    3. Compute per-question Recall@k, Precision@k, RR.
    4. Aggregate across all questions.
    """

    def __init__(
        self,
        db: Session,
        vector_store: FAISSVectorStore,
        keyword_store: KeywordStore,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._keyword_store = keyword_store

    def run(
        self,
        dataset: EvalDataset,
        eval_user: User,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Run the evaluation dataset and return results.

        Args:
            dataset: The EvalDataset to evaluate.
            eval_user: User whose authorization scope is used for retrieval.
            k: Number of results to retrieve per question.

        Returns:
            Dict with 'aggregate' metrics and 'per_question' details.
        """
        k = k or settings.retrieval_top_k
        retriever = HybridRetriever(self._vector_store, self._keyword_store, self._db)
        per_question_results: list[dict[str, Any]] = []

        for q in dataset.questions:
            try:
                result = self._eval_question(q, retriever, eval_user, k)
                per_question_results.append(result)
            except Exception as exc:
                logger.warning("Failed to evaluate question %s: %s", q.id, exc)
                per_question_results.append({
                    "question_id": q.id,
                    "question": q.question,
                    "category": q.category,
                    "error": str(exc),
                    "retrieved_chunk_ids": [],
                    "relevant_chunk_ids": q.relevant_chunk_ids,
                    "expect_abstain": q.expect_abstain,
                    "did_abstain": False,
                })

        aggregate = compute_aggregate_metrics(per_question_results)

        return {
            "dataset": dataset.name,
            "total_questions": len(dataset.questions),
            "aggregate": aggregate,
            "per_question": per_question_results,
        }

    def _eval_question(
        self,
        q: EvalQuestion,
        retriever: HybridRetriever,
        eval_user: User,
        k: int,
    ) -> dict[str, Any]:
        retrieved = retriever.retrieve(
            query=q.question,
            user_id=eval_user.id,
            user_role=eval_user.role,
            top_k=k,
        )

        retrieved_chunk_ids = [r.chunk_id for r in retrieved]
        retrieved_doc_ids = list({r.document_id for r in retrieved})
        top_score = retrieved[0].rrf_score if retrieved else 0.0
        did_abstain = top_score < settings.evidence_threshold

        return {
            "question_id": q.id,
            "question": q.question,
            "category": q.category,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "relevant_chunk_ids": q.relevant_chunk_ids,
            "relevant_document_ids": q.relevant_document_ids,
            "expect_abstain": q.expect_abstain,
            "did_abstain": did_abstain,
            "top_score": round(top_score, 4),
            "retrieval_count": len(retrieved),
        }
