from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.dataset import load_dataset
from app.evaluation.runner import EvalRunner
from app.models.evaluation import EvalRun
from app.models.user import User
from app.repositories.document_repo import DocumentRepository
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)
settings = get_settings()


class EvaluationService:
    """Service layer for triggering and storing evaluation runs."""

    def __init__(
        self,
        db: Session,
        vector_store: FAISSVectorStore,
        keyword_store: KeywordStore,
    ) -> None:
        self._db = db
        self._vector_store = vector_store
        self._keyword_store = keyword_store

    def run_evaluation(
        self,
        dataset_name: str,
        eval_user: User,
    ) -> EvalRun:
        """Load dataset, run evaluation, and persist results.

        Args:
            dataset_name: Name of the .jsonl file in data/evaluation/ (without extension).
            eval_user: User whose authorization scope is used.

        Returns:
            EvalRun ORM object with persisted results.
        """
        dataset_path = settings.sample_documents_dir.parent / "evaluation" / f"{dataset_name}.jsonl"
        dataset = load_dataset(dataset_path)

        runner = EvalRunner(self._db, self._vector_store, self._keyword_store)
        results = runner.run(dataset, eval_user)

        config_snapshot = {
            "embedding_model": settings.embedding_model,
            "retrieval_top_k": settings.retrieval_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "enable_reranker": settings.enable_reranker,
            "evidence_threshold": settings.evidence_threshold,
        }

        eval_run = EvalRun(
            dataset_name=dataset_name,
            config_snapshot=config_snapshot,
            results=results,
        )
        self._db.add(eval_run)
        self._db.commit()
        self._db.refresh(eval_run)

        logger.info(
            "Eval run completed: dataset=%s mrr=%.4f recall@10=%.4f",
            dataset_name,
            results.get("aggregate", {}).get("mrr", 0),
            results.get("aggregate", {}).get("recall_at_10", 0),
        )
        return eval_run

    def get_run(self, run_id: str) -> EvalRun | None:
        return self._db.get(EvalRun, run_id)

    def list_runs(self) -> list[EvalRun]:
        from sqlalchemy import select
        return list(
            self._db.execute(
                select(EvalRun).order_by(EvalRun.run_at.desc())
            ).scalars().all()
        )
