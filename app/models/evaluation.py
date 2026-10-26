from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, String, Text
from sqlalchemy import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class EvalRun(Base):
    """Stores the results of a benchmark evaluation run.

    ``config_snapshot`` captures the retrieval/generation settings active at
    the time of the run so results are reproducible.
    ``results`` stores the per-question details and aggregate metrics.
    """

    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    dataset_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # Snapshot of Settings relevant to this run
    config_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # {aggregate: {recall_at_k, mrr, ...}, per_question: [...]}
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    run_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    def __repr__(self) -> str:
        return f"<EvalRun id={self.id!r} dataset={self.dataset_name!r}>"
