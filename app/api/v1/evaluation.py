from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.dependencies import get_db, get_keyword_store, get_vector_store
from app.models.user import User
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore
from app.schemas.evaluation import EvalRunListResponse, EvalRunRequest, EvalRunResponse
from app.security.auth_guards import require_admin
from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/eval", tags=["Evaluation"])


def _get_eval_service(
    db: Session = Depends(get_db),
    vector_store: FAISSVectorStore = Depends(get_vector_store),
    keyword_store: KeywordStore = Depends(get_keyword_store),
) -> EvaluationService:
    return EvaluationService(db, vector_store, keyword_store)


@router.post("/run", response_model=EvalRunResponse, status_code=201)
def run_evaluation(
    payload: EvalRunRequest,
    current_user: User = Depends(require_admin),
    eval_service: EvaluationService = Depends(_get_eval_service),
) -> EvalRunResponse:
    """Trigger an evaluation run using the specified benchmark dataset.

    Admin-only. Runs synchronously — may take several minutes for large datasets.
    """
    eval_run = eval_service.run_evaluation(
        dataset_name=payload.dataset_name,
        eval_user=current_user,
    )
    return EvalRunResponse.model_validate(eval_run)


@router.get("/runs", response_model=EvalRunListResponse)
def list_eval_runs(
    current_user: User = Depends(require_admin),
    eval_service: EvaluationService = Depends(_get_eval_service),
) -> EvalRunListResponse:
    """List all past evaluation runs."""
    runs = eval_service.list_runs()
    return EvalRunListResponse(
        items=[EvalRunResponse.model_validate(r) for r in runs],
        total=len(runs),
    )


@router.get("/runs/{run_id}", response_model=EvalRunResponse)
def get_eval_run(
    run_id: str,
    current_user: User = Depends(require_admin),
    eval_service: EvaluationService = Depends(_get_eval_service),
) -> EvalRunResponse:
    """Get results for a specific evaluation run."""
    from app.core.exceptions import NotFoundError
    run = eval_service.get_run(run_id)
    if not run:
        raise NotFoundError("Evaluation run not found.")
    return EvalRunResponse.model_validate(run)
