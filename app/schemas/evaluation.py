from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class EvalRunRequest(BaseModel):
    dataset_name: str = "benchmark_v1"


class EvalRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_name: str
    results: Optional[dict[str, Any]] = None
    run_at: datetime


class EvalRunListResponse(BaseModel):
    items: list[EvalRunResponse]
    total: int
