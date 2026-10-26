from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    """A single evaluation question with expected answer metadata."""

    id: str
    question: str
    expected_answer: str | None = None
    # UUIDs of the documents that contain the answer (for retrieval evaluation)
    relevant_document_ids: list[str] = Field(default_factory=list)
    # Expected chunk IDs that should appear in retrieval results
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    # Category for analysis: factual|multi_chunk|multi_doc|paraphrase|missing|ambiguous|security|oos
    category: str = "factual"
    # Whether the question should result in abstention
    expect_abstain: bool = False


class EvalDataset(BaseModel):
    """Collection of evaluation questions."""

    name: str
    version: str = "1.0"
    questions: list[EvalQuestion]


def load_dataset(path: Path) -> EvalDataset:
    """Load an evaluation dataset from a JSONL file.

    Each line in the JSONL file is a JSON object representing an EvalQuestion,
    plus optional top-level metadata on the first line.

    Simple format: one JSON object per line, each is an EvalQuestion.

    Args:
        path: Path to the .jsonl file.

    Returns:
        EvalDataset loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If any line fails validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    questions: list[EvalQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                questions.append(EvalQuestion(**data))
            except Exception as exc:
                raise ValueError(f"Invalid eval question at line {lineno}: {exc}") from exc

    return EvalDataset(
        name=path.stem,
        questions=questions,
    )
