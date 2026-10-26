from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.embedder import EmbeddingService
from app.models.chunk import Chunk
from app.repositories.chunk_repo import ChunkRepository
from app.repositories.document_repo import DocumentRepository
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore

logger = get_logger(__name__)
settings = get_settings()

# RRF constant — 60 is the standard value from the original paper
_RRF_K = 60


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its metadata and fusion score."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_number: Optional[int]
    section_heading: Optional[str]
    rrf_score: float
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    vector_rank: Optional[int] = None
    keyword_rank: Optional[int] = None


def _reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],   # (chunk_id, score)
    keyword_results: list[tuple[str, float]],  # (chunk_id, score)
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Combine two ranked lists using Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)

    This is parameter-free (k is a smoothing constant, not tuned per query)
    and consistently outperforms simple score normalization across retrieval
    benchmarks (Cormack et al., 2009).

    Args:
        vector_results: List of (chunk_id, score) sorted by desc score.
        keyword_results: List of (chunk_id, score) sorted by desc score.
        k: RRF smoothing constant (default 60).

    Returns:
        List of (chunk_id, rrf_score) sorted by descending RRF score.
    """
    rrf_scores: dict[str, float] = {}

    for rank, (chunk_id, _score) in enumerate(vector_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    for rank, (chunk_id, _score) in enumerate(keyword_results, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Combines FAISS vector search with SQLite FTS5 keyword search using RRF.

    Authorization is enforced in two places:
    1. ``document_repo.get_authorized_document_ids()`` produces the allowed list.
    2. ``chunk_repo.get_by_vector_index_ids()`` filters FAISS results to
       only chunks belonging to authorized documents.
    3. ``keyword_store.search()`` filters FTS5 results to authorized doc IDs.

    The LLM never sees content from documents the user is not authorized for.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        keyword_store: KeywordStore,
        db: Session,
    ) -> None:
        self._vector_store = vector_store
        self._keyword_store = keyword_store
        self._db = db
        self._chunk_repo = ChunkRepository(db)
        self._doc_repo = DocumentRepository(db)

    def retrieve(
        self,
        query: str,
        user_id: str,
        user_role: str,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        document_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Execute hybrid retrieval and return fused, authorized results.

        Args:
            query: User's natural-language question.
            user_id: ID of the requesting user.
            user_role: Role of the requesting user ("admin" or "user").
            top_k: Number of final results to return after RRF.
            fetch_k: Candidates to fetch per source before fusion.
            document_ids: Optional explicit document scope (further restricts
                          the user's authorized set — never expands it).

        Returns:
            List of RetrievedChunk objects ordered by descending RRF score.
        """
        top_k = top_k or settings.retrieval_top_k
        fetch_k = fetch_k or settings.retrieval_fetch_k

        # ── Step 1: Authorization scope ────────────────────────────────────────
        authorized_ids = self._doc_repo.get_authorized_document_ids(user_id, user_role)
        if not authorized_ids:
            logger.info("No authorized documents for user %s", user_id)
            return []

        if document_ids:
            # Intersect with user's explicit scope — never expand beyond authorization
            authorized_ids = [d for d in document_ids if d in set(authorized_ids)]
            if not authorized_ids:
                return []

        # ── Step 2: Get authorized FAISS vector IDs ────────────────────────────
        # Build the set of vector IDs that belong to authorized chunks
        authorized_chunks = self._chunk_repo.get_by_vector_index_ids(
            list(range(self._vector_store.total_vectors)),
            authorized_ids,
        )
        if not authorized_chunks:
            return []

        allowed_vector_ids = [
            c.vector_index_id for c in authorized_chunks if c.vector_index_id is not None
        ]

        # ── Step 3: Embed query ────────────────────────────────────────────────
        embedder = EmbeddingService()
        query_vec = embedder.embed_query(query)

        # ── Step 4: Vector search (auth-filtered) ─────────────────────────────
        vector_raw = self._vector_store.search(
            query_vec, k=fetch_k, allowed_vector_ids=allowed_vector_ids
        )
        # Map FAISS vector_id → chunk_id
        vector_results: list[tuple[str, float]] = []
        vid_to_chunk: dict[int, Chunk] = {
            c.vector_index_id: c for c in authorized_chunks if c.vector_index_id is not None
        }
        for vid, score in vector_raw:
            chunk = vid_to_chunk.get(vid)
            if chunk:
                vector_results.append((chunk.id, score))

        # ── Step 5: Keyword search (auth-filtered) ────────────────────────────
        keyword_results = self._keyword_store.search(query, authorized_ids, k=fetch_k)

        # ── Step 6: RRF fusion ─────────────────────────────────────────────────
        fused = _reciprocal_rank_fusion(vector_results, keyword_results)
        fused_top = fused[:top_k]

        # ── Step 7: Enrich with metadata ──────────────────────────────────────
        fused_ids = [cid for cid, _ in fused_top]
        chunk_map: dict[str, Chunk] = {c.id: c for c in self._chunk_repo.get_by_ids(fused_ids)}
        vector_score_map = dict(vector_results)
        keyword_score_map = dict(keyword_results)

        # Fetch document filenames for citation
        from app.models.document import Document
        from sqlalchemy import select
        doc_ids_needed = {chunk_map[cid].document_id for cid in fused_ids if cid in chunk_map}
        docs_stmt = select(Document).where(Document.id.in_(doc_ids_needed))
        doc_map = {
            d.id: d.original_filename
            for d in self._db.execute(docs_stmt).scalars().all()
        }

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, rrf_score) in enumerate(fused_top):
            chunk = chunk_map.get(chunk_id)
            if not chunk:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=doc_map.get(chunk.document_id, "unknown"),
                    text=chunk.text,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    rrf_score=rrf_score,
                    vector_score=vector_score_map.get(chunk_id),
                    keyword_score=keyword_score_map.get(chunk_id),
                    vector_rank=next(
                        (i + 1 for i, (cid, _) in enumerate(vector_results) if cid == chunk_id),
                        None,
                    ),
                    keyword_rank=next(
                        (i + 1 for i, (cid, _) in enumerate(keyword_results) if cid == chunk_id),
                        None,
                    ),
                )
            )

        logger.info(
            "Hybrid retrieval: %d vector + %d keyword → %d fused (user=%s)",
            len(vector_results),
            len(keyword_results),
            len(results),
            user_id,
        )
        return results
