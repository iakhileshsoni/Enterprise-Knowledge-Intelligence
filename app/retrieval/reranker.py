from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

if TYPE_CHECKING:
    from app.retrieval.hybrid_retriever import RetrievedChunk


class Reranker:
    """Optional cross-encoder reranker.

    Disabled by default (``ENABLE_RERANKER=false``) because the
    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` model requires ~200 MB RAM
    and ~300 ms per batch on CPU.

    Enable with ``ENABLE_RERANKER=true`` in your ``.env`` file.

    The reranker scores (query, passage) pairs and re-orders the candidates.
    It is applied AFTER RRF fusion, so it only sees the top-k fused results.
    """

    def __init__(self) -> None:
        self._model = None
        self._enabled = settings.enable_reranker

        if self._enabled:
            self._load_model()

    def _load_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", settings.reranker_model)
            self._model = CrossEncoder(settings.reranker_model, device="cpu")
            logger.info("Reranker loaded.")
        except Exception as exc:
            logger.warning(
                "Failed to load reranker model %r: %s. Reranking disabled.",
                settings.reranker_model,
                exc,
            )
            self._enabled = False

    def rerank(
        self,
        query: str,
        chunks: list["RetrievedChunk"],
        top_k: int | None = None,
    ) -> list["RetrievedChunk"]:
        """Rerank retrieved chunks using the cross-encoder.

        If the reranker is disabled or fails to load, returns chunks unchanged.

        Args:
            query: The user's question.
            chunks: Retrieved chunks to rerank.
            top_k: Number of chunks to return after reranking.

        Returns:
            Reranked list of chunks (top_k or fewer).
        """
        if not self._enabled or self._model is None or not chunks:
            return chunks[: (top_k or len(chunks))]

        top_k = top_k or settings.reranker_top_k

        try:
            pairs = [(query, c.text) for c in chunks]
            scores = self._model.predict(pairs, show_progress_bar=False)
            ranked = sorted(
                zip(chunks, scores), key=lambda x: x[1], reverse=True
            )
            return [c for c, _ in ranked[:top_k]]
        except Exception as exc:
            logger.warning("Reranker failed: %s. Returning unranked results.", exc)
            return chunks[:top_k]
