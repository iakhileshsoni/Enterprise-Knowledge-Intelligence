from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Thread-safe singleton lock
_model_lock = threading.Lock()
_model_instance: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the singleton SentenceTransformer model, loading it on first call.

    The model is loaded once and reused across all requests.  Loading takes
    ~2-4 seconds on first call; subsequent calls are instantaneous.

    Thread-safety: Protected by ``_model_lock`` so concurrent startup
    requests don't double-load the model.
    """
    global _model_instance
    if _model_instance is None:
        with _model_lock:
            if _model_instance is None:
                logger.info(
                    "Loading embedding model: %s", settings.embedding_model
                )
                _model_instance = SentenceTransformer(
                    settings.embedding_model,
                    device="cpu",
                )
                logger.info("Embedding model loaded successfully.")
    return _model_instance


class EmbeddingService:
    """Wraps SentenceTransformer with batching, normalization, and logging.

    All embeddings are L2-normalized so that FAISS IndexFlatIP computes
    cosine similarity correctly (dot product on unit vectors = cosine similarity).
    """

    def __init__(self) -> None:
        self._batch_size = settings.embedding_batch_size
        self._model_name = settings.embedding_model
        self._dim = settings.embedding_dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return a float32 numpy array of shape (N, D).

        Args:
            texts: List of text strings to embed.  May be empty.

        Returns:
            numpy array of shape (len(texts), embedding_dimension), dtype float32.
            L2-normalized.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        model = _get_model()
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,   # L2 normalize for cosine via dot product
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, D)."""
        return self.embed_texts([query])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim
