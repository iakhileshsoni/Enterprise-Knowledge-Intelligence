from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Implement this interface to swap FAISS for Chroma, pgvector, or any other
    backend.  All methods must be thread-safe.

    Terminology:
    - ``vector_id``: The integer ID assigned by the store to a vector.
      This is stored in ``Chunk.vector_index_id`` in SQLite.
    - ``chunk_id``: The UUID string identifying the Chunk in SQLite.
    """

    @abstractmethod
    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> list[int]:
        """Add vectors to the store.

        Args:
            vectors: Float32 numpy array of shape (N, D).
            chunk_ids: List of N chunk UUID strings (for reconciliation).

        Returns:
            List of N integer vector IDs assigned by the store.
        """

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        allowed_vector_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Search for the k nearest vectors.

        Args:
            query_vector: Float32 numpy array of shape (1, D) or (D,).
            k: Number of results to return.
            allowed_vector_ids: If provided, restrict results to these IDs.

        Returns:
            List of (vector_id, score) tuples, sorted by descending score.
        """

    @abstractmethod
    def delete(self, vector_ids: list[int]) -> None:
        """Remove vectors from the store by their integer IDs."""

    @abstractmethod
    def save(self) -> None:
        """Persist the index to disk."""

    @abstractmethod
    def load(self) -> None:
        """Load the index from disk."""

    @abstractmethod
    def is_ready(self) -> bool:
        """Return True if the store has been initialised and is ready to serve queries."""

    @property
    @abstractmethod
    def total_vectors(self) -> int:
        """Return the number of vectors currently stored."""
