from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.core.config import get_settings
from app.core.exceptions import IndexNotReadyError
from app.core.logging import get_logger
from app.retrieval.vector_store.base import VectorStore

logger = get_logger(__name__)
settings = get_settings()


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store using IndexFlatIP (exact cosine similarity).

    Design decisions:
    - ``IndexFlatIP``: Exact search; no approximation error.  Fast enough for
      up to ~500k vectors on CPU (sub-second for our expected scale).
    - Vectors MUST be L2-normalized before insertion (done by EmbeddingService).
      Dot product on unit vectors equals cosine similarity.
    - ``_id_to_chunk``: Maps FAISS sequential integers → chunk UUIDs.
      Persisted alongside the index as JSON so the mapping survives restarts.
    - Thread safety: All mutations are serialised via ``_lock``.

    Persistence:
    - ``faiss_index_path``: The FAISS binary index file.
    - ``faiss_meta_path``:  JSON dict mapping str(vector_id) → chunk_uuid.

    Authorization:
    - ``search()`` accepts ``allowed_vector_ids`` to restrict results.
      This is a simple post-filter; the full index is always searched.
      For our expected scale (<100k vectors) this is acceptably fast.
      At larger scale, FAISS IDMap or a pre-filter strategy is preferable.
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self._index_path = index_path or settings.faiss_index_path
        self._meta_path = meta_path or settings.faiss_meta_path
        self._dim = dimension or settings.embedding_dimension
        self._lock = threading.Lock()
        self._index: Optional[faiss.IndexFlatIP] = None
        # Maps FAISS sequential integer ID → chunk UUID string
        self._id_to_chunk: dict[int, str] = {}
        self._next_id: int = 0

        # Auto-load if index files exist
        if self._index_path.exists() and self._meta_path.exists():
            try:
                self.load()
            except Exception as exc:
                logger.warning("Failed to load existing FAISS index: %s — starting fresh.", exc)
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        self._index = faiss.IndexFlatIP(self._dim)
        self._id_to_chunk = {}
        self._next_id = 0
        logger.info("Initialised empty FAISS IndexFlatIP (dim=%d)", self._dim)

    # ── VectorStore interface ─────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, chunk_ids: list[str]) -> list[int]:
        if vectors.shape[0] == 0:
            return []

        assert vectors.shape[1] == self._dim, (
            f"Expected dimension {self._dim}, got {vectors.shape[1]}"
        )

        vectors = vectors.astype(np.float32)

        with self._lock:
            start_id = self._next_id
            self._index.add(vectors)  # type: ignore[union-attr]
            assigned_ids = list(range(start_id, start_id + len(chunk_ids)))
            for vid, cid in zip(assigned_ids, chunk_ids):
                self._id_to_chunk[vid] = cid
            self._next_id += len(chunk_ids)

        logger.debug("FAISS: added %d vectors (total=%d)", len(chunk_ids), self._next_id)
        return assigned_ids

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        allowed_vector_ids: list[int] | None = None,
    ) -> list[tuple[int, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        query = query_vector.reshape(1, -1).astype(np.float32)
        # Fetch more candidates than k when we need to filter
        fetch_k = min(k * 5, self._index.ntotal) if allowed_vector_ids is not None else k

        with self._lock:
            scores, indices = self._index.search(query, fetch_k)  # type: ignore[union-attr]

        results: list[tuple[int, float]] = []
        allowed_set = set(allowed_vector_ids) if allowed_vector_ids is not None else None

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            if allowed_set is not None and idx not in allowed_set:
                continue
            results.append((int(idx), float(score)))
            if len(results) >= k:
                break

        return results

    def delete(self, vector_ids: list[int]) -> None:
        """FAISS IndexFlatIP does not support ID-based removal.

        On deletion we mark the mapping as removed.  The vector remains in the
        FAISS index but will never be returned because it won't appear in the
        allowed_vector_ids whitelist (the chunk record in SQLite is deleted).

        For production use with frequent deletions, switch to IndexIDMap2 or
        rebuild the index periodically.
        """
        with self._lock:
            for vid in vector_ids:
                self._id_to_chunk.pop(vid, None)

    def save(self) -> None:
        with self._lock:
            if self._index is None:
                return
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_path))
            meta = {
                "id_to_chunk": {str(k): v for k, v in self._id_to_chunk.items()},
                "next_id": self._next_id,
                "dimension": self._dim,
            }
            self._meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("FAISS index saved → %s", self._index_path)

    def load(self) -> None:
        with self._lock:
            self._index = faiss.read_index(str(self._index_path))
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._id_to_chunk = {int(k): v for k, v in meta["id_to_chunk"].items()}
            self._next_id = meta.get("next_id", self._index.ntotal)
            self._dim = meta.get("dimension", self._dim)
        logger.info(
            "FAISS index loaded: %d vectors from %s", self._index.ntotal, self._index_path
        )

    def is_ready(self) -> bool:
        return self._index is not None

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def get_chunk_id(self, vector_id: int) -> Optional[str]:
        return self._id_to_chunk.get(vector_id)

    def get_all_vector_ids_for_chunks(self, chunk_ids: set[str]) -> list[int]:
        """Reverse lookup: chunk_uuid → vector_id (for authorization pre-filter)."""
        return [vid for vid, cid in self._id_to_chunk.items() if cid in chunk_ids]
