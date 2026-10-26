from __future__ import annotations

import numpy as np
import pytest


class TestFAISSVectorStore:
    def test_add_and_search(self, vector_store):
        vectors = np.random.randn(5, 384).astype(np.float32)
        # Normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= norms
        chunk_ids = [f"chunk-{i}" for i in range(5)]
        assigned_ids = vector_store.add(vectors, chunk_ids)
        assert len(assigned_ids) == 5

        query = vectors[0:1]
        results = vector_store.search(query, k=3)
        assert len(results) > 0
        top_id, top_score = results[0]
        assert top_score > 0.9  # Should be near-perfect self-match

    def test_auth_filter_restricts_results(self, vector_store):
        vectors = np.eye(384, dtype=np.float32)[:3]  # 3 orthogonal unit vectors
        chunk_ids = ["a", "b", "c"]
        assigned = vector_store.add(vectors, chunk_ids)

        # Only allow vector_id 0
        results = vector_store.search(vectors[0:1], k=3, allowed_vector_ids=[assigned[0]])
        returned_ids = [r[0] for r in results]
        assert all(rid == assigned[0] for rid in returned_ids)

    def test_is_ready_after_init(self, vector_store):
        assert vector_store.is_ready()

    def test_total_vectors_increments(self, vector_store):
        assert vector_store.total_vectors == 0
        vectors = np.random.randn(3, 384).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        vector_store.add(vectors, ["a", "b", "c"])
        assert vector_store.total_vectors == 3

    def test_save_and_reload(self, vector_store, tmp_path):
        vectors = np.random.randn(2, 384).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        vector_store.add(vectors, ["x", "y"])
        vector_store.save()

        # Load into a new instance from the same paths
        from app.retrieval.vector_store.faiss_store import FAISSVectorStore
        loaded = FAISSVectorStore(
            index_path=vector_store._index_path,
            meta_path=vector_store._meta_path,
        )
        assert loaded.total_vectors == 2

    def test_delete_removes_from_mapping(self, vector_store):
        vectors = np.random.randn(2, 384).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        assigned = vector_store.add(vectors, ["del-me", "keep-me"])
        vector_store.delete([assigned[0]])
        # The mapping should no longer contain the deleted ID
        assert vector_store.get_chunk_id(assigned[0]) is None
        assert vector_store.get_chunk_id(assigned[1]) == "keep-me"


class TestKeywordStore:
    def test_add_and_search(self, keyword_store):
        keyword_store.add("chunk-1", "doc-1", "The annual leave policy states 20 days.")
        results = keyword_store.search("annual leave", ["doc-1"], k=5)
        assert len(results) > 0
        assert results[0][0] == "chunk-1"

    def test_auth_filter_prevents_unauthorized_results(self, keyword_store):
        keyword_store.add("chunk-1", "doc-1", "Secret document content.")
        keyword_store.add("chunk-2", "doc-2", "Public document content.")
        results = keyword_store.search("content", ["doc-2"], k=5)
        result_chunk_ids = [r[0] for r in results]
        assert "chunk-1" not in result_chunk_ids
        assert "chunk-2" in result_chunk_ids

    def test_delete_document_removes_chunks(self, keyword_store):
        keyword_store.add("chunk-1", "doc-1", "Deletable content.")
        keyword_store.delete_document("doc-1")
        results = keyword_store.search("content", ["doc-1"], k=5)
        assert len(results) == 0

    def test_empty_authorized_docs_returns_empty(self, keyword_store):
        keyword_store.add("chunk-1", "doc-1", "Content here.")
        results = keyword_store.search("content", [], k=5)
        assert results == []

    def test_batch_add(self, keyword_store):
        records = [
            ("c1", "d1", "Remote work policy details."),
            ("c2", "d1", "Hybrid attendance requirements."),
        ]
        keyword_store.add_batch(records)
        results = keyword_store.search("remote work", ["d1"], k=5)
        assert len(results) >= 1
