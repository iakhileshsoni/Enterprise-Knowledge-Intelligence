from __future__ import annotations

import numpy as np
import pytest

from app.ingestion.embedder import EmbeddingService
from app.core.config import get_settings

settings = get_settings()


class TestEmbeddingService:
    """Unit tests for the embedding service.

    These tests load the real sentence-transformer model.
    Skip them if you want to avoid model download during CI.
    Mark with @pytest.mark.slow if needed.
    """

    def test_embed_returns_correct_shape(self):
        service = EmbeddingService()
        texts = ["hello world", "this is a test"]
        result = service.embed_texts(texts)
        assert result.shape == (2, settings.embedding_dimension)

    def test_embed_returns_float32(self):
        service = EmbeddingService()
        result = service.embed_texts(["test text"])
        assert result.dtype == np.float32

    def test_embed_single_query_shape(self):
        service = EmbeddingService()
        result = service.embed_query("what is the leave policy?")
        assert result.shape == (1, settings.embedding_dimension)

    def test_empty_input_returns_empty_array(self):
        service = EmbeddingService()
        result = service.embed_texts([])
        assert result.shape == (0, settings.embedding_dimension)

    def test_embeddings_are_unit_normalized(self):
        """Embeddings must be L2-normalized for FAISS cosine similarity to work."""
        service = EmbeddingService()
        result = service.embed_texts(["normalize this text"])
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_similar_texts_have_higher_similarity(self):
        service = EmbeddingService()
        e1 = service.embed_texts(["What is the annual leave entitlement?"])
        e2 = service.embed_texts(["How many days of leave do employees get per year?"])
        e3 = service.embed_texts(["What is the password reset policy?"])

        sim_12 = float(np.dot(e1, e2.T)[0, 0])
        sim_13 = float(np.dot(e1, e3.T)[0, 0])
        # Similar questions should have higher cosine similarity
        assert sim_12 > sim_13

    def test_model_name_property(self):
        service = EmbeddingService()
        assert "MiniLM" in service.model_name or "minilm" in service.model_name.lower()

    def test_dimension_property(self):
        service = EmbeddingService()
        assert service.dimension == settings.embedding_dimension
