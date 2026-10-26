from __future__ import annotations

import pytest

from app.retrieval.hybrid_retriever import _reciprocal_rank_fusion


class TestRRFFusion:
    def test_empty_inputs_return_empty(self):
        result = _reciprocal_rank_fusion([], [])
        assert result == []

    def test_single_source_returns_ranked_list(self):
        vector_results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        result = _reciprocal_rank_fusion(vector_results, [])
        ids = [r[0] for r in result]
        # All IDs present
        assert set(ids) == {"a", "b", "c"}

    def test_rrf_score_higher_for_consistently_top_ranked(self):
        # "a" appears at rank 1 in both lists — should have highest RRF
        vector_results = [("a", 0.9), ("b", 0.7)]
        keyword_results = [("a", 5.0), ("c", 4.0)]
        result = _reciprocal_rank_fusion(vector_results, keyword_results)
        top_id = result[0][0]
        assert top_id == "a"

    def test_item_only_in_one_list_still_included(self):
        vector_results = [("a", 0.9)]
        keyword_results = [("b", 5.0)]
        result = _reciprocal_rank_fusion(vector_results, keyword_results)
        ids = [r[0] for r in result]
        assert "a" in ids
        assert "b" in ids

    def test_rrf_scores_are_positive(self):
        vector_results = [("a", 0.9), ("b", 0.8)]
        keyword_results = [("b", 5.0), ("c", 4.0)]
        result = _reciprocal_rank_fusion(vector_results, keyword_results)
        for _id, score in result:
            assert score > 0

    def test_result_sorted_descending(self):
        vector_results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        keyword_results = [("c", 5.0), ("b", 4.0), ("a", 3.0)]
        result = _reciprocal_rank_fusion(vector_results, keyword_results)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
