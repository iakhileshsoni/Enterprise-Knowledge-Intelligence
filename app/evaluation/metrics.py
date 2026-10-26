from __future__ import annotations

from typing import Any


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Recall@k: fraction of relevant items found in the top-k retrieved.

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Args:
        retrieved_ids: Ordered list of retrieved chunk/doc IDs (best first).
        relevant_ids: Ground-truth relevant IDs.
        k: Cut-off rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = top_k & set(relevant_ids)
    return len(hits) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Precision@k: fraction of top-k retrieved items that are relevant.

    Precision@k = |retrieved[:k] ∩ relevant| / k

    Args:
        retrieved_ids: Ordered list of retrieved IDs.
        relevant_ids: Ground-truth relevant IDs.
        k: Cut-off rank.

    Returns:
        Float in [0, 1]. Returns 0.0 if k == 0.
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in set(relevant_ids))
    return hits / k


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Reciprocal Rank: 1/rank of the first relevant result.

    Args:
        retrieved_ids: Ordered list of retrieved IDs.
        relevant_ids: Ground-truth relevant IDs.

    Returns:
        Float in (0, 1]. Returns 0.0 if no relevant item is found.
    """
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(per_query_rr: list[float]) -> float:
    """Mean Reciprocal Rank across multiple queries.

    Args:
        per_query_rr: List of per-query reciprocal rank values.

    Returns:
        Float in [0, 1]. Returns 0.0 if the list is empty.
    """
    if not per_query_rr:
        return 0.0
    return sum(per_query_rr) / len(per_query_rr)


def compute_aggregate_metrics(
    per_question_results: list[dict[str, Any]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute aggregate metrics across all evaluation questions.

    Args:
        per_question_results: Each entry has keys:
            - retrieved_chunk_ids: list[str]
            - retrieved_doc_ids: list[str]
            - relevant_chunk_ids: list[str]
            - relevant_document_ids: list[str]
            - expect_abstain: bool
            - did_abstain: bool
        k_values: List of k values for Recall@k and Precision@k.

    Returns:
        Dict of aggregate metric name → float value.
    """
    k_values = k_values or [1, 3, 5, 10]

    rr_values: list[float] = []
    recall_k: dict[int, list[float]] = {k: [] for k in k_values}
    precision_k: dict[int, list[float]] = {k: [] for k in k_values}
    abstention_correct = 0
    abstention_total_expected = 0
    false_abstentions = 0
    false_answers = 0

    for result in per_question_results:
        retrieved_chunks = result.get("retrieved_chunk_ids", [])
        relevant_chunks = result.get("relevant_chunk_ids", [])
        expect_abstain = result.get("expect_abstain", False)
        did_abstain = result.get("did_abstain", False)

        if expect_abstain:
            abstention_total_expected += 1
            if did_abstain:
                abstention_correct += 1
        else:
            if did_abstain:
                false_abstentions += 1

        if relevant_chunks:
            rr_values.append(reciprocal_rank(retrieved_chunks, relevant_chunks))
            for k in k_values:
                recall_k[k].append(recall_at_k(retrieved_chunks, relevant_chunks, k))
                precision_k[k].append(precision_at_k(retrieved_chunks, relevant_chunks, k))

    metrics: dict[str, float] = {
        "mrr": mean_reciprocal_rank(rr_values),
        "abstention_accuracy": (
            abstention_correct / abstention_total_expected
            if abstention_total_expected > 0
            else 0.0
        ),
        "false_abstention_rate": (
            false_abstentions / max(1, len(per_question_results) - abstention_total_expected)
        ),
    }

    for k in k_values:
        if recall_k[k]:
            metrics[f"recall_at_{k}"] = sum(recall_k[k]) / len(recall_k[k])
            metrics[f"precision_at_{k}"] = sum(precision_k[k]) / len(precision_k[k])

    return metrics
