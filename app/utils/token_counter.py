from __future__ import annotations

import re

# Approximate characters-to-tokens ratio for English text.
# GPT-style models: ~4 chars per token. MiniLM context is 512 tokens.
# We use a conservative 3.5 chars/token to avoid underestimating.
_CHARS_PER_TOKEN: float = 3.5


def count_tokens_approx(text: str) -> int:
    """Approximate token count using character-based heuristic.

    This is intentionally approximate — it avoids the overhead of loading a
    full tokenizer just to enforce context budgets.  Use it for soft limits
    where ±20% accuracy is acceptable.

    Args:
        text: Input text.

    Returns:
        Estimated token count (always >= 1).
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate ``text`` so that its approximate token count does not exceed ``max_tokens``.

    Truncation happens at word boundaries to avoid cutting mid-word.

    Args:
        text: Input text.
        max_tokens: Maximum allowed approximate token count.

    Returns:
        Original or truncated text.
    """
    if count_tokens_approx(text) <= max_tokens:
        return text

    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    truncated = text[:max_chars]
    # Find the last whitespace within the truncated region for a clean cut
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return truncated + " …[truncated]"


def fits_in_budget(texts: list[str], max_tokens: int) -> list[str]:
    """Return as many texts as fit within ``max_tokens``, in order.

    Args:
        texts: Ordered list of text strings to include.
        max_tokens: Cumulative token budget.

    Returns:
        Subset of ``texts`` that fits within the budget.
    """
    budget = max_tokens
    result: list[str] = []
    for t in texts:
        cost = count_tokens_approx(t)
        if cost > budget:
            break
        result.append(t)
        budget -= cost
    return result
