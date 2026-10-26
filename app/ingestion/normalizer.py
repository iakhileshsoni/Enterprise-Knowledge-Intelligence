from __future__ import annotations

import re
import unicodedata


# Regex to collapse runs of whitespace (not newlines) to a single space
_WHITESPACE_RE = re.compile(r"[^\S\n]+")
# Regex to collapse 3+ consecutive newlines to exactly 2
_NEWLINES_RE = re.compile(r"\n{3,}")
# Common ligatures and unicode replacements
_LIGATURE_MAP = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u00ad": "",  # soft hyphen
    }
)


def normalize_text(text: str) -> str:
    """Normalize extracted text for embedding and storage.

    Steps:
    1. NFC Unicode normalization (canonical decomposition + composition).
    2. Ligature replacement (fi, fl, ff → ASCII equivalents).
    3. Collapse horizontal whitespace to single space.
    4. Collapse 3+ newlines to 2 (preserve paragraph breaks).
    5. Strip leading/trailing whitespace.

    Args:
        text: Raw extracted text.

    Returns:
        Normalized text ready for chunking.
    """
    if not text:
        return ""

    # Step 1: Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)

    # Step 2: Ligature replacements and curly-quote normalization
    text = text.translate(_LIGATURE_MAP)

    # Step 3: Collapse horizontal whitespace
    text = _WHITESPACE_RE.sub(" ", text)

    # Step 4: Collapse excessive newlines
    text = _NEWLINES_RE.sub("\n\n", text)

    # Step 5: Strip
    return text.strip()
