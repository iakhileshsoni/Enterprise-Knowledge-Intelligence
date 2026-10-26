from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Allowed characters in sanitized filenames
_SAFE_FILENAME_RE = re.compile(r"[^\w\-.]")
_MAX_FILENAME_LENGTH = 200


def safe_filename(original: str) -> str:
    """Produce a safe, path-traversal-resistant filename.

    Strategy:
    - Replace any character that is not alphanumeric, hyphen, underscore, or dot.
    - Collapse consecutive dots (prevents ``..`` traversal).
    - Truncate to a safe length.
    - Prepend a UUID prefix for uniqueness (prevents overwrite collisions).

    Args:
        original: The original filename from the upload.

    Returns:
        A safe filename string (not a full path).
    """
    # Get just the base name — prevents path components in the filename
    base = Path(original).name
    # Replace unsafe characters with underscores
    sanitized = _SAFE_FILENAME_RE.sub("_", base)
    # Remove consecutive dots
    sanitized = re.sub(r"\.{2,}", ".", sanitized)
    # Truncate stem to avoid filesystem limits
    stem = Path(sanitized).stem[:80]
    suffix = Path(sanitized).suffix.lower()
    sanitized = f"{stem}{suffix}"
    if not sanitized or sanitized == ".":
        sanitized = "unnamed"

    # Prepend UUID for uniqueness
    unique_name = f"{uuid.uuid4().hex[:8]}_{sanitized}"
    return unique_name[:_MAX_FILENAME_LENGTH]


def safe_upload_path(upload_dir: Path, filename: str) -> Path:
    """Return the absolute upload path, verifying it stays within upload_dir.

    This prevents path traversal attacks where a malicious filename could
    escape the intended upload directory.

    Args:
        upload_dir: The configured upload directory (from Settings).
        filename: A sanitized filename (should already be processed by safe_filename).

    Returns:
        Resolved absolute Path within upload_dir.

    Raises:
        ValidationError: If the resolved path escapes upload_dir.
    """
    upload_dir = upload_dir.resolve()
    target = (upload_dir / filename).resolve()

    # Verify the resolved path is inside the upload directory
    try:
        target.relative_to(upload_dir)
    except ValueError:
        logger.error(
            "Path traversal attempt detected! filename=%r resolved_path=%s",
            filename,
            target,
        )
        raise ValidationError("Invalid file path detected.")

    return target
