from __future__ import annotations

import io
import mimetypes
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Magic byte signatures for allowed file types.
# We check the actual file content, not just the filename extension.
_MAGIC_BYTES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": b"PK\x03\x04",
}

_ALLOWED_EXTENSIONS: set[str] = {".pdf", ".docx"}


def validate_upload(
    filename: str,
    content: bytes,
    declared_mime_type: str | None = None,
) -> str:
    """Validate an uploaded file by size, extension, and magic bytes.

    Args:
        filename: Original filename from the upload form.
        content: Raw file content bytes.
        declared_mime_type: MIME type declared by the client (not trusted alone).

    Returns:
        Detected MIME type string if validation passes.

    Raises:
        FileTooLargeError: Content exceeds the configured maximum.
        UnsupportedFileTypeError: Extension or magic bytes do not match.
    """
    # ── 1. Size check ─────────────────────────────────────────────────────────
    if len(content) > settings.max_upload_size_bytes:
        raise FileTooLargeError(
            f"File size {len(content):,} bytes exceeds the limit of "
            f"{settings.max_upload_size_bytes:,} bytes "
            f"({settings.max_upload_size_bytes // 1_048_576} MB)."
        )

    if len(content) == 0:
        raise UnsupportedFileTypeError("Uploaded file is empty.")

    # ── 2. Extension check ────────────────────────────────────────────────────
    suffix = Path(filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"File extension {suffix!r} is not supported. "
            f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )

    # ── 3. Magic bytes check (content-based type detection) ───────────────────
    detected_mime: str | None = None
    for mime_type, magic in _MAGIC_BYTES.items():
        if content[: len(magic)] == magic:
            detected_mime = mime_type
            break

    if detected_mime is None:
        raise UnsupportedFileTypeError(
            "File content does not match any supported format. "
            "Ensure the file is a valid PDF or DOCX."
        )

    # ── 4. Consistency check: extension ↔ detected MIME ───────────────────────
    expected_mime_for_ext = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if expected_mime_for_ext.get(suffix) != detected_mime:
        raise UnsupportedFileTypeError(
            f"File extension {suffix!r} does not match detected content type. "
            "Possible file corruption or mismatch."
        )

    logger.debug(
        "File validation passed: filename=%r size=%d mime=%s",
        filename,
        len(content),
        detected_mime,
    )
    return detected_mime
