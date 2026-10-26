from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 65_536) -> str:
    """Compute the SHA-256 hex digest of a file without loading it entirely into memory.

    Args:
        path: Absolute or relative path to the file.
        chunk_size: Read buffer size in bytes (default 64 KB).

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while data := fh.read(chunk_size):
            digest.update(data)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hex digest of an in-memory bytes object."""
    return hashlib.sha256(data).hexdigest()
