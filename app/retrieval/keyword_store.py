from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# FTS5 DDL
_CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_id    UNINDEXED,
    document_id UNINDEXED,
    text,
    tokenize = 'porter ascii'
);
"""

_CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS fts_metadata (
    chunk_id    TEXT PRIMARY KEY,
    document_id TEXT NOT NULL
);
"""


class KeywordStore:
    """SQLite FTS5-based keyword search store.

    Separate from the main SQLAlchemy session because:
    1. FTS5 virtual tables require raw SQLite, not SQLAlchemy ORM.
    2. We want independent connection management for the FTS index.

    The store uses a separate SQLite file (``fts_index.db``) in the
    vector_index_dir so that deleting the FTS index does not touch the main DB.

    Authorization:
    - ``search()`` accepts ``authorized_doc_ids`` and filters with a subquery.
    - FTS5 MATCH is always scoped to authorized documents.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or (settings.vector_index_dir / "fts_index.db")
        self._lock = threading.Lock()
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _ensure_tables(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.execute(_CREATE_FTS_TABLE)
            conn.execute(_CREATE_METADATA_TABLE)
            conn.commit()

    def add(self, chunk_id: str, document_id: str, text: str) -> None:
        """Insert a single chunk into the FTS index."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO fts_chunks(chunk_id, document_id, text) VALUES (?,?,?)",
                    (chunk_id, document_id, text),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO fts_metadata(chunk_id, document_id) VALUES (?,?)",
                    (chunk_id, document_id),
                )
                conn.commit()

    def add_batch(self, records: list[tuple[str, str, str]]) -> None:
        """Insert many (chunk_id, document_id, text) tuples efficiently."""
        if not records:
            return
        with self._lock:
            with self._get_conn() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO fts_chunks(chunk_id, document_id, text) VALUES (?,?,?)",
                    records,
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO fts_metadata(chunk_id, document_id) VALUES (?,?)",
                    [(r[0], r[1]) for r in records],
                )
                conn.commit()
        logger.debug("FTS: indexed %d chunks", len(records))

    def search(
        self,
        query: str,
        authorized_doc_ids: list[str],
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """Search the FTS index for ``query`` within authorized documents.

        Args:
            query: Natural-language query (FTS5 MATCH syntax; special chars escaped).
            authorized_doc_ids: Restrict results to these document IDs.
            k: Maximum results to return.

        Returns:
            List of (chunk_id, bm25_score) tuples sorted by descending score.
            BM25 score from FTS5 is negative (lower = better); we negate it for
            consistency with the vector retriever (higher = better).
        """
        if not authorized_doc_ids:
            return []

        # Escape FTS5 special characters to prevent query injection
        clean_query = _escape_fts5_query(query)
        if not clean_query:
            return []

        placeholders = ",".join("?" * len(authorized_doc_ids))
        sql = f"""
            SELECT chunk_id, -bm25(fts_chunks) AS score
            FROM fts_chunks
            WHERE fts_chunks MATCH ?
              AND document_id IN ({placeholders})
            ORDER BY score DESC
            LIMIT ?
        """
        params = [clean_query] + authorized_doc_ids + [k]

        with self._get_conn() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError as exc:
                # Malformed FTS query
                logger.warning("FTS5 query error: %s (query=%r)", exc, query)
                return []

        return [(row[0], float(row[1])) for row in rows]

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks for a document from the FTS index."""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    "DELETE FROM fts_chunks WHERE document_id = ?", (document_id,)
                )
                conn.execute(
                    "DELETE FROM fts_metadata WHERE document_id = ?", (document_id,)
                )
                conn.commit()
        logger.debug("FTS: deleted chunks for document %s", document_id)

    def is_ready(self) -> bool:
        return self._db_path.exists()


def _escape_fts5_query(query: str) -> str:
    """Escape FTS5 special characters in a query string.

    FTS5 special characters: " ( ) - NOT AND OR ^
    We wrap the cleaned query in double-quotes for phrase matching,
    then fall back to simple word matching if the query is short.
    """
    # Remove characters that break FTS5 syntax
    import re
    # Keep alphanumeric, spaces, hyphens, apostrophes
    clean = re.sub(r'[^\w\s\'-]', ' ', query)
    clean = clean.strip()
    if not clean:
        return ""
    # Use OR between words for better recall
    words = clean.split()
    if not words:
        return ""
    return " OR ".join(words)
