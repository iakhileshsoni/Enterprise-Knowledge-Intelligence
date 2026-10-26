from __future__ import annotations

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore
from app.security.auth_guards import get_current_user, require_admin

# Re-export for convenience in route files
__all__ = ["get_db", "get_current_user", "require_admin", "get_vector_store", "get_keyword_store"]


def get_vector_store() -> FAISSVectorStore:
    """Return the application-level FAISS vector store singleton.

    The instance is created at application startup in main.py and stored
    on app.state.  This function is a FastAPI dependency that retrieves it.
    """
    from app.main import app
    return app.state.vector_store


def get_keyword_store() -> KeywordStore:
    """Return the application-level FTS5 keyword store singleton."""
    from app.main import app
    return app.state.keyword_store
