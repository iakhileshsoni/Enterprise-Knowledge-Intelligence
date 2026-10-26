from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.user import User
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore
from app.services.auth_service import AuthService, hash_password

from sqlalchemy.pool import StaticPool

# ── In-memory SQLite for tests ────────────────────────────────────────────────
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db() -> Session:
    """Provide a clean in-memory SQLite session for each test function."""
    Base.metadata.create_all(bind=test_engine)
    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def vector_store(tmp_path) -> FAISSVectorStore:
    """Provide a fresh FAISS vector store backed by a temp directory."""
    return FAISSVectorStore(
        index_path=tmp_path / "test.index",
        meta_path=tmp_path / "test_meta.json",
    )


@pytest.fixture(scope="function")
def keyword_store(tmp_path) -> KeywordStore:
    """Provide a fresh FTS5 keyword store backed by a temp DB file."""
    return KeywordStore(db_path=tmp_path / "test_fts.db")


@pytest.fixture(scope="function")
def test_user(db: Session) -> User:
    """Create a standard test user."""
    auth = AuthService(db)
    return auth.register("testuser", "testuser@example.com", "TestPass123!")


@pytest.fixture(scope="function")
def admin_user(db: Session) -> User:
    """Create an admin test user."""
    auth = AuthService(db)
    return auth.register("adminuser", "admin@example.com", "AdminPass123!", role="admin")


@pytest.fixture(scope="function")
def other_user(db: Session) -> User:
    """Create a second standard user (for cross-user isolation tests)."""
    auth = AuthService(db)
    return auth.register("otheruser", "other@example.com", "OtherPass123!")


@pytest.fixture(scope="function")
def test_user_token(db: Session, test_user: User) -> str:
    """Return a valid access token for the test user."""
    auth = AuthService(db)
    token_response = auth.create_token_pair(test_user)
    return token_response.access_token


@pytest.fixture(scope="function")
def admin_token(db: Session, admin_user: User) -> str:
    """Return a valid access token for the admin user."""
    auth = AuthService(db)
    token_response = auth.create_token_pair(admin_user)
    return token_response.access_token


@pytest.fixture(scope="function")
def client(db: Session, vector_store: FAISSVectorStore, keyword_store: KeywordStore):
    """Provide a FastAPI TestClient with the test DB and stores injected."""
    from app.api.dependencies import get_vector_store, get_keyword_store

    def override_db():
        yield db

    def override_vector_store():
        return vector_store

    def override_keyword_store():
        return keyword_store

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_vector_store] = override_vector_store
    app.dependency_overrides[get_keyword_store] = override_keyword_store

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
