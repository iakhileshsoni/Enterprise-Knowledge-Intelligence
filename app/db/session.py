from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# ── Engine ────────────────────────────────────────────────────────────────────
# SQLite-specific tuning:
#   - connect_args={"check_same_thread": False} is required for FastAPI because
#     a request may be handled on a different thread than the one that opened
#     the connection.
#   - WAL mode dramatically improves concurrent read throughput with SQLite.
#   - foreign_keys=ON ensures referential integrity (SQLite disables it by default).

connect_args: dict = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug,
    # SQLite note: pool_size / max_overflow don't apply to StaticPool / NullPool.
    # For PostgreSQL, add: pool_size=5, max_overflow=10, pool_pre_ping=True
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _connection_record) -> None:  # type: ignore[type-arg]
    """Apply SQLite performance and correctness PRAGMAs on every new connection."""
    if settings.database_url.startswith("sqlite"):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA cache_size=-64000;")  # 64 MB page cache
        cursor.close()


# ── Session Factory ───────────────────────────────────────────────────────────
SessionLocal: sessionmaker[Session] = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


# ── FastAPI Dependency ────────────────────────────────────────────────────────
def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after the request.

    Usage in FastAPI routes::

        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
