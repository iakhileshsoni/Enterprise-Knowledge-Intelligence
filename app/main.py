from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import app.models  # noqa: F401 — ensure all ORM models are registered

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware, setup_cors
from app.db.base import Base
from app.db.session import engine
from app.retrieval.keyword_store import KeywordStore
from app.retrieval.vector_store.faiss_store import FAISSVectorStore

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise shared resources on startup, clean up on shutdown."""

    # ── Database setup ────────────────────────────────────────────────────────
    logger.info("Creating database tables if they do not exist…")
    Base.metadata.create_all(bind=engine)
    logger.info("Database ready.")

    # ── Vector store (FAISS) ──────────────────────────────────────────────────
    logger.info("Initialising FAISS vector store…")
    vector_store = FAISSVectorStore()
    application.state.vector_store = vector_store
    logger.info(
        "FAISS ready: %d vectors loaded.", vector_store.total_vectors
    )

    # ── Keyword store (FTS5) ──────────────────────────────────────────────────
    logger.info("Initialising keyword store (SQLite FTS5)…")
    keyword_store = KeywordStore()
    application.state.keyword_store = keyword_store
    logger.info("Keyword store ready.")

    logger.info(
        "🚀 %s v%s started in %s mode.",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )

    yield  # Server is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Saving FAISS index before shutdown…")
    if vector_store.total_vectors > 0:
        vector_store.save()
    logger.info("Shutdown complete.")


# ── Application factory ───────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Enterprise Knowledge Assistant — Advanced RAG Platform. "
        "Upload documents, ask questions, receive grounded answers with verifiable citations."
    ),
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
setup_cors(app)
app.add_middleware(RequestContextMiddleware)

# ── Exception Handlers ────────────────────────────────────────────────────────

@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """Unified handler for all AppError subclasses.

    Returns a consistent error envelope without exposing internal details.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        "AppError [%s]: %s (request_id=%s)",
        exc.error_code,
        exc.message,
        request_id,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "request_id": request_id,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unexpected exceptions — never expose stack traces."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled exception (request_id=%s)", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please contact support.",
                "request_id": request_id,
            }
        },
    )


# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.v1 import auth, documents, conversations, evaluation, health  # noqa: E402

API_PREFIX = "/api/v1"

app.include_router(health.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(documents.router, prefix=API_PREFIX)
app.include_router(conversations.router, prefix=API_PREFIX)
app.include_router(evaluation.router, prefix=API_PREFIX)
