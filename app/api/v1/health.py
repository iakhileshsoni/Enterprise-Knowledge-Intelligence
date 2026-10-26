from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.generation.ollama_client import OllamaClient
from app.retrieval.vector_store.faiss_store import FAISSVectorStore

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health")
async def health() -> dict:
    """Liveness check — returns 200 if the server process is running."""
    return {"status": "ok", "version": settings.app_version}


@router.get("/ready")
async def readiness() -> JSONResponse:
    """Readiness check — verifies critical dependencies are available.

    Checks:
    - Ollama server reachability
    - Configured LLM model availability
    - FAISS vector index status

    Returns 200 if all checks pass, 503 if any critical check fails.
    """
    checks: dict[str, dict] = {}
    all_ok = True

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama = OllamaClient()
    ollama_available = await ollama.is_available()
    model_available = await ollama.model_exists() if ollama_available else False

    checks["ollama"] = {
        "status": "ok" if ollama_available else "error",
        "url": settings.ollama_base_url,
    }
    checks["llm_model"] = {
        "status": "ok" if model_available else "warning",
        "model": settings.llm_model,
        "detail": (
            "Model ready"
            if model_available
            else "Model not found — run: ollama pull " + settings.llm_model
        ),
    }

    if not ollama_available:
        all_ok = False

    # ── Vector index ──────────────────────────────────────────────────────────
    index_path = settings.faiss_index_path
    checks["vector_index"] = {
        "status": "ok" if index_path.exists() else "warning",
        "path": str(index_path),
        "detail": "Ready" if index_path.exists() else "No documents ingested yet",
    }

    status_code = 200 if all_ok else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ok else "degraded",
            "checks": checks,
        },
    )
