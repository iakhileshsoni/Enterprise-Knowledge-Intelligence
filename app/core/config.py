from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "Enterprise Knowledge Assistant"
    app_version: str = "0.1.0"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/app.db"

    # ── Storage Paths ─────────────────────────────────────────────────────────
    upload_dir: Path = Path("data/uploads")
    vector_index_dir: Path = Path("data/vector_index")
    sample_documents_dir: Path = Path("data/sample_documents")

    # ── Ollama / LLM ──────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b"
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    llm_temperature: float = 0.1
    llm_context_window: int = 4096

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 1000       # characters
    chunk_overlap: int = 150     # characters
    min_chunk_size: int = 100    # discard smaller chunks

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 10        # final results returned after RRF fusion
    retrieval_fetch_k: int = 20      # candidates fetched per source before fusion
    similarity_threshold: float = 0.3
    evidence_threshold: float = 0.25  # below this → abstain

    # ── Reranker (optional, disabled by default) ──────────────────────────────
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5

    # ── Query Rewriting (optional, disabled by default) ───────────────────────
    enable_query_rewriting: bool = False

    # ── Context / Conversation ────────────────────────────────────────────────
    max_context_tokens: int = 2000
    max_conversation_turns: int = 6

    # ── Upload Limits ─────────────────────────────────────────────────────────
    max_upload_size_bytes: int = 52_428_800  # 50 MB
    allowed_mime_types: List[str] = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    # ── Authentication / JWT ──────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_USE_openssl_rand_hex_32"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("upload_dir", "vector_index_dir", "sample_documents_dir", mode="before")
    @classmethod
    def _ensure_directory_exists(cls, v: str | Path) -> Path:
        """Auto-create required directories on startup."""
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ── Derived Properties ────────────────────────────────────────────────────
    @property
    def is_testing(self) -> bool:
        return self.environment == "testing"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def faiss_index_path(self) -> Path:
        return self.vector_index_dir / "faiss.index"

    @property
    def faiss_meta_path(self) -> Path:
        return self.vector_index_dir / "faiss_meta.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings singleton."""
    return Settings()
