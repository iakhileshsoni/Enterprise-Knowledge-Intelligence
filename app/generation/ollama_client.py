from __future__ import annotations

import json
from typing import Any, AsyncIterator, Optional

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMTimeoutError, LLMUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class OllamaClient:
    """Async HTTP client for the Ollama local model runtime.

    Ollama API reference: https://github.com/ollama/ollama/blob/main/docs/api.md

    Connection note (Docker):
    - When running inside a Docker container on Windows, set
      ``OLLAMA_BASE_URL=http://host.docker.internal:11434``
      to reach the Ollama server running on the host.
    - When running directly on the host, use ``http://localhost:11434``.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self._model = model or settings.llm_model
        self._timeout = timeout or settings.llm_timeout_seconds

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a generation request to Ollama and return the complete response text.

        Args:
            prompt: The user/context prompt.
            system: Optional system message.
            temperature: Sampling temperature (default from settings).

        Returns:
            Generated text as a single string.

        Raises:
            LLMUnavailableError: If the Ollama server is unreachable.
            LLMTimeoutError: If the request exceeds the configured timeout.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or settings.llm_temperature,
                "num_ctx": settings.llm_context_window,
            },
        }
        if system:
            payload["system"] = system

        url = f"{self._base_url}/api/generate"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")

        except httpx.TimeoutException as exc:
            logger.error("Ollama request timed out after %.1f s", self._timeout)
            raise LLMTimeoutError(
                f"LLM request timed out after {self._timeout} seconds."
            ) from exc
        except httpx.ConnectError as exc:
            logger.error("Cannot connect to Ollama at %s: %s", self._base_url, exc)
            raise LLMUnavailableError(
                f"Cannot reach Ollama server at {self._base_url}. "
                "Ensure Ollama is running: `ollama serve`"
            ) from exc
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP error %d: %s", exc.response.status_code, exc.response.text)
            raise LLMUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error calling Ollama")
            raise LLMUnavailableError(f"Unexpected LLM error: {exc}") from exc

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        """Send a chat-format request to Ollama (/api/chat endpoint).

        Args:
            messages: List of {"role": "user|assistant|system", "content": "..."}.
            temperature: Sampling temperature.

        Returns:
            Assistant reply as a string.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or settings.llm_temperature,
                "num_ctx": settings.llm_context_window,
            },
        }

        url = f"{self._base_url}/api/chat"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")

        except httpx.TimeoutException as exc:
            raise LLMTimeoutError() from exc
        except httpx.ConnectError as exc:
            raise LLMUnavailableError() from exc
        except httpx.HTTPStatusError as exc:
            raise LLMUnavailableError(
                f"Ollama returned HTTP {exc.response.status_code}."
            ) from exc

    async def is_available(self) -> bool:
        """Probe the Ollama server and return True if reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def model_exists(self) -> bool:
        """Return True if the configured model is available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                resp.raise_for_status()
                models = [m.get("name", "") for m in resp.json().get("models", [])]
                return any(self._model in m for m in models)
        except Exception:
            return False
