from __future__ import annotations


class AppError(Exception):
    """Base class for all application-level exceptions.

    Sub-classes set class-level defaults for ``status_code`` and ``error_code``
    so they can be caught and serialised uniformly in the exception handlers
    registered in ``app/main.py``.
    """

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.__class__.message
        super().__init__(self.message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(error_code={self.error_code!r}, message={self.message!r})"


# ── 4xx Client Errors ─────────────────────────────────────────────────────────

class NotFoundError(AppError):
    status_code = 404
    error_code = "NOT_FOUND"
    message = "Resource not found."


class PermissionDeniedError(AppError):
    status_code = 403
    error_code = "PERMISSION_DENIED"
    message = "You do not have permission to access this resource."


class AuthenticationError(AppError):
    status_code = 401
    error_code = "AUTHENTICATION_FAILED"
    message = "Authentication failed. Please provide a valid token."


class ValidationError(AppError):
    status_code = 422
    error_code = "VALIDATION_ERROR"
    message = "Input validation failed."


class ConflictError(AppError):
    status_code = 409
    error_code = "CONFLICT"
    message = "Resource conflict."


class RateLimitError(AppError):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please wait before retrying."


# ── Document Errors ───────────────────────────────────────────────────────────

class DocumentNotFoundError(NotFoundError):
    error_code = "DOCUMENT_NOT_FOUND"
    message = "Document not found or access denied."


class DocumentProcessingError(AppError):
    status_code = 422
    error_code = "DOCUMENT_PROCESSING_ERROR"
    message = "Document could not be processed."


class DuplicateDocumentError(ConflictError):
    error_code = "DUPLICATE_DOCUMENT"
    message = "An identical document has already been uploaded."


class UnsupportedFileTypeError(AppError):
    status_code = 415
    error_code = "UNSUPPORTED_FILE_TYPE"
    message = "File type is not supported. Accepted types: PDF, DOCX."


class FileTooLargeError(AppError):
    status_code = 413
    error_code = "FILE_TOO_LARGE"
    message = "File exceeds the maximum allowed upload size."


class IngestionError(AppError):
    status_code = 500
    error_code = "INGESTION_ERROR"
    message = "Document ingestion failed. See server logs for details."


# ── Conversation Errors ───────────────────────────────────────────────────────

class ConversationNotFoundError(NotFoundError):
    error_code = "CONVERSATION_NOT_FOUND"
    message = "Conversation not found."


# ── LLM / Generation Errors ───────────────────────────────────────────────────

class LLMUnavailableError(AppError):
    status_code = 503
    error_code = "LLM_UNAVAILABLE"
    message = "Language model is currently unavailable. Please try again later."


class LLMTimeoutError(AppError):
    status_code = 504
    error_code = "LLM_TIMEOUT"
    message = "Language model request timed out."


class InsufficientEvidenceError(AppError):
    """Not a real HTTP error — the caller returns a 200 with abstain=True."""
    status_code = 200
    error_code = "INSUFFICIENT_EVIDENCE"
    message = "Insufficient evidence found in the knowledge base to answer this question."


# ── Index / Retrieval Errors ──────────────────────────────────────────────────

class IndexNotReadyError(AppError):
    status_code = 503
    error_code = "INDEX_NOT_READY"
    message = "Vector index is not ready. Please wait for document ingestion to complete."


class RetrievalError(AppError):
    status_code = 500
    error_code = "RETRIEVAL_ERROR"
    message = "An error occurred during retrieval."


# ── User Errors ───────────────────────────────────────────────────────────────

class UserNotFoundError(NotFoundError):
    error_code = "USER_NOT_FOUND"
    message = "User not found."


class UserAlreadyExistsError(ConflictError):
    error_code = "USER_ALREADY_EXISTS"
    message = "A user with this username or email already exists."
