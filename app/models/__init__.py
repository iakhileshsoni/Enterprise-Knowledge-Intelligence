"""ORM model registry.

Import all models here so that:
1. ``Base.metadata`` is fully populated when Alembic runs ``env.py``.
2. SQLAlchemy's relationship resolution can find all classes at import time.
"""
from app.models.user import User
from app.models.document import Document, DocumentPermission, IngestionJob
from app.models.chunk import Chunk
from app.models.conversation import Conversation, Message
from app.models.evaluation import EvalRun

__all__ = [
    "User",
    "Document",
    "DocumentPermission",
    "IngestionJob",
    "Chunk",
    "Conversation",
    "Message",
    "EvalRun",
]
