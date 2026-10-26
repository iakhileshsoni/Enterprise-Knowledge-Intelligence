from __future__ import annotations

import pytest

from app.core.exceptions import DocumentNotFoundError, DuplicateDocumentError, UnsupportedFileTypeError
from app.repositories.document_repo import DocumentRepository
from app.services.document_service import DocumentService


def _minimal_docx_bytes() -> bytes:
    """Create a minimal valid DOCX file in memory."""
    from docx import Document
    import io
    doc = Document()
    doc.add_paragraph("This is a test document with policy information.")
    doc.add_paragraph("Employees are entitled to 20 days of annual leave.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


class TestDocumentUpload:
    def test_upload_valid_docx(self, db, vector_store, keyword_store, test_user, tmp_path):
        import os
        os.environ["UPLOAD_DIR"] = str(tmp_path / "uploads")
        from app.core.config import get_settings
        settings = get_settings()
        settings.__dict__["upload_dir"] = tmp_path / "uploads"
        (tmp_path / "uploads").mkdir(exist_ok=True)

        svc = DocumentService(db, vector_store, keyword_store)
        content = _minimal_docx_bytes()
        doc = svc.handle_upload(
            owner_id=test_user.id,
            filename="test_policy.docx",
            content=content,
        )
        assert doc.id is not None
        assert doc.status == "pending"
        assert doc.owner_id == test_user.id

    def test_upload_invalid_extension_raises(self, db, vector_store, keyword_store, test_user):
        svc = DocumentService(db, vector_store, keyword_store)
        with pytest.raises(UnsupportedFileTypeError):
            svc.handle_upload(
                owner_id=test_user.id,
                filename="malware.exe",
                content=b"MZ\x00\x00",
            )

    def test_upload_empty_file_raises(self, db, vector_store, keyword_store, test_user):
        svc = DocumentService(db, vector_store, keyword_store)
        with pytest.raises(UnsupportedFileTypeError):
            svc.handle_upload(
                owner_id=test_user.id,
                filename="empty.docx",
                content=b"",
            )

    def test_duplicate_upload_raises(self, db, vector_store, keyword_store, test_user, tmp_path):
        from app.core.config import get_settings
        settings = get_settings()
        settings.__dict__["upload_dir"] = tmp_path / "uploads"
        (tmp_path / "uploads").mkdir(exist_ok=True)

        svc = DocumentService(db, vector_store, keyword_store)
        content = _minimal_docx_bytes()
        svc.handle_upload(owner_id=test_user.id, filename="policy.docx", content=content)

        with pytest.raises(DuplicateDocumentError):
            svc.handle_upload(owner_id=test_user.id, filename="policy.docx", content=content)


class TestDocumentIngestion:
    def test_full_ingestion_pipeline(self, db, vector_store, keyword_store, test_user, tmp_path):
        """Test that a DOCX can be ingested end-to-end."""
        from app.core.config import get_settings
        settings = get_settings()
        settings.__dict__["upload_dir"] = tmp_path / "uploads"
        (tmp_path / "uploads").mkdir(exist_ok=True)
        settings.__dict__["vector_index_dir"] = tmp_path / "vector_index"
        (tmp_path / "vector_index").mkdir(exist_ok=True)

        svc = DocumentService(db, vector_store, keyword_store)
        content = _minimal_docx_bytes()
        doc = svc.handle_upload(
            owner_id=test_user.id,
            filename="leave_policy.docx",
            content=content,
        )
        # Run synchronously for testing
        svc.run_ingestion(doc.id)

        # Reload from DB
        doc_repo = DocumentRepository(db)
        updated = doc_repo.get_by_id(doc.id)
        assert updated.status == "ready"
        assert vector_store.total_vectors > 0

    def test_authorization_enforcement(self, db, vector_store, keyword_store, test_user, other_user):
        """User B cannot access User A's documents."""
        doc_repo = DocumentRepository(db)
        assert not doc_repo.user_can_read("nonexistent-doc", other_user.id)
