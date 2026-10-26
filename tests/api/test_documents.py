from __future__ import annotations

import io
import pytest


def _make_docx_upload(filename: str = "test.docx"):
    from docx import Document
    doc = Document()
    doc.add_paragraph("Test content for the document.")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf, filename


class TestDocumentAPI:
    def test_upload_requires_auth(self, client):
        buf, name = _make_docx_upload()
        resp = client.post(
            "/api/v1/documents",
            files={"file": (name, buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )
        assert resp.status_code == 401

    def test_upload_valid_docx(self, client, test_user_token, tmp_path):
        from app.core.config import get_settings
        s = get_settings()
        s.__dict__["upload_dir"] = tmp_path / "uploads"
        (tmp_path / "uploads").mkdir(exist_ok=True)

        buf, name = _make_docx_upload()
        resp = client.post(
            "/api/v1/documents",
            files={"file": (name, buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "document_id" in data
        assert data["status"] == "pending"

    def test_upload_unsupported_type_returns_415(self, client, test_user_token):
        resp = client.post(
            "/api/v1/documents",
            files={"file": ("bad.txt", io.BytesIO(b"hello"), "text/plain")},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert resp.status_code == 415

    def test_list_documents_requires_auth(self, client):
        resp = client.get("/api/v1/documents")
        assert resp.status_code == 401

    def test_list_documents_empty_for_new_user(self, client, test_user_token):
        resp = client.get(
            "/api/v1/documents",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_cross_user_document_not_visible(self, client, test_user_token, admin_token, tmp_path):
        """User B cannot see User A's document."""
        from app.core.config import get_settings
        s = get_settings()
        s.__dict__["upload_dir"] = tmp_path / "uploads"
        (tmp_path / "uploads").mkdir(exist_ok=True)

        # Admin uploads a document
        buf, name = _make_docx_upload("admin_doc.docx")
        upload_resp = client.post(
            "/api/v1/documents",
            files={"file": (name, buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert upload_resp.status_code == 202
        doc_id = upload_resp.json()["document_id"]

        # testuser tries to access admin's document directly
        resp = client.get(
            f"/api/v1/documents/{doc_id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert resp.status_code == 404  # Returns 404, not 403, to avoid revealing existence
