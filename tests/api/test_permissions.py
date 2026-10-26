from __future__ import annotations

import pytest


class TestPermissions:
    """Tests proving that authorization is enforced at the service/repository level."""

    def test_user_cannot_access_others_document(self, db, test_user, other_user):
        from app.repositories.document_repo import DocumentRepository
        from app.models.document import Document
        # Create a document owned by test_user
        doc_repo = DocumentRepository(db)
        doc = doc_repo.create(
            owner_id=test_user.id,
            filename="test.docx",
            original_filename="test.docx",
            file_hash="abc123",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            file_size_bytes=1000,
            storage_path="/fake/path/test.docx",
        )
        assert not doc_repo.user_can_read(doc.id, other_user.id)

    def test_owner_can_access_own_document(self, db, test_user):
        from app.repositories.document_repo import DocumentRepository
        doc_repo = DocumentRepository(db)
        doc = doc_repo.create(
            owner_id=test_user.id,
            filename="mine.docx",
            original_filename="mine.docx",
            file_hash="def456",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            file_size_bytes=500,
            storage_path="/fake/path/mine.docx",
        )
        assert doc_repo.user_can_read(doc.id, test_user.id)

    def test_explicit_permission_grants_access(self, db, test_user, other_user, admin_user):
        from app.repositories.document_repo import DocumentRepository
        doc_repo = DocumentRepository(db)
        doc = doc_repo.create(
            owner_id=test_user.id,
            filename="shared.docx",
            original_filename="shared.docx",
            file_hash="ghi789",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            file_size_bytes=700,
            storage_path="/fake/path/shared.docx",
        )
        # Other user cannot read yet
        assert not doc_repo.user_can_read(doc.id, other_user.id)
        # Grant permission
        doc_repo.grant_permission(doc.id, other_user.id, granted_by=test_user.id)
        # Now other user can read
        assert doc_repo.user_can_read(doc.id, other_user.id)

    def test_admin_sees_all_ready_documents(self, db, test_user, admin_user):
        from app.repositories.document_repo import DocumentRepository
        doc_repo = DocumentRepository(db)
        for i in range(3):
            doc = doc_repo.create(
                owner_id=test_user.id,
                filename=f"doc{i}.docx",
                original_filename=f"doc{i}.docx",
                file_hash=f"hash{i}",
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                file_size_bytes=100,
                storage_path=f"/fake/doc{i}.docx",
            )
            doc_repo.update_status(doc.id, "ready")
        ids = doc_repo.get_authorized_document_ids(admin_user.id, "admin")
        assert len(ids) == 3

    def test_deleted_document_not_in_authorized_list(self, db, test_user):
        from app.repositories.document_repo import DocumentRepository
        doc_repo = DocumentRepository(db)
        doc = doc_repo.create(
            owner_id=test_user.id,
            filename="del.docx",
            original_filename="del.docx",
            file_hash="delhash",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            file_size_bytes=100,
            storage_path="/fake/del.docx",
        )
        doc_repo.update_status(doc.id, "ready")
        ids_before = doc_repo.get_authorized_document_ids(test_user.id, "user")
        assert doc.id in ids_before

        doc_repo.soft_delete(doc.id)
        ids_after = doc_repo.get_authorized_document_ids(test_user.id, "user")
        assert doc.id not in ids_after
