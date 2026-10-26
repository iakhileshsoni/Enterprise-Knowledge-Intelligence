from __future__ import annotations

import pytest

from app.generation.citation_validator import CitationValidator
from app.generation.prompt_builder import INSUFFICIENT_EVIDENCE_MARKER
from app.retrieval.hybrid_retriever import RetrievedChunk


def _make_chunk(chunk_id: str, doc_id: str, text: str = "sample text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        filename="test_doc.docx",
        text=text,
        page_number=1,
        section_heading="Test Section",
        rrf_score=0.5,
    )


class TestCitationValidator:
    def test_insufficient_evidence_marker_detected(self):
        validator = CitationValidator()
        assert validator.is_insufficient_evidence(INSUFFICIENT_EVIDENCE_MARKER)

    def test_empty_output_is_insufficient(self):
        validator = CitationValidator()
        assert validator.is_insufficient_evidence("")
        assert validator.is_insufficient_evidence("   ")

    def test_valid_citation_extracted(self):
        validator = CitationValidator()
        chunk = _make_chunk("chunk-1", "doc-1")
        source_map = {1: "chunk-1"}
        retrieved = [chunk]
        answer, citations = validator.validate_and_extract(
            "The policy states X. [SOURCE 1]", source_map, retrieved
        )
        assert answer is not None
        assert len(citations) == 1
        assert citations[0].chunk_id == "chunk-1"
        assert citations[0].document_id == "doc-1"
        assert citations[0].filename == "test_doc.docx"

    def test_fabricated_source_number_rejected(self):
        validator = CitationValidator()
        chunk = _make_chunk("chunk-1", "doc-1")
        source_map = {1: "chunk-1"}
        retrieved = [chunk]
        # LLM claims [SOURCE 5] which doesn't exist in source_map
        answer, citations = validator.validate_and_extract(
            "The answer is X. [SOURCE 5]", source_map, retrieved
        )
        assert answer is not None
        assert len(citations) == 0  # Rejected fabricated citation

    def test_multiple_valid_citations(self):
        validator = CitationValidator()
        chunk1 = _make_chunk("chunk-1", "doc-1", "Leave policy text")
        chunk2 = _make_chunk("chunk-2", "doc-2", "Remote work text")
        source_map = {1: "chunk-1", 2: "chunk-2"}
        retrieved = [chunk1, chunk2]
        answer, citations = validator.validate_and_extract(
            "Leave [SOURCE 1] and remote work [SOURCE 2].", source_map, retrieved
        )
        assert len(citations) == 2
        assert {c.chunk_id for c in citations} == {"chunk-1", "chunk-2"}

    def test_duplicate_source_references_deduplicated(self):
        validator = CitationValidator()
        chunk = _make_chunk("chunk-1", "doc-1")
        source_map = {1: "chunk-1"}
        retrieved = [chunk]
        # LLM references [SOURCE 1] twice
        answer, citations = validator.validate_and_extract(
            "First mention [SOURCE 1]. Second mention [SOURCE 1].", source_map, retrieved
        )
        assert len(citations) == 1  # Deduplicated

    def test_insufficient_evidence_response_returns_no_answer(self):
        validator = CitationValidator()
        source_map = {1: "chunk-1"}
        retrieved = [_make_chunk("chunk-1", "doc-1")]
        answer, citations = validator.validate_and_extract(
            INSUFFICIENT_EVIDENCE_MARKER, source_map, retrieved
        )
        assert answer is None
        assert citations == []

    def test_excerpt_truncated_to_250_chars(self):
        long_text = "x" * 400
        validator = CitationValidator()
        chunk = _make_chunk("chunk-1", "doc-1", text=long_text)
        source_map = {1: "chunk-1"}
        answer, citations = validator.validate_and_extract(
            "Answer [SOURCE 1]", source_map, [chunk]
        )
        assert len(citations) == 1
        assert len(citations[0].excerpt) <= 254  # 250 + "…"
