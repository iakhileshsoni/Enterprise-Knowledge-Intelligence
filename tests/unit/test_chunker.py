from __future__ import annotations

import pytest

from app.ingestion.chunker import StructureAwareChunker, _split_text_recursive
from app.ingestion.parsers.base import ParsedDocument, TextBlock


def _make_doc(blocks: list[tuple[str, int, str | None]]) -> ParsedDocument:
    """Helper: create ParsedDocument from (text, page, heading) tuples."""
    return ParsedDocument(
        blocks=[
            TextBlock(text=t, page_number=p, section_heading=h)
            for t, p, h in blocks
        ]
    )


class TestRecursiveSplitter:
    def test_short_text_returned_as_is(self):
        result = _split_text_recursive("hello world", chunk_size=100, chunk_overlap=10, min_chunk_size=5)
        assert result == ["hello world"]

    def test_long_text_split_at_paragraph(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = _split_text_recursive(text, chunk_size=25, chunk_overlap=5, min_chunk_size=5)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 35  # Allow slight overage due to overlap

    def test_min_chunk_size_filters_tiny_chunks(self):
        text = "Hi.\n\nThis is a longer paragraph that should be kept."
        result = _split_text_recursive(text, chunk_size=60, chunk_overlap=5, min_chunk_size=10)
        for chunk in result:
            assert len(chunk) >= 10

    def test_empty_text_returns_empty(self):
        result = _split_text_recursive("", chunk_size=100, chunk_overlap=10, min_chunk_size=5)
        assert result == []


class TestStructureAwareChunker:
    def test_empty_document_returns_no_chunks(self):
        chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
        result = chunker.chunk(ParsedDocument())
        assert result == []

    def test_single_block_under_chunk_size(self):
        doc = _make_doc([("This is a short block.", 1, "Introduction")])
        chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].section_heading == "Introduction"
        assert chunks[0].page_number == 1

    def test_heading_preserved_across_sub_chunks(self):
        # Create a block that will be split
        long_text = "word " * 300  # 1500 chars
        doc = _make_doc([(long_text, 1, "Policy Section")])
        chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.section_heading == "Policy Section"

    def test_different_headings_produce_separate_chunk_groups(self):
        doc = _make_doc([
            ("Block 1 content.", 1, "Section A"),
            ("Block 2 content.", 1, "Section B"),
        ])
        chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=5)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        assert chunks[0].section_heading == "Section A"
        assert chunks[1].section_heading == "Section B"

    def test_chunk_indices_sequential(self):
        doc = _make_doc([
            ("Block A.", 1, "A"),
            ("Block B.", 2, "B"),
            ("Block C.", 3, "C"),
        ])
        chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50, min_chunk_size=5)
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_version_attribute_exists(self):
        assert StructureAwareChunker.VERSION is not None
