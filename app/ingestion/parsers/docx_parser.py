from __future__ import annotations

import re

from docx import Document as DocxDocument
from docx.oxml.ns import qn

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger
from app.ingestion.parsers.base import BaseParser, ParsedDocument, TextBlock

logger = get_logger(__name__)

# Heading styles in python-docx follow the pattern "Heading 1", "Heading 2", etc.
_HEADING_STYLE_RE = re.compile(r"^Heading\s+\d+$", re.IGNORECASE)


def _is_heading_style(style_name: str | None) -> bool:
    if not style_name:
        return False
    return bool(_HEADING_STYLE_RE.match(style_name))


class DOCXParser(BaseParser):
    """Extract text from DOCX files using python-docx.

    Strategy:
    - Iterate paragraphs in document order.
    - Paragraphs with Heading styles update the current section heading.
    - Non-heading paragraphs accumulate text under the current heading.
    - Tables are extracted cell-by-cell and appended as a single block.
    - Page numbers are estimated: DOCX does not expose rendered page breaks
      reliably; we use XML ``<w:lastRenderedPageBreak>`` as a hint.
    """

    @property
    def supported_mime_types(self) -> list[str]:
        return [
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]

    def parse(self, file_path: str) -> ParsedDocument:
        try:
            doc = DocxDocument(file_path)
        except Exception as exc:
            raise DocumentProcessingError(
                f"Cannot open DOCX: {file_path} — {exc}"
            ) from exc

        blocks: list[TextBlock] = []
        current_heading: str | None = None
        current_page = 1
        char_offset = 0

        metadata = {
            "title": doc.core_properties.title or "",
            "author": doc.core_properties.author or "",
            "source": "python-docx",
        }

        # Process paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Detect page breaks embedded in paragraph XML
            if para._p.find(qn("w:lastRenderedPageBreak")) is not None:
                current_page += 1

            style_name = para.style.name if para.style else None
            if _is_heading_style(style_name):
                current_heading = text
                # Include heading as its own block so it is searchable
                char_end = char_offset + len(text)
                blocks.append(
                    TextBlock(
                        text=text,
                        page_number=current_page,
                        section_heading=current_heading,
                        char_start=char_offset,
                        char_end=char_end,
                    )
                )
                char_offset = char_end + 2
            else:
                char_end = char_offset + len(text)
                blocks.append(
                    TextBlock(
                        text=text,
                        page_number=current_page,
                        section_heading=current_heading,
                        char_start=char_offset,
                        char_end=char_end,
                    )
                )
                char_offset = char_end + 2

        # Process tables
        for table in doc.tables:
            rows: list[str] = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                table_text = "\n".join(rows)
                char_end = char_offset + len(table_text)
                blocks.append(
                    TextBlock(
                        text=table_text,
                        page_number=current_page,
                        section_heading=current_heading,
                        char_start=char_offset,
                        char_end=char_end,
                    )
                )
                char_offset = char_end + 2

        logger.debug("DOCXParser: %d blocks extracted", len(blocks))
        return ParsedDocument(blocks=blocks, metadata=metadata)
