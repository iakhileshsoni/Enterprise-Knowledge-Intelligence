from __future__ import annotations

import re

import fitz  # PyMuPDF

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger
from app.ingestion.parsers.base import BaseParser, ParsedDocument, TextBlock

logger = get_logger(__name__)

# Heuristic: lines that look like headings (short, often bold, no period at end)
_HEADING_PATTERN = re.compile(
    r"^(?:\d+[\.\)]\s+)?[A-Z][A-Za-z0-9 \-:,\(\)]{3,80}$"
)


def _looks_like_heading(text: str) -> bool:
    """Return True if the text looks like a section heading.

    This is a heuristic, not a guarantee.  PyMuPDF's span flags can provide
    font-weight information which is more reliable for PDFs that embed font
    metadata.
    """
    text = text.strip()
    if not text or len(text) > 120 or "\n" in text:
        return False
    return bool(_HEADING_PATTERN.match(text))


class PDFParser(BaseParser):
    """Extract text from PDF files using PyMuPDF (fitz).

    Each page is processed independently.  Headings are detected heuristically
    by examining span flags (bold) and text length.  The extracted TextBlocks
    carry page numbers and the best-guess section heading at the time of extraction.

    OCR is NOT included in this implementation.  Scanned PDFs will produce
    empty or very sparse text.  OCR support can be added via pytesseract as a
    Phase 3 extension.
    """

    @property
    def supported_mime_types(self) -> list[str]:
        return ["application/pdf"]

    def parse(self, file_path: str) -> ParsedDocument:
        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise DocumentProcessingError(
                f"Cannot open PDF: {file_path} — {exc}"
            ) from exc

        blocks: list[TextBlock] = []
        current_heading: str | None = None

        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": doc.page_count,
            "source": "pymupdf",
        }

        char_offset = 0

        for page_num, page in enumerate(doc, start=1):
            # get_text("dict") returns structured block/span data
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block; 1 = image
                    continue

                block_text_parts: list[str] = []

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue

                        # Detect headings via bold flag (bit 4) or font size heuristic
                        is_bold = bool(span.get("flags", 0) & (1 << 4))
                        if is_bold and _looks_like_heading(span_text):
                            current_heading = span_text

                        block_text_parts.append(span.get("text", ""))

                block_text = " ".join(block_text_parts).strip()
                if not block_text:
                    continue

                # Also check if the whole block looks like a heading
                if _looks_like_heading(block_text):
                    current_heading = block_text

                char_end = char_offset + len(block_text)
                blocks.append(
                    TextBlock(
                        text=block_text,
                        page_number=page_num,
                        section_heading=current_heading,
                        char_start=char_offset,
                        char_end=char_end,
                    )
                )
                char_offset = char_end + 2  # account for the "\n\n" join

        doc.close()
        logger.debug(
            "PDFParser: %d blocks extracted from %d pages",
            len(blocks),
            metadata["page_count"],
        )
        return ParsedDocument(blocks=blocks, metadata=metadata)
