from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.normalizer import normalize_text
from app.ingestion.parsers.base import ParsedDocument, TextBlock

logger = get_logger(__name__)
settings = get_settings()

CHUNKER_VERSION = "1.0"


@dataclass
class Chunk:
    """A single text chunk ready for embedding and indexing."""

    text: str
    chunk_index: int
    page_number: int = 1
    section_heading: str | None = None
    char_start: int | None = None
    char_end: int | None = None


def _split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> list[str]:
    """Recursively split text at paragraph → sentence → word boundaries.

    This is the fallback strategy used inside each heading section when the
    section text exceeds ``chunk_size`` characters.

    The hierarchy is:
    1. Try to split at double-newlines (paragraph boundaries).
    2. If segments are still too large, split at sentence-ending punctuation.
    3. If still too large, split at whitespace (word boundaries).
    4. As a last resort, split at character boundaries.
    """
    if len(text) <= chunk_size:
        if len(text) >= min_chunk_size:
            return [text]
        return []

    separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    for sep in separators:
        if sep and sep not in text:
            continue

        if sep:
            parts = text.split(sep)
        else:
            # Character-level split as last resort
            parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        # Re-combine parts into chunks with overlap
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current and len(current) >= min_chunk_size:
                    chunks.append(current)
                # Overlap: carry over the tail of the current chunk
                if chunk_overlap > 0 and current:
                    overlap_text = current[-chunk_overlap:].strip()
                    current = (overlap_text + sep + part).strip() if overlap_text else part.strip()
                else:
                    current = part.strip()

        if current and len(current) >= min_chunk_size:
            chunks.append(current)

        if chunks:
            return chunks

    return [text] if len(text) >= min_chunk_size else []


class StructureAwareChunker:
    """Heading-aware chunker with recursive fallback.

    Algorithm:
    1. Group TextBlocks by their ``section_heading``.
    2. Concatenate blocks within the same heading into a section string.
    3. If the section fits in one chunk, emit it as-is.
    4. If it is too large, apply ``_split_text_recursive`` to produce
       sub-chunks that still carry the section heading.
    5. Assign chunk_index sequentially across all sections.

    The heading is preserved in every sub-chunk so that citations can reference
    the originating section even when the section is split.
    """

    VERSION = CHUNKER_VERSION

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        min_chunk_size: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size or settings.min_chunk_size

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        """Convert a ParsedDocument into a list of Chunks."""
        if not parsed.blocks:
            return []

        # Group consecutive blocks by section heading
        sections: list[tuple[str | None, list[TextBlock]]] = []
        current_heading: str | None = parsed.blocks[0].section_heading
        current_group: list[TextBlock] = []

        for block in parsed.blocks:
            if block.section_heading != current_heading and current_group:
                sections.append((current_heading, current_group))
                current_heading = block.section_heading
                current_group = [block]
            else:
                current_group.append(block)

        if current_group:
            sections.append((current_heading, current_group))

        # Produce Chunk objects
        result: list[Chunk] = []
        chunk_index = 0

        for heading, blocks in sections:
            section_text = normalize_text(
                "\n\n".join(b.text for b in blocks if b.text.strip())
            )
            if not section_text:
                continue

            page = blocks[0].page_number if blocks else 1
            char_start = blocks[0].char_start if blocks else None
            char_end = blocks[-1].char_end if blocks else None

            if len(section_text) <= self.chunk_size:
                result.append(
                    Chunk(
                        text=section_text,
                        chunk_index=chunk_index,
                        page_number=page,
                        section_heading=heading,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
                chunk_index += 1
            else:
                sub_texts = _split_text_recursive(
                    section_text,
                    self.chunk_size,
                    self.chunk_overlap,
                    self.min_chunk_size,
                )
                for sub in sub_texts:
                    result.append(
                        Chunk(
                            text=sub,
                            chunk_index=chunk_index,
                            page_number=page,
                            section_heading=heading,
                            char_start=None,  # sub-positions not tracked
                            char_end=None,
                        )
                    )
                    chunk_index += 1

        logger.debug(
            "Chunker: %d blocks → %d chunks (heading_aware + recursive fallback)",
            len(parsed.blocks),
            len(result),
        )
        return result
