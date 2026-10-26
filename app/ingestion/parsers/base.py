from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """A contiguous block of text with position metadata.

    Parsers produce a list of TextBlocks.  The chunker consumes them.
    """

    text: str
    page_number: int = 1
    # Best-guess section heading active at the time this block starts.
    # Parsers set this from headings in the source document.
    section_heading: str | None = None
    # Character offset within the full document text (set by parsers that track it)
    char_start: int | None = None
    char_end: int | None = None


@dataclass
class ParsedDocument:
    """Result of parsing a single file.

    ``blocks`` is an ordered list of TextBlocks.
    ``metadata`` holds document-level information (title, author, page count, …).
    """

    blocks: list[TextBlock] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Concatenate all block texts with double newlines."""
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


class BaseParser(ABC):
    """Abstract base class for file parsers.

    To add support for a new file type, subclass BaseParser and implement
    ``parse()``.  Register the subclass in ``ParserFactory`` in ``pipeline.py``.
    """

    @property
    @abstractmethod
    def supported_mime_types(self) -> list[str]:
        """Return the list of MIME types this parser handles."""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """Extract text and metadata from the file at ``file_path``.

        Args:
            file_path: Absolute path to the file on disk.

        Returns:
            ParsedDocument with all extracted blocks and metadata.

        Raises:
            DocumentProcessingError: If the file cannot be read or parsed.
        """
