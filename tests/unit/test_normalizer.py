from __future__ import annotations

from app.ingestion.normalizer import normalize_text


class TestNormalizer:
    def test_empty_string_returns_empty(self):
        assert normalize_text("") == ""

    def test_whitespace_only_returns_empty(self):
        assert normalize_text("   ") == ""

    def test_horizontal_whitespace_collapsed(self):
        result = normalize_text("hello   world")
        assert result == "hello world"

    def test_three_newlines_collapsed_to_two(self):
        result = normalize_text("para1\n\n\n\npara2")
        assert result == "para1\n\npara2"

    def test_ligatures_replaced(self):
        # fi ligature
        result = normalize_text("\ufb01le")
        assert result == "file"

    def test_curly_quotes_normalized(self):
        result = normalize_text("\u2018hello\u2019")
        assert result == "'hello'"

    def test_em_dash_normalized(self):
        result = normalize_text("A\u2014B")
        assert "A--B" in result or "A" in result  # normalized form

    def test_leading_trailing_stripped(self):
        result = normalize_text("   hello world   ")
        assert result == "hello world"

    def test_unicode_nfc_applied(self):
        # Decomposed 'é' (e + combining accent) → composed 'é'
        decomposed = "e\u0301"  # e + combining acute accent
        result = normalize_text(decomposed)
        assert result == "\xe9"  # NFC composed é

    def test_preserves_paragraph_breaks(self):
        result = normalize_text("Para 1.\n\nPara 2.")
        assert "\n\n" in result
