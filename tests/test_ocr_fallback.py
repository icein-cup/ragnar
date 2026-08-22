from pathlib import Path
from ingestion.parser import DoclingParser


class StubConverter:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def convert(self, path):
        self.calls += 1
        return self._results.pop(0)


class StubDoc:
    def __init__(self, markdown, pages=1):
        self._markdown = markdown
        self._pages = pages

    def export_to_markdown(self):
        return self._markdown

    def iterate_items(self):
        class Item:
            text = self._markdown

            class _P:
                page_no = 1
            prov = [_P()]
        return [(Item(), 0)]


class StubResult:
    def __init__(self, doc):
        self.document = doc


def test_dense_text_does_not_trigger_ocr():
    converter = StubConverter([StubResult(StubDoc("x" * 5000))])
    parser = DoclingParser(converter=converter, ocr_converter=converter)

    parsed = parser.parse(Path("a.pdf"))

    assert converter.calls == 1
    assert parsed.low_confidence is False


def test_sparse_text_triggers_ocr_and_flags_low_confidence():
    plain = StubConverter([StubResult(StubDoc("tiny"))])
    ocr = StubConverter([StubResult(StubDoc("recovered text " * 50))])
    parser = DoclingParser(converter=plain, ocr_converter=ocr)

    parsed = parser.parse(Path("scan.pdf"))

    assert ocr.calls == 1
    assert parsed.low_confidence is True
    assert "recovered" in parsed.markdown


def test_placeholder_only_markdown_triggers_ocr():
    # Docling emits `<!-- image -->` for image-only regions; the placeholders
    # must not count as extracted text for the density check.
    plain = StubConverter([StubResult(StubDoc("<!-- image -->\n<!-- image -->"))])
    ocr = StubConverter([StubResult(StubDoc("recovered text " * 50))])
    parser = DoclingParser(converter=plain, ocr_converter=ocr)

    parsed = parser.parse(Path("scan.pdf"))

    assert ocr.calls == 1
    assert parsed.low_confidence is True


class EmptyBlocksDoc(StubDoc):
    """Markdown is dense with placeholders, but no text blocks are extracted."""

    def iterate_items(self):
        return []


def test_empty_blocks_trigger_ocr_even_with_dense_placeholder_markdown():
    plain = StubConverter([StubResult(EmptyBlocksDoc("<!-- image -->\n" * 100))])
    ocr = StubConverter([StubResult(StubDoc("recovered text " * 50))])
    parser = DoclingParser(converter=plain, ocr_converter=ocr)

    parsed = parser.parse(Path("scan.pdf"))

    assert ocr.calls == 1
    assert parsed.low_confidence is True
