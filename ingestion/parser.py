from dataclasses import dataclass, field
from pathlib import Path
import re
import threading

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.base_models import InputFormat

# Thread count for Docling's layout/table models is read from the
# DOCLING_NUM_THREADS env var (docling's AcceleratorOptions is a pydantic
# BaseSettings with env_prefix="DOCLING_") — set in docker-compose.yml
# rather than hardcoded here, since the right value depends on how many
# CPUs the container actually gets.


def _default_converter() -> DocumentConverter:
    # OCR is off by default — the corpus is predominantly native-text PDFs,
    # and running OCR unconditionally is both slow and (per exploration)
    # triggers model downloads even when nothing needs OCR'ing. OCR is
    # enabled explicitly elsewhere only when text extraction comes back
    # near-empty.
    options = PdfPipelineOptions()
    options.do_ocr = False
    # ACCURATE (the default) roughly doubles TableFormer's cost for a
    # precision gain this pipeline doesn't need — table content also gets a
    # deterministic aggregate summary block (table_summary.py), so exact
    # cell-level structure isn't load-bearing here.
    options.table_structure_options.mode = TableFormerMode.FAST
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })


def _ocr_converter() -> DocumentConverter:
    # Used as a fallback when the OCR-disabled default converter comes back
    # with near-empty text — typically scanned/image-only PDFs.
    options = PdfPipelineOptions()
    options.do_ocr = True
    options.table_structure_options.mode = TableFormerMode.FAST
    return DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
    })


# Below this density (chars of extracted text per page), we suspect the
# extraction missed content (e.g. a scanned page) and re-parse with OCR.
OCR_TRIGGER_CHARS_PER_PAGE = 50


def _sheet_name(item, doc) -> str | None:
    """Sheet title for a table that came from a spreadsheet, else None.

    In Excel, each table's parent is a group labelled 'sheet' whose name is
    the sheet title. PDFs have no such group, so this returns None and the
    table is located by page number instead.
    """
    parent = getattr(item, "parent", None)
    if parent is None:
        return None
    try:
        group = parent.resolve(doc)
    except Exception:
        return None
    if "sheet" in str(getattr(group, "label", "")).lower():
        return getattr(group, "name", None)
    return None


def _export_table_df(item, doc):
    """A table item's content lives in a structured grid, not `.text` — this
    is the one place that grid is materialized. Exported once per item and
    shared by `_looks_like_a_table` and `_table_summary` below (it used to be
    exported twice, once for each)."""
    try:
        return item.export_to_dataframe(doc)
    except Exception:
        return None


def _looks_like_a_table(df) -> bool:
    """Filters out Docling's occasional misclassification of repetitive or
    fixed-position text as a table.

    A genuine table (PDF or spreadsheet) has at least two columns; a
    single-column match is far more likely to be misread prose than real
    tabular data, and even if it were a genuine single-column table,
    treating it as prose (chunked by the token budget, like any other
    text) is harmless. Doesn't catch every misclassification — a fake
    table can occasionally get split into 2+ columns too — but it's a
    cheap, safe filter for the common case with no real downside.
    """
    if df is None:
        return True  # export failed — can't verify, trust Docling's own classification
    return df.shape[1] >= 2


def _table_summary(df, sheet: str | None) -> str | None:
    """Deterministic aggregate summary for a table item, or None.

    A summary is a nice-to-have on top of the table's own (already-indexed)
    content, never load-bearing for it - any failure here (malformed data,
    an unexpected pandas edge case) must degrade to "no summary" rather than
    take down parsing of the whole document, so the whole thing is one
    try/except rather than two.
    """
    if df is None:
        return None
    from ingestion.table_summary import summarize_table
    try:
        return summarize_table(df, sheet=sheet)
    except Exception:
        return None


@dataclass
class Block:
    text: str
    page: int | None = None
    is_table: bool = False
    sheet: str | None = None
    is_summary: bool = False


@dataclass
class ParsedDocument:
    markdown: str
    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0
    low_confidence: bool = False

    @property
    def chars_per_page(self) -> float:
        if self.page_count == 0:
            return 0.0
        # Docling emits `<!-- image -->` placeholders into the markdown for
        # image-only regions; counting them as text inflates the density and
        # masks a near-empty extraction, so strip HTML comments and whitespace
        # before measuring. Markdown comments carry no indexed content.
        text = re.sub(r"<!--.*?-->", "", self.markdown, flags=re.DOTALL).strip()
        return len(text) / self.page_count


class DoclingParser:
    """Converts a source file into markdown plus provenance-carrying blocks.

    Blocks are what the chunker consumes; the markdown is for human display
    only. Flattening to markdown loses page numbers, so the two are kept
    separate deliberately.

    DocumentConverter is not thread-safe, so converters are stored in a
    threading.local — each worker thread gets its own lazily-created
    instance. The optional explicit ``converter`` / ``ocr_converter``
    arguments (used by tests) are shared as-is; tests run single-threaded.
    # ponytail: N workers = N× converter model memory (~1-2 GB each).
    """

    def __init__(self, converter: DocumentConverter | None = None,
                 ocr_converter: DocumentConverter | None = None):
        self._explicit_converter = converter
        self._explicit_ocr_converter = ocr_converter
        self._tls = threading.local()

    @property
    def _converter(self) -> DocumentConverter:
        if self._explicit_converter is not None:
            return self._explicit_converter
        if not hasattr(self._tls, "converter"):
            self._tls.converter = _default_converter()
        return self._tls.converter

    @property
    def _ocr_converter(self) -> DocumentConverter | None:
        if not hasattr(self._tls, "ocr_converter"):
            self._tls.ocr_converter = self._explicit_ocr_converter
        return self._tls.ocr_converter

    @_ocr_converter.setter
    def _ocr_converter(self, value: DocumentConverter | None) -> None:
        self._tls.ocr_converter = value

    def parse(self, path: Path) -> ParsedDocument:
        parsed = self._parse_with(self._converter, path)

        # Trigger OCR when the first-pass extraction looks empty (no blocks or
        # zero meaningful text), or when a multi-page document is suspiciously
        # sparse. A short but valid one-page document (cover sheet, memo) is
        # allowed to stay below the 50-char/page threshold without being flagged
        # as low-confidence.
        if (
            not parsed.blocks
            or parsed.chars_per_page == 0
            or (parsed.page_count > 1 and parsed.chars_per_page < OCR_TRIGGER_CHARS_PER_PAGE)
        ):
            if self._ocr_converter is None:
                self._ocr_converter = _ocr_converter()
            parsed = self._parse_with(self._ocr_converter, path)
            parsed.low_confidence = True

        return parsed

    def _parse_with(self, converter: DocumentConverter, path: Path) -> ParsedDocument:
        doc = converter.convert(str(path)).document

        blocks: list[Block] = []
        pages: set[int] = set()

        for item, _level in doc.iterate_items():
            raw_is_table = type(item).__name__.lower().startswith("table")

            if raw_is_table:
                # Table items carry no `.text` — their content lives in a
                # structured grid that must be exported explicitly. Without
                # this, every table (Excel sheets, PDF tables) is silently
                # dropped and never indexed.
                try:
                    text = item.export_to_markdown(doc)
                except Exception:
                    text = ""
                sheet = _sheet_name(item, doc)
                df = _export_table_df(item, doc)
                is_table = _looks_like_a_table(df)
            else:
                text = getattr(item, "text", "") or ""
                sheet = None
                is_table = False
                df = None

            if not text.strip():
                continue

            page = None
            prov = getattr(item, "prov", None)
            if prov and isinstance(prov, (list, tuple)):
                page = getattr(prov[0], "page_no", None)

            # For spreadsheet tables the "page number" is just the sheet
            # ordinal — the sheet name is the meaningful locator, so drop the
            # synthetic page and let citations read "file.xlsx, sheet Q1".
            if sheet is not None:
                page = None
            elif page is not None:
                pages.add(page)

            blocks.append(Block(
                text=text,
                page=page,
                is_table=is_table,
                sheet=sheet,
            ))

            # For each table, also emit a precomputed aggregate summary as
            # its own block. Retrieval can then surface a ready "total = X"
            # fact for aggregation questions instead of asking the LLM to add
            # up rows it may only partially see. is_table=False so it reads as
            # a prose fact, not a table fragment.
            if is_table:
                summary = _table_summary(df, sheet)
                if summary:
                    blocks.append(Block(
                        text=summary, page=page, sheet=sheet,
                        is_table=False, is_summary=True,
                    ))

        return ParsedDocument(
            markdown=doc.export_to_markdown(),
            blocks=blocks,
            page_count=len(pages) or 1,
        )
