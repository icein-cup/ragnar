import pytest

from ingestion.parser import Block, ParsedDocument
from ingestion.chunkers.structural import StructuralChunker
from ingestion.chunkers.registry import build_chunker


def _doc(blocks):
    return ParsedDocument(markdown="x", blocks=blocks, page_count=1)


def test_chunks_do_not_span_page_boundaries():
    doc = _doc([
        Block(text="Short A.", page=1),
        Block(text="Short B.", page=2),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    for c in chunks:
        assert not ("Short A" in c.text and "Short B" in c.text)


def test_small_adjacent_blocks_on_same_page_are_merged():
    doc = _doc([
        Block(text="First sentence.", page=1),
        Block(text="Second sentence.", page=1),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    assert len(chunks) == 1
    assert "First sentence." in chunks[0].text
    assert "Second sentence." in chunks[0].text


def test_oversized_block_is_split():
    doc = _doc([Block(text="word " * 4000, page=1)])
    chunks = StructuralChunker(target_tokens=100).chunk(doc, "d", "f.pdf")
    assert len(chunks) > 1


def test_table_blocks_are_never_merged_with_prose():
    doc = _doc([
        Block(text="Intro prose.", page=1),
        Block(text="| a | b |", page=1, is_table=True),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "Intro prose" not in table_chunks[0].text


def test_table_rows_per_group_config_is_applied():
    table_text = (
        "| Client | Region | Value |\n"
        "|---|---|---|\n"
        "| A | North | 1000 |\n"
        "| B | South | 2000 |\n"
        "| C | East | 3000 |\n"
        "| D | West | 4000 |"
    )
    doc = _doc([Block(text=table_text, page=1, is_table=True)])
    chunker = build_chunker({"strategy": "structural", "table_rows_per_group": 2})
    chunks = chunker.chunk(doc, "d", "f.pdf")

    # With rows_per_group=2 and 4 data rows, we expect 2 table chunks
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.is_table
        assert "| Client | Region | Value |" in chunk.text


def test_chunk_indices_are_sequential():
    doc = _doc([Block(text=f"Block {i}.", page=i) for i in range(1, 6)])
    chunks = StructuralChunker().chunk(doc, "d", "f.pdf")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_build_chunker_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown chunker"):
        build_chunker({"strategy": "bogus"})


def test_build_chunker_structural_applies_config():
    chunker = build_chunker({"strategy": "structural", "target_tokens": 300})
    assert isinstance(chunker, StructuralChunker)
    assert chunker.target_chars == int(300 * 3.5)


def test_low_confidence_propagates_to_chunks():
    doc = ParsedDocument(
        markdown="x",
        blocks=[
            Block(text="First sentence.", page=1),
            Block(text="Second sentence.", page=2),
        ],
        page_count=2,
        low_confidence=True,
    )
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    assert chunks
    assert all(c.low_confidence for c in chunks)


def test_sheet_name_is_prepended_to_table_chunk_text():
    doc = _doc([
        Block(text="| Client | Value |\n|---|---|\n| Acme | 1000 |",
              page=None, is_table=True, sheet="Q1 Sales"),
    ])
    chunks = StructuralChunker().chunk(doc, "d", "sales.xlsx")

    assert chunks
    # Sheet name searchable in the embedded text, and the table still intact.
    assert "Sheet: Q1 Sales" in chunks[0].text
    assert "Acme" in chunks[0].text
    assert chunks[0].sheet == "Q1 Sales"


def test_table_summary_is_isolated_from_following_prose():
    """A summary block merged with prose stamps is_summary=True onto that
    prose, and should_refuse_aggregation defers entirely on any is_summary
    hit — so the aggregation guard would switch itself off on unrelated
    retrievals."""
    doc = _doc([
        Block(text="| a | 1 |\n|---|---|\n| b | 2 |", page=3, is_table=True),
        Block(text="Aggregate column summary: 2 rows total.", page=3,
              is_summary=True),
        Block(text="Ordinary prose about warranty terms.", page=3),
    ])
    chunks = StructuralChunker(target_tokens=500).chunk(doc, "d", "f.pdf")

    summaries = [c for c in chunks if c.is_summary]
    assert len(summaries) == 1
    assert "warranty" not in summaries[0].text
    assert any("warranty" in c.text and not c.is_summary and not c.is_table
               for c in chunks)
