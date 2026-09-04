import pytest

from ingestion.parser import Block, ParsedDocument
from ingestion.chunkers.registry import build_chunker
from ingestion.chunkers.semantic import SemanticChunker
from tests.fakes import ScriptedEmbedder


def _doc(blocks):
    return ParsedDocument(markdown="x", blocks=blocks, page_count=1)


def test_cuts_at_the_similarity_drop():
    # Three sentences about cats, three about stocks — an obvious cosine
    # cliff between the two topics, nothing but noise within each.
    cat_sentences = [
        "Cats are great pets.",
        "Cats like to sleep.",
        "Cats often purr.",
    ]
    stock_sentences = [
        "Stocks rose sharply today.",
        "The market gained value.",
        "Investors were pleased.",
    ]
    vectors = {s: [1.0, 0.0] for s in cat_sentences}
    vectors.update({s: [0.0, 1.0] for s in stock_sentences})
    embedder = ScriptedEmbedder(vectors)

    text = " ".join(cat_sentences + stock_sentences)
    doc = _doc([Block(text=text, page=1)])
    chunks = SemanticChunker(embedder).chunk(doc, "d", "f.txt")

    assert len(chunks) == 2
    assert "Cats" in chunks[0].text and "Stocks" not in chunks[0].text
    assert "Stocks" in chunks[1].text and "Cats" not in chunks[1].text


def test_page_boundary_wins_over_identical_vectors():
    embedder = ScriptedEmbedder({
        "Sentence A.": [1.0, 0.0],
        "Sentence B.": [1.0, 0.0],
    })
    doc = _doc([
        Block(text="Sentence A.", page=1),
        Block(text="Sentence B.", page=2),
    ])
    chunks = SemanticChunker(embedder).chunk(doc, "d", "f.txt")

    assert len(chunks) == 2
    for c in chunks:
        assert not ("Sentence A" in c.text and "Sentence B" in c.text)


def test_ceiling_is_enforced_even_with_no_semantic_reason_to_cut():
    sentences = [
        "Alpha bravo charlie delta.",
        "Echo foxtrot golf hotel.",
        "India juliet kilo lima.",
        "Mike november oscar papa.",
        "Quebec romeo sierra tango.",
    ]
    # Identical vectors: zero distance everywhere, nothing for the
    # similarity threshold to catch — only the size ceiling can split this.
    embedder = ScriptedEmbedder({s: [1.0, 0.0, 0.0] for s in sentences})
    doc = _doc([Block(text=" ".join(sentences), page=1)])
    chunker = SemanticChunker(embedder, target_tokens=20)
    chunks = chunker.chunk(doc, "d", "f.txt")

    assert len(chunks) > 1
    assert all(len(c.text) <= chunker.target_chars for c in chunks)


def test_tables_bypass_semantic_splitting():
    table_text = (
        "| Client | Region | Value |\n"
        "|---|---|---|\n"
        "| A | North | 1000 |\n"
        "| B | South | 2000 |\n"
    )
    doc = _doc([
        Block(text="Intro prose.", page=1),
        Block(text=table_text, page=1, is_table=True),
    ])
    embedder = ScriptedEmbedder({"Intro prose.": [1.0, 0.0]})
    chunks = SemanticChunker(embedder).chunk(doc, "d", "f.pdf")

    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) == 1
    assert "Intro prose" not in table_chunks[0].text
    # The table's own content never went through embed() / sentence splitting.
    assert not any("Client" in call_text
                   for call in embedder.calls for call_text in call)


def test_sheet_name_is_prepended_to_table_chunk_text():
    doc = _doc([
        Block(text="| Client | Value |\n|---|---|\n| Acme | 1000 |",
              page=None, is_table=True, sheet="Q1 Sales"),
    ])
    embedder = ScriptedEmbedder({})
    chunks = SemanticChunker(embedder).chunk(doc, "d", "sales.xlsx")

    assert chunks
    assert "Sheet: Q1 Sales" in chunks[0].text
    assert "Acme" in chunks[0].text
    assert chunks[0].sheet == "Q1 Sales"


def test_single_sentence_document_does_not_crash():
    embedder = ScriptedEmbedder({"Only one sentence here.": [1.0, 0.0]})
    doc = _doc([Block(text="Only one sentence here.", page=1)])
    chunks = SemanticChunker(embedder).chunk(doc, "d", "f.txt")

    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."


def test_short_embedder_response_raises_instead_of_crashing_on_stopiteration():
    """A truncated/deduped embed response used to blow up as a bare
    StopIteration deep inside a list comprehension. Must fail loud and
    clear instead."""
    class ShortEmbedder:
        def embed(self, texts):
            return [[1.0, 0.0]] * (len(texts) - 1)  # one short

    doc = _doc([Block(text="One. Two. Three.", page=1)])

    with pytest.raises(RuntimeError, match="vectors"):
        SemanticChunker(ShortEmbedder()).chunk(doc, "d", "f.txt")


def test_one_embed_call_per_document():
    sentences = ["Alpha one.", "Alpha two.", "Beta one.", "Beta two."]
    embedder = ScriptedEmbedder({s: [1.0, 0.0] for s in sentences})
    doc = _doc([
        Block(text="Alpha one. Alpha two.", page=1),
        Block(text="Beta one. Beta two.", page=2),
    ])
    SemanticChunker(embedder).chunk(doc, "d", "f.txt")

    assert len(embedder.calls) == 1
    assert embedder.calls[0] == sentences


def test_build_chunker_semantic_without_embedder_raises():
    with pytest.raises(ValueError, match="embedder"):
        build_chunker({"strategy": "semantic"})


def test_build_chunker_semantic_with_embedder_applies_config():
    embedder = ScriptedEmbedder({})
    chunker = build_chunker(
        {"strategy": "semantic", "target_tokens": 300,
         "semantic_breakpoint_percentile": 80},
        embedder=embedder,
    )
    assert isinstance(chunker, SemanticChunker)
    assert chunker.target_chars == int(300 * 3.5)
    assert chunker.breakpoint_percentile == 80
