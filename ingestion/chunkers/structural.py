from core.models import Chunk
from ingestion.chunkers.grouping import group_blocks
from ingestion.parser import ParsedDocument

# Rough tokens-per-character for mixed PL/EN text. English-centric BPE
# tokenizers compress Polish less efficiently than English, so Polish text
# yields fewer chars-per-token than English's typical ~4.
CHARS_PER_TOKEN = 3.5


class StructuralChunker:
    """Merges adjacent blocks up to a token budget, never crossing pages,
    never mixing tables with prose."""

    def __init__(self, target_tokens: int = 500, overlap_tokens: int = 50,
                 rows_per_group: int = 20):
        self.target_chars = int(target_tokens * CHARS_PER_TOKEN)
        self.overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)
        self.rows_per_group = rows_per_group

    def chunk(self, parsed: ParsedDocument, doc_id: str,
              filename: str) -> list[Chunk]:
        groups = group_blocks(parsed.blocks, max_chars=self.target_chars)

        chunks: list[Chunk] = []
        index = 0
        for group in groups:
            text = "\n\n".join(b.text.strip() for b in group)
            head = group[0]

            if head.is_table:
                from ingestion.tables import chunk_table_markdown
                pieces = chunk_table_markdown(text, rows_per_group=self.rows_per_group) or [text]
                if head.sheet:
                    # Put the sheet name into the embedded text (not just the
                    # citation metadata) so questions that reference a sheet
                    # by name — "what's in the Q2 sheet?" — actually retrieve.
                    pieces = [f"Sheet: {head.sheet}\n\n{p}" for p in pieces]
            else:
                pieces = self._split(text)

            for piece in pieces:
                chunks.append(Chunk(
                    doc_id=doc_id,
                    filename=filename,
                    text=piece,
                    chunk_index=index,
                    page=head.page,
                    sheet=head.sheet,
                    is_table=head.is_table,
                    low_confidence=parsed.low_confidence,
                    is_summary=head.is_summary,
                ))
                index += 1

        return chunks

    def _split(self, text: str) -> list[str]:
        if len(text) <= self.target_chars:
            return [text]

        pieces = []
        start = 0
        step = max(self.target_chars - self.overlap_chars, 1)
        while start < len(text):
            pieces.append(text[start:start + self.target_chars])
            start += step
        return pieces
