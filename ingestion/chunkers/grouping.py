from ingestion.parser import Block


def group_blocks(blocks: list[Block], max_chars: int | None) -> list[list[Block]]:
    """Groups adjacent blocks that belong in the same chunk.

    Shared by every text-based chunker so the provenance invariants live in
    one place: tables and precomputed table summaries are always isolated
    into their own group, a group never crosses a page boundary, and (when
    `max_chars` is given) a group is flushed before it would exceed that
    size. Passing `max_chars=None` skips the size-based flush entirely —
    used by SemanticChunker, which wants a whole page of prose in one group
    so topic boundaries, not character counts, decide where it gets cut.

    Summary blocks are isolated for the same reason tables are, but the
    consequence is sharper: the chunkers take their flags from `group[0]`,
    so a summary merged with trailing prose would stamp `is_summary=True`
    onto that prose — and `should_refuse_aggregation` defers entirely when
    any retrieved chunk carries that flag, silently disabling the
    aggregation guard on hits unrelated to the table it was computed from.
    """
    groups: list[list[Block]] = []
    current: list[Block] = []

    def flush():
        if current:
            groups.append(list(current))
            current.clear()

    for block in blocks:
        if not block.text.strip():
            continue

        if block.is_table or block.is_summary:
            flush()
            groups.append([block])
            continue

        if current:
            same_page = current[-1].page == block.page
            oversized = (
                max_chars is not None
                and sum(len(b.text) for b in current) + len(block.text) > max_chars
            )
            if not same_page or oversized:
                flush()

        current.append(block)

    flush()
    return groups
