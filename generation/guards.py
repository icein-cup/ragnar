import re

from core.models import SearchResult

# English and Polish aggregation intent markers. Complete words only —
# anything that needs to match inflected endings belongs in
# AGGREGATION_STEMS below instead.
AGGREGATION_TERMS = {
    # English
    "total", "sum", "average", "mean", "count", "how many", "how much",
    "highest", "lowest", "largest", "smallest", "top", "rank", "ranking",
    "aggregate", "overall", "combined", "altogether",
    # Polish
    "suma", "sumy", "razem", "łącznie", "lacznie", "ile", "średnia",
    "srednia", "ranking", "łączna", "laczna", "ogółem", "ogolem",
}

# Polish superlatives inflect for gender and case ("największa kwota",
# "najwyższy przychód"), so only the stem is stable. These must NOT get a
# trailing word boundary — `\bnajwiększ\b` cannot match "największa", since
# the following letter is itself a word character. That is precisely the bug
# these were split out to fix.
AGGREGATION_STEMS = {
    "największ", "najwieksz", "najmniejsz", "najwyższ", "najwyzsz",
}

TABLE_MAJORITY = 0.5

# Single-word terms match on both boundaries ("total" but not "totally");
# stems match on the leading boundary only, so inflected endings still hit;
# multi-word terms are plain substrings. All three are derived from the sets
# above, which stay the one place to edit the vocabulary.
_WORD_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(
        re.escape(t) for t in sorted(AGGREGATION_TERMS) if " " not in t
    )
)
_STEM_RE = re.compile(
    r"\b(?:%s)" % "|".join(re.escape(t) for t in sorted(AGGREGATION_STEMS))
)
_PHRASES = tuple(sorted(t for t in AGGREGATION_TERMS if " " in t))


def _has_aggregation_intent(question: str) -> bool:
    lowered = question.lower()
    return (
        bool(_WORD_RE.search(lowered))
        or bool(_STEM_RE.search(lowered))
        or any(p in lowered for p in _PHRASES)
    )


def _is_table_heavy(results: list[SearchResult]) -> bool:
    if not results:
        return False
    tables = sum(1 for r in results if r.chunk.is_table)
    return tables / len(results) > TABLE_MAJORITY


def should_refuse_aggregation(question: str,
                               results: list[SearchResult]) -> bool:
    """Fires only when BOTH signals are present.

    Either signal alone produces false positives: a question containing
    "total" about a prose contract must not be blocked, and a lookup
    question over a table must not be blocked either.

    Defers entirely when a precomputed aggregate summary was retrieved —
    the LLM can read a ready, deterministically-correct total from it, so
    there is nothing to refuse.
    """
    if any(r.chunk.is_summary for r in results):
        return False
    return _has_aggregation_intent(question) and _is_table_heavy(results)


def aggregation_refusal(results: list[SearchResult]) -> str:
    sources = []
    for r in results:
        label = r.chunk.filename
        if r.chunk.sheet:
            label = f"{label} (sheet {r.chunk.sheet})"
        if label not in sources:
            sources.append(label)

    listed = ", ".join(sources)
    return (
        "This looks like a question that requires calculating across a whole "
        "table. I can only read individual rows, so any total I gave you "
        "could be wrong.\n\n"
        f"The relevant data is in: {listed}"
    )
