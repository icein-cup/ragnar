import re

from core.models import SearchResult

# Words that mean the question ranges over a SET of rows, not one cell.
# That distinction is the whole job here. This list used to also hold
# "how many", "how much", "count", "rank", "ranking" and the superlatives,
# which matched 29 of the 125 benchmark questions while only about one of
# them was a real aggregation: "how many stores does Netto have" is a single
# cell, and refusing it costs a correct answer for nothing.
#
# Scored against eval/golden_aggregation.yaml (8 real aggregations, 8
# single-cell lookups, all phrased with aggregation vocabulary): 8/8 caught,
# 0/8 lookups wrongly caught. The old list caught 16/16.
AGGREGATION_TERMS = {
    # English
    "total", "sum", "average", "mean", "aggregate", "overall", "combined",
    "altogether", "all",
    # Polish
    "suma", "sumy", "razem", "łącznie", "lacznie", "średnia", "srednia",
    "łączna", "laczna", "ogółem", "ogolem", "wszystkie", "wszystkich",
}

# Partitive phrasings: "how many OF the listed airports", "how many DIFFERENT
# chains". The "of"/"different" is what turns a count into a count over a
# set — "how many stores does Netto have" has neither.
AGGREGATION_PATTERNS = (
    r"how man(?:y|ies) (?:of|different)\b",
    r"how much of\b",
    r"ile (?:z|spośród|sposrod)\b",
)

# DELIBERATELY NOT HERE: the superlatives (highest, lowest, largest,
# smallest, top, most, and the Polish największ-/najwyższ- stems).
#
# They are genuinely ambiguous and the evidence does not settle them. Over a
# table split across many chunks, "which airport has the most movements" is
# unsafe for the same reason a sum is — the model sees a fragment and reports
# the maximum of that fragment. Over a rank-sorted table it is a plain lookup
# of row 1.
#
# Including them costs 3 of the 8 lookups in the probe set, which is measured;
# the benefit is hypothetical, because the guard only runs at all when
# retrieval is table-majority. Left out for now, with the trade-off written
# down rather than decided silently. The sharper rule, if this ever needs
# settling: fire on a superlative only when several chunks of the SAME table
# were retrieved, which is exactly the fragment case.
#
# If they are ever restored, the Polish ones must go back as STEMS with a
# leading boundary only — "największ" and not r"\bnajwiększ\b" — because
# those superlatives inflect for gender and case ("największa kwota",
# "najwyższy przychód") and a trailing \b cannot match when the next
# character is itself a word character. That was a real bug once.

TABLE_MAJORITY = 0.5

# Single-word terms match on both boundaries ("total" but not "totally");
# multi-word terms are plain substrings. Both derive from the set above,
# which stays the one place to edit the vocabulary.
_WORD_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(
        re.escape(t) for t in sorted(AGGREGATION_TERMS) if " " not in t
    )
)
_PATTERN_RE = re.compile("|".join(AGGREGATION_PATTERNS))
_PHRASES = tuple(sorted(t for t in AGGREGATION_TERMS if " " in t))


def _has_aggregation_intent(question: str) -> bool:
    lowered = question.lower()
    return (
        bool(_WORD_RE.search(lowered))
        or bool(_PATTERN_RE.search(lowered))
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


# The sentinel SYSTEM_PROMPT tells the model to lead a non-answer with. This
# is the primary refusal signal: it is a contract we set, so it does not
# change when the model does. REFUSAL_PATTERNS below stays as a fallback for
# a model that ignores the instruction.
NO_ANSWER = "NO_ANSWER"

# Phrases a model reaches for when the excerpts do not cover the question.
# English and Polish, matching AGGREGATION_TERMS' bilingual scope. These were
# harvested from qwen2.5's output, so they describe one model's habits rather
# than refusal in general - which is exactly why NO_ANSWER above exists.
REFUSAL_PATTERNS = (
    # English
    r"do(?:es)? not (?:contain|provide|specify|mention|state|include)",
    r"not (?:provided|specified|mentioned|available|found|listed)",
    r"no information",
    r"there is no ",
    r"cannot be (?:determined|answered|found)",
    r"does not appear",
    # Polish
    r"nie zawiera",
    r"nie podano",
    r"nie ma informacji",
    r"brak informacji",
    r"nie zosta[łl]o (?:podane|okre[śs]lone)",
)

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")
# Punctuation/quoting a model wraps the sentinel in at the very end of a
# reply ("...Therefore, NO_ANSWER.", '..."NO_ANSWER")'). Stripped before the
# suffix check below, never removed from the middle of the text. Includes a
# literal space (a model's trailing whitespace before punctuation) — easy to
# miss reading the string, so spelled out here rather than left implicit.
_TRAILING_WRAP = ".!?\"')] \n\t"


def _first_sentence(text: str) -> str:
    text = text.strip()
    match = _SENTENCE_END_RE.search(text)
    return text[:match.start()] if match else text


def _ends_with_no_answer(text: str) -> bool:
    return text.rstrip(_TRAILING_WRAP).endswith(NO_ANSWER)


def is_refusal(text: str) -> bool:
    """Did the model say the excerpts do not answer the question?

    The retrieval-side refusal in ``classify`` only fires when nothing
    cleared the floors, which on a large corpus is almost never — so the
    model writes "the excerpts do not contain that" and the app still marks
    it an answer and staples five citations to it.

    The NO_ANSWER sentinel is checked first: as a PREFIX (SYSTEM_PROMPT asks
    for it to lead the reply) or a SUFFIX (a model reasoning its way to "no
    answer" sometimes states that conclusion at the end instead — "...the
    second excerpt does not provide his birth date. Therefore, NO_ANSWER.").
    Measured: 4 of 90 HybridQA answers did exactly this and were scored as
    wrong answers rather than refusals, because a leading-only check missed
    them.

    Anchored to prefix/suffix rather than "anywhere" on purpose: an answer
    that legitimately discusses the token as data ("The build flag is called
    NO_ANSWER in the config.") has real content on both sides of it within
    the same sentence, and must not be swept up. The sentinel is trusted
    wherever it leads or trails because it is machinery we mandate, not
    prose we are guessing at — that trust does not extend to the middle of
    an otherwise ordinary sentence.

    The phrase regex is a fallback for a model that ignores the instruction
    entirely, and it only recognises the two languages someone thought to
    list. For that fallback ONLY, just the FIRST sentence is examined — an
    answer that states a fact and then caveats a missing detail ("Larry
    McMurtry wrote it. The excerpts do not give the year.") is an answer, not
    a refusal, and matching the whole text would throw it away along with its
    citations.
    """
    lstripped = text.lstrip()
    if lstripped.startswith(NO_ANSWER) or _ends_with_no_answer(text):
        return True
    return bool(_REFUSAL_RE.search(_first_sentence(text)))


# What to show when the model gives the sentinel and nothing else.
# Qwen3.8-27 does exactly that: its whole reply is "NO_ANSWER", so stripping
# the token leaves an empty string and the answer renders as blank.
NO_ANSWER_MESSAGE = (
    "The retrieved excerpts do not contain the answer to that question."
)


def refusal_text(text: str) -> str:
    """Reader-facing text for a refusal: token removed, never empty.

    Use this wherever a refusal is shown or stored. strip_no_answer alone is
    not enough — a model that replies with the bare sentinel would leave
    nothing on screen.
    """
    return strip_no_answer(text).strip() or NO_ANSWER_MESSAGE


def strip_no_answer(text: str) -> str:
    """Remove the NO_ANSWER sentinel, leaving the explanation.

    The token is machinery, not prose. Call ``is_refusal`` before this, not
    after — stripping is what makes the sentinel invisible to it.

    Leading form: strip the token, keep the rest — the usual case.

    Trailing form ("...does not provide his birth date. Therefore,
    NO_ANSWER."): the final sentence is dropped whole rather than leaving a
    dangling connector like "Therefore, ." — everything up to the previous
    sentence boundary is kept, since that is usually a complete thought on
    its own. See is_refusal for why this is anchored to prefix/suffix and
    not a freeform search.
    """
    stripped = text.lstrip()
    if stripped.startswith(NO_ANSWER):
        return stripped[len(NO_ANSWER):].lstrip()

    if _ends_with_no_answer(text):
        rstripped = text.rstrip()
        matches = list(_SENTENCE_END_RE.finditer(rstripped))
        if matches:
            return rstripped[:matches[-1].end()].rstrip()
        return ""  # the sentinel was the only sentence

    return text
