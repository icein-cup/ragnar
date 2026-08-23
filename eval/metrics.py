import re


def refusal_accuracy(cases: list[dict]) -> float:
    """Did the system refuse exactly when it should have?

    Not a Ragas metric — deterministic, needs no judge, and covers the
    failure mode that matters most: confidently answering something that
    is not in the corpus.
    """
    if not cases:
        return 0.0
    correct = sum(
        1 for c in cases if bool(c["refused"]) == bool(c["out_of_corpus"])
    )
    return correct / len(cases)


# Characters that may sit between a source title and a sub-article suffix,
# e.g. "Alpine skiing at the 1988 Winter Olympics – Men's super-G".
_SEPARATORS = ("(", "–", "-", ",")


def _canon(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()


def _label_matches(source: str, cited: str) -> bool:
    """True when a cited label resolves to ``source`` or a sub-page of it.

    A citation like "Alpine skiing at the 1988 Winter Olympics – Men's
    super-G" is about the exact event named by source "Alpine skiing at the
    1988 Winter Olympics", just at a more specific page. Wikipedia tables
    link both the parent and its sub-articles, and retrieval can rank either;
    both are the correct evidence. Exact substring over-penalises the
    sub-article, so we accept a word-boundary prefix match as a hit.
    """
    s = _canon(source)
    c = _canon(cited)
    if s == c:
        return True
    if c.startswith(s):
        tail = c[len(s):]
        if not tail or tail[0].isspace() or tail[0] in _SEPARATORS:
            return True
    return False


def _sources_cited(case: dict, require_all: bool) -> bool:
    matches = [
        any(_label_matches(src, c) for c in case["citations"])
        for src in case["expected_sources"]
    ]
    return all(matches) if require_all else any(matches)


def citation_accuracy(cases: list[dict]) -> float:
    """Did the cited document match the expected source?"""
    scored = [c for c in cases if not c["out_of_corpus"]]
    if not scored:
        return 0.0
    correct = sum(1 for c in scored if _sources_cited(c, require_all=False))
    return correct / len(scored)


def multi_hop_citation_accuracy(cases: list[dict]) -> float:
    """Over multihop entries: did the answer cite EVERY expected source?

    The cross-document multi-hop signal. A single-hop answer can pass
    ``citation_accuracy`` by citing any one source; a multi-hop answer must
    surface all of them, so this is the free deterministic metric for the
    capability the benchmark was added to exercise.
    """
    scored = [c for c in cases
              if not c["out_of_corpus"] and c.get("multihop")]
    if not scored:
        return 0.0
    correct = sum(1 for c in scored if _sources_cited(c, require_all=True))
    return correct / len(scored)
