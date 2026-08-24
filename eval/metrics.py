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


def _words(text: str) -> str:
    """Alphanumeric words only, space-joined.

    Punctuation must not decide correctness: HybridQA golden answers carry
    tokenizer spacing ("April 24 , 1898") that no model reproduces, and
    comparing raw strings marks those wrong for a comma.
    """
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def answer_accuracy(cases: list[dict]) -> float:
    """Did the answer actually state the expected answer?

    The metric this harness was missing. citation_accuracy asks whether the
    right DOCUMENT was cited, which a system can satisfy while misreading that
    document completely — measured at 96% cited vs 50% correct on the same run.
    Containment, not equality: the model writes a sentence around the fact.
    """
    scored = [c for c in cases if not c["out_of_corpus"] and not c["refused"]]
    if not scored:
        return 0.0
    correct = sum(1 for c in scored
                  if _words(c["expected_answer"]) in _words(c["answer"]))
    return correct / len(scored)


def citation_precision(cases: list[dict]) -> float:
    """Of the sources cited, what fraction were ones the question needed?

    citation_accuracy asks whether the right source appeared somewhere in the
    list; nothing there punishes a list that also carries four irrelevant
    documents, and citing everything retrieved is the cheapest way to pass it.
    This is the other half: an answer about Multan that cites Hong Kong,
    Memphis and Albany alongside the right table scores 0.25 here.
    """
    scored = [c for c in cases if not c["out_of_corpus"] and c["citations"]]
    if not scored:
        return 0.0
    per_case = [
        sum(1 for cited in c["citations"]
            if any(_label_matches(src, cited) for src in c["expected_sources"]))
        / len(c["citations"])
        for c in scored
    ]
    return sum(per_case) / len(per_case)


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
