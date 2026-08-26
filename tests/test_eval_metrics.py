import pytest

from eval.metrics import (answer_accuracy, refusal_accuracy, citation_accuracy,
                          citation_precision, multi_hop_citation_accuracy)


def test_refusal_accuracy_rewards_correct_refusals():
    cases = [
        {"out_of_corpus": True, "refused": True},
        {"out_of_corpus": False, "refused": False},
    ]
    assert refusal_accuracy(cases) == 1.0


def test_refusal_accuracy_penalises_answering_out_of_corpus():
    cases = [{"out_of_corpus": True, "refused": False}]
    assert refusal_accuracy(cases) == 0.0


def test_refusal_accuracy_penalises_refusing_in_corpus():
    cases = [{"out_of_corpus": False, "refused": True}]
    assert refusal_accuracy(cases) == 0.0


def test_citation_accuracy_matches_expected_source():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["report.pdf, p. 4"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 1.0


def test_citation_accuracy_fails_on_wrong_source():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["other.pdf, p. 1"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 0.0


def test_citation_accuracy_skips_out_of_corpus_cases():
    cases = [
        {"expected_sources": [], "citations": [], "out_of_corpus": True},
        {"expected_sources": ["a.pdf"], "citations": ["a.pdf, p. 1"],
         "out_of_corpus": False},
    ]
    assert citation_accuracy(cases) == 1.0


def test_citation_accuracy_rejects_substring_filename_match():
    cases = [{
        "expected_sources": ["report.pdf"],
        "citations": ["quarterly_report.pdf, p. 4"],
        "out_of_corpus": False,
    }]
    assert citation_accuracy(cases) == 0.0


def test_multi_hop_citation_accuracy_requires_all_sources():
    cases = [{
        "expected_sources": ["table A", "passage B"],
        "citations": ["table A, p. 1", "passage B, p. 2"],
        "out_of_corpus": False,
        "multihop": True,
    }]
    assert multi_hop_citation_accuracy(cases) == 1.0


def test_multi_hop_citation_accuracy_fails_when_a_source_missing():
    cases = [{
        "expected_sources": ["table A", "passage B"],
        "citations": ["table A, p. 1"],
        "out_of_corpus": False,
        "multihop": True,
    }]
    assert multi_hop_citation_accuracy(cases) == 0.0


def test_multi_hop_citation_accuracy_skips_non_multihop_and_ooc():
    cases = [
        {"expected_sources": ["a.pdf"], "citations": ["a.pdf, p. 1"],
         "out_of_corpus": False, "multihop": False},
        {"expected_sources": [], "citations": [], "out_of_corpus": True},
        {"expected_sources": ["t", "p"], "citations": ["t", "p"],
         "out_of_corpus": False, "multihop": True},
    ]
    assert multi_hop_citation_accuracy(cases) == 1.0


def test_multi_hop_citation_accuracy_accepts_sub_article_citation():
    cases = [{
        "expected_sources": ["Alpine skiing at the 1988 Winter Olympics"],
        "citations": ["Alpine skiing at the 1988 Winter Olympics – Men's super-G"],
        "out_of_corpus": False,
        "multihop": True,
    }]
    assert multi_hop_citation_accuracy(cases) == 1.0


def test_citation_precision_penalises_citing_everything_retrieved():
    cases = [{
        "expected_sources": ["Multan"],
        "citations": ["Multan", "Hong Kong", "Memphis, Tennessee"],
        "out_of_corpus": False,
    }]
    assert citation_precision(cases) == pytest.approx(1 / 3)


def test_citation_precision_is_one_when_only_expected_sources_are_cited():
    cases = [{
        "expected_sources": ["table A", "passage B"],
        "citations": ["table A", "passage B"],
        "out_of_corpus": False,
    }]
    assert citation_precision(cases) == 1.0


def test_citation_precision_skips_cases_with_no_citations():
    cases = [
        {"expected_sources": ["a.pdf"], "citations": [], "out_of_corpus": False},
        {"expected_sources": ["a.pdf"], "citations": ["a.pdf"], "out_of_corpus": False},
    ]
    assert citation_precision(cases) == 1.0


def test_answer_accuracy_requires_the_expected_answer_in_the_text():
    cases = [{
        "expected_answer": "the Chenab River",
        "answer": "Multan sits on the banks of the Chenab River.",
        "refused": False, "out_of_corpus": False,
    }]
    assert answer_accuracy(cases) == 1.0


def test_answer_accuracy_fails_a_confidently_wrong_answer():
    # The failure citation_accuracy cannot see: right document, wrong fact.
    cases = [{
        "expected_answer": "Sadie Robertson",
        "answer": "Kel Mitchell came in second to Alfonso Ribeiro.",
        "refused": False, "out_of_corpus": False,
    }]
    assert answer_accuracy(cases) == 0.0


def test_answer_accuracy_skips_refusals_and_out_of_corpus():
    cases = [
        {"expected_answer": "", "answer": "", "refused": True, "out_of_corpus": True},
        {"expected_answer": "x", "answer": "", "refused": True, "out_of_corpus": False},
        {"expected_answer": "SC-4471", "answer": "It is SC-4471.",
         "refused": False, "out_of_corpus": False},
    ]
    assert answer_accuracy(cases) == 1.0


# ── run_eval.resolve_answer ──────────────────────────────────────────────────
# Self-correction already generates a full answer; the UI shows that draft
# directly. The eval used to throw it away and regenerate, paying for the most
# expensive call in the pipeline twice and measuring a path no user takes.

from eval.run_eval import resolve_answer
from generation.answerer import AnswerMode, Answer


class _CountingAnswerer:
    def __init__(self, text="regenerated"):
        self.text = text
        self.calls = 0
        self.kwargs = []

    def answer(self, question, results, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        return Answer(text=self.text, citations=["a.pdf, p. 1"])


class _Outcome:
    def __init__(self, draft=None, results=()):
        self.draft_answer = draft
        self.results = list(results)


def _chunk_result(filename="a.pdf", page=1, text="Kiwi is a fast-food chain."):
    from core.models import Chunk, SearchResult
    return SearchResult(
        chunk=Chunk(doc_id="d", filename=filename, text=text,
                    chunk_index=0, page=page),
        score=0.7,
    )


def test_a_draft_is_reused_instead_of_regenerating_the_answer():
    answerer = _CountingAnswerer()
    outcome = _Outcome(draft="Kiwi is the chain.", results=[_chunk_result()])

    text, citations, refused, reused = resolve_answer(
        "q?", outcome, AnswerMode.ANSWER, answerer)

    assert answerer.calls == 0, "the draft made a second generation unnecessary"
    assert (text, refused, reused) == ("Kiwi is the chain.", False, True)
    assert citations == ["a.pdf, p. 1"]


def test_no_draft_falls_back_to_generating_the_answer():
    answerer = _CountingAnswerer()
    outcome = _Outcome(draft=None, results=[_chunk_result()])

    text, citations, refused, reused = resolve_answer(
        "q?", outcome, AnswerMode.ANSWER, answerer)

    assert answerer.calls == 1
    assert (text, reused) == ("regenerated", False)
    assert citations == ["a.pdf, p. 1"]


def test_a_refusing_draft_keeps_its_citations_off_and_hides_the_sentinel():
    answerer = _CountingAnswerer()
    outcome = _Outcome(draft="NO_ANSWER The excerpts stop at 2019.",
                       results=[_chunk_result()])

    text, citations, refused, reused = resolve_answer(
        "q?", outcome, AnswerMode.ANSWER, answerer)

    assert answerer.calls == 0
    assert refused is True
    assert citations == [], "a non-answer must not read as a sourced one"
    assert text == "The excerpts stop at 2019."


def test_a_guard_refusal_ignores_any_draft():
    answerer = _CountingAnswerer()
    outcome = _Outcome(draft="a draft that must not be used",
                       results=[_chunk_result()])

    text, citations, refused, reused = resolve_answer(
        "q?", outcome, AnswerMode.AGGREGATION_REFUSED, answerer)

    assert (text, citations, refused, reused) == ("", [], True, False)
    assert answerer.calls == 0


# answer_accuracy drops refusals from its denominator, so a model that refuses
# nearly everything scores near 1.00 on it. answer_coverage is the number that
# catches that, and the two must always be read together.

from eval.metrics import answer_coverage


def _case(expected, answer, refused=False, ooc=False):
    return {"expected_answer": expected, "answer": answer,
            "refused": refused, "out_of_corpus": ooc}


def test_accuracy_and_coverage_diverge_on_a_model_that_refuses_everything():
    cases = [_case("Karachi", "Karachi is the largest."),
             _case("Lahore", "Lahore.")]
    cases += [_case("x", "", refused=True) for _ in range(8)]

    assert answer_accuracy(cases) == 1.0, "answered 2, both right"
    assert answer_coverage(cases) == 0.2, "but only 2 of 10 questions answered"


def test_they_agree_when_nothing_is_refused():
    cases = [_case("Karachi", "Karachi."), _case("Lahore", "Something else.")]
    assert answer_accuracy(cases) == answer_coverage(cases) == 0.5


def test_out_of_corpus_questions_are_outside_both():
    cases = [_case("Karachi", "Karachi."),
             _case("", "", refused=True, ooc=True)]
    assert answer_accuracy(cases) == 1.0
    assert answer_coverage(cases) == 1.0


def test_a_refusal_never_counts_as_correct_even_if_it_echoes_the_answer():
    cases = [_case("Three", "NO_ANSWER The excerpts list three tables but "
                            "not the figure asked for.", refused=True)]
    assert answer_coverage(cases) == 0.0


def test_the_answer_temperature_reaches_the_generation_call():
    """A temperature sweep is meaningless if the value never leaves argparse.
    None must still be passed through — the LLM client's own default (0.0)
    is what it means, and forcing a number here would hide that."""
    answerer = _CountingAnswerer()
    outcome = _Outcome(draft=None, results=[_chunk_result()])

    resolve_answer("q?", outcome, AnswerMode.ANSWER, answerer, 0.2)
    resolve_answer("q?", outcome, AnswerMode.ANSWER, answerer)

    assert answerer.kwargs == [{"temperature": 0.2}, {"temperature": None}]
