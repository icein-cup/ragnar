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
