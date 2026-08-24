from core.models import Chunk, SearchResult
import pytest

from generation.guards import (
    aggregation_refusal,
    is_refusal,
    should_refuse_aggregation,
)


def _result(text, is_table, filename="data.xlsx", sheet="Q1"):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename=filename, text=text, chunk_index=0,
                    is_table=is_table, sheet=sheet if is_table else None),
        score=0.9,
    )


def test_aggregation_question_over_tables_is_refused():
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation("What is the total revenue?", results)


def test_polish_aggregation_question_over_tables_is_refused():
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]
    assert should_refuse_aggregation("Jaka jest suma przychodów?", results)


def test_aggregation_wording_over_prose_is_NOT_refused():
    """The false-positive case the two-signal design exists to prevent."""
    results = [
        _result("The total liability is capped at 50,000 EUR.", False),
        _result("Payment terms are net 30.", False),
    ]
    assert not should_refuse_aggregation("What is the total liability?",
                                         results)


def test_lookup_question_over_tables_is_NOT_refused():
    results = [_result("| Acme | 1000 |", True)]
    assert not should_refuse_aggregation("What is Acme's value?", results)


def test_mixed_results_mostly_prose_are_NOT_refused():
    results = [
        _result("prose one", False),
        _result("prose two", False),
        _result("| a | 1 |", True),
    ]
    assert not should_refuse_aggregation("What is the total?", results)


def test_refusal_message_names_file_and_sheet():
    results = [_result("| a | 1 |", True, filename="sales.xlsx", sheet="Q1")]
    message = aggregation_refusal(results)

    assert "sales.xlsx" in message
    assert "Q1" in message


def test_empty_results_are_not_refused_by_this_guard():
    assert not should_refuse_aggregation("What is the total?", [])


def test_word_containing_aggregation_substring_is_NOT_refused():
    results = [_result("| Acme | 1000 |", True), _result("| Beta | 2000 |", True)]
    assert not should_refuse_aggregation("Is the laptop covered under warranty?", results)
    assert not should_refuse_aggregation("What is the discount policy?", results)


def test_polish_word_containing_aggregation_substring_is_NOT_refused():
    results = [_result("| Acme | 1000 |", True)]
    assert not should_refuse_aggregation("Gdzie mogę kupić bilet?", results)


def _summary(text="Aggregate column summary: total (sum): 6000"):
    return SearchResult(
        chunk=Chunk(doc_id="d", filename="f.xlsx", text=text, chunk_index=0,
                    is_summary=True, sheet="Q1"),
        score=0.9,
    )


def test_aggregation_defers_to_precomputed_summary():
    # aggregation intent + table results, but a summary is present -> do NOT refuse
    results = [_summary(), _result("| a | 1 |", True)]
    assert not should_refuse_aggregation("What is the total revenue?", results)


def test_polish_superlative_stems_match_their_inflected_forms():
    """The stems are prefixes; wrapping them in a trailing \\b made every
    inflected form miss, which is how Polish actually writes them."""
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]

    for question in (
        "Jaka jest największa kwota?",
        "Jaka jest najwyższa pensja?",
        "Kto ma najmniejszy przychód?",
        "Jaki jest najwyzszy koszt?",
        "Ktory dostawca ma najwiekszy udzial?",
    ):
        assert should_refuse_aggregation(question, results), question


def test_superlative_stems_do_not_fire_on_prose():
    results = [
        _result("The largest supplier is Acme.", False),
        _result("Payment terms are net 30.", False),
    ]
    assert not should_refuse_aggregation("Jaka jest największa kwota?", results)


@pytest.mark.parametrize("text", [
    "The provided excerpts do not contain any information about rainfall.",
    "The elevation of Wailuku, Hawaii is not provided in the given excerpts.",
    "There is no information provided about a guest appearing in 2018.",
    "The question cannot be answered with the provided information.",
    "Fragmenty nie zawierają informacji o tej umowie.",
    "Brak informacji na ten temat w dokumentach.",
])
def test_is_refusal_catches_the_models_own_refusal(text):
    assert is_refusal(text)


@pytest.mark.parametrize("text", [
    "Multan sits on the banks of the Chenab River.",
    # States the fact first, caveats a missing detail second — still an answer,
    # and matching the whole text would strip its citations.
    "Larry McMurtry was the author. However, the excerpts do not give the year.",
    "The hometown is Houston, Texas. The excerpts do not provide its area.",
])
def test_is_refusal_leaves_real_answers_alone(text):
    assert not is_refusal(text)


def test_is_refusal_reads_a_bare_value_then_a_denial_as_a_refusal():
    """A stray token with no sentence break before the denial stays inside the
    first sentence, so the denial wins. That is the safe reading: the model
    disowned the value it just emitted."""
    assert is_refusal("0\n\nThe excerpts do not provide this number.")
