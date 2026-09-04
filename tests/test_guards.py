from core.models import Chunk, SearchResult
import pytest

from generation.guards import (
    NO_ANSWER,
    aggregation_refusal,
    is_refusal,
    refusal_text,
    should_refuse_aggregation,
    strip_no_answer,
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


def test_superlatives_no_longer_fire_the_guard_in_either_language():
    """Deliberate narrowing, symmetric across languages.

    "Which supplier has the largest share" is a lookup over a sorted table as
    often as it is a scan of an unsorted one, and the guard cannot tell which.
    Measured cost of keeping them: 3 of the 8 single-cell lookups in
    eval/golden_aggregation.yaml refused. See the note in guards.py, which
    also records how to restore the Polish stems correctly.
    """
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]

    for question in (
        "Jaka jest największa kwota?",
        "Jaka jest najwyższa pensja?",
        "Which supplier has the largest share?",
        "Which airport has the highest number of movements?",
    ):
        assert not should_refuse_aggregation(question, results), question


def test_polish_collective_terms_still_fire():
    """The narrowing dropped Polish superlatives, not Polish support."""
    results = [_result("| a | 1 |", True), _result("| b | 2 |", True)]

    for question in (
        "Jaka jest suma wszystkich kwot?",
        "Ile wynosi łącznie przychód?",
        "Jaka jest średnia pensja?",
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


# The NO_ANSWER sentinel — the refusal signal that does not depend on which
# model is answering, unlike REFUSAL_PATTERNS above.

def test_no_answer_sentinel_is_a_refusal_whatever_follows_it():
    # The point of the token: the explanation can be in any language, or use
    # wording no one thought to add to REFUSAL_PATTERNS, and it still counts.
    assert is_refusal(f"{NO_ANSWER} I could not locate that in the material.")
    assert is_refusal(f"{NO_ANSWER} Az információ nem szerepel a részletekben.")
    assert is_refusal(f"  {NO_ANSWER}\nNothing here covers the question.")


def test_strip_no_answer_removes_the_token_and_leaves_the_reason():
    assert strip_no_answer(
        f"{NO_ANSWER} The excerpts cover 2019, not 2020."
    ) == "The excerpts cover 2019, not 2020."


def test_strip_no_answer_leaves_a_real_answer_untouched():
    text = "Multan sits on the banks of the Chenab River."
    assert strip_no_answer(text) == text


def test_a_mid_answer_mention_of_the_token_is_not_a_refusal():
    # Only a leading or trailing token counts. Real content on both sides of
    # it within the same sentence means it's being discussed as data, not
    # emitted as the sentinel, so citations are kept.
    assert not is_refusal(f"The build flag is called {NO_ANSWER} in the config.")


# A model that reasons its way to "no answer" sometimes states that
# conclusion at the END instead of leading with it, per SYSTEM_PROMPT. These
# four are the actual cases that slipped through before the trailing check
# was added — measured against eval/reports/20260824-132940.json, where they
# were scored as wrong answers rather than refusals.

@pytest.mark.parametrize("text", [
    "Based on the information provided, the politician who took office in "
    "1934 and left in 1937 is Ernesto Ramos Antonini. The second excerpt "
    "does not provide his birth date. Therefore, NO_ANSWER.",
    "The location is named after Mecklenburg-Strelitz. However, the "
    "excerpts do not provide information on who owns this location. "
    "Therefore, NO_ANSWER.",
    "No actress in the excerpts has the same first and last name starting "
    "with the same letter.\n\nNO_ANSWER",
])
def test_a_trailing_sentinel_is_a_refusal(text):
    assert is_refusal(text)


def test_stripping_a_trailing_sentinel_drops_the_bare_declaration_sentence():
    text = ("The second excerpt does not provide his birth date. "
            "Therefore, NO_ANSWER.")
    stripped = strip_no_answer(text)
    assert stripped == "The second excerpt does not provide his birth date."
    assert NO_ANSWER not in stripped


def test_stripping_a_trailing_sentinel_on_its_own_line():
    text = "No actress matches that description.\n\nNO_ANSWER"
    stripped = strip_no_answer(text)
    assert stripped == "No actress matches that description."
    assert NO_ANSWER not in stripped


def test_refusal_text_never_leaks_a_trailing_sentinel():
    text = "The excerpt lacks that figure. Therefore, NO_ANSWER."
    shown = refusal_text(text)
    assert NO_ANSWER not in shown
    assert shown == "The excerpt lacks that figure."
