import pytest
from core.models import Chunk, SearchResult
from generation.answerer import (
    Answerer,
    AnswerMode,
    NO_RESULTS_MESSAGE,
    build_excerpts,
    citation_labels,
    classify,
)
from generation.prompts import CONVERSATION_SYSTEM_PROMPT


class StubLLM:
    def __init__(self, reply="Contract number is SC-4471."):
        self.reply = reply
        self.prompts = []
        self.histories = []
        self.opts = []

    def generate(self, system, user, *, model=None, temperature=None, history=None):
        self.prompts.append((system, user))
        self.histories.append(history)
        self.opts.append((model, temperature))
        return self.reply

    def stream(self, system, user, *, model=None, temperature=None, history=None):
        self.prompts.append((system, user))
        self.histories.append(history)
        self.opts.append((model, temperature))
        yield self.reply


def _result(filename, page, text, score=0.9, is_table=False, is_summary=False):
    return SearchResult(
        chunk=Chunk(
            doc_id="d1",
            filename=filename,
            text=text,
            chunk_index=0,
            page=page,
            is_table=is_table,
            is_summary=is_summary,
        ),
        score=score,
    )


def test_citations_derive_from_chunks_not_model_output():
    llm = StubLLM(reply="The answer is in doc_that_does_not_exist.pdf")
    answerer = Answerer(llm)

    answer = answerer.answer("q", [_result("real.pdf", 4, "text")])

    assert answer.citations == ["real.pdf, p. 4"]


def test_citations_are_deduplicated_by_file_and_page():
    answerer = Answerer(StubLLM())
    results = [
        _result("a.pdf", 1, "one"),
        _result("a.pdf", 1, "two"),
        _result("a.pdf", 2, "three"),
    ]

    answer = answerer.answer("q", results)

    assert answer.citations == ["a.pdf, p. 1", "a.pdf, p. 2"]


def test_empty_results_refuse_without_calling_the_model():
    llm = StubLLM()
    answerer = Answerer(llm)

    answer = answerer.answer("q", [])

    assert answer.refused is True
    assert answer.citations == []
    assert llm.prompts == []


def test_context_includes_source_labels_for_each_chunk():
    llm = StubLLM()
    Answerer(llm).answer("q", [_result("a.pdf", 7, "body text")])

    _system, user = llm.prompts[0]
    assert "a.pdf, p. 7" in user
    assert "body text" in user


def test_citation_labels_dedupes_preserving_first_seen_order():
    results = [
        _result("b.pdf", 2, "x"),
        _result("a.pdf", 1, "y"),
        _result("b.pdf", 2, "z"),  # duplicate label, must not reappear
    ]
    assert citation_labels(results) == ["b.pdf, p. 2", "a.pdf, p. 1"]


def test_citation_labels_empty_for_no_results():
    assert citation_labels([]) == []


def test_build_excerpts_pairs_labels_with_text():
    assert build_excerpts([_result("a.pdf", 3, "body")]) == [("a.pdf, p. 3", "body")]


def test_stream_yields_answer_and_uses_source_labels():
    llm = StubLLM(reply="streamed answer")
    chunks = list(Answerer(llm).stream("q", [_result("a.pdf", 7, "body text")]))

    assert "".join(chunks) == "streamed answer"
    _system, user = llm.prompts[0]
    assert "a.pdf, p. 7" in user and "body text" in user


def test_per_request_model_and_temperature_are_threaded_to_the_llm():
    llm = StubLLM()
    Answerer(llm).answer(
        "q", [_result("a.pdf", 1, "x")], model="llama3", temperature=0.7
    )
    assert llm.opts[0] == ("llama3", 0.7)

    list(
        Answerer(llm).stream(
            "q", [_result("a.pdf", 1, "x")], model="qwen", temperature=0.2
        )
    )
    assert llm.opts[1] == ("qwen", 0.2)


# --- classify: the shared refuse / guard / answer policy ---------------------


def test_classify_no_results_when_search_refused():
    assert classify("anything", True, []) is AnswerMode.NO_RESULTS


def test_classify_answers_normal_prose_question():
    results = [_result("a.pdf", 1, "the contract number is SC-4471")]
    assert classify("what is the contract number?", False, results) is AnswerMode.ANSWER


def test_classify_refuses_aggregation_over_table_heavy_results():
    results = [
        _result("a.pdf", 1, "| x | y |", is_table=True),
        _result("a.pdf", 2, "| a | b |", is_table=True),
    ]
    assert (
        classify("what is the total?", False, results) is AnswerMode.AGGREGATION_REFUSED
    )


def test_classify_defers_aggregation_when_a_summary_was_retrieved():
    # A precomputed aggregate summary means the total is already a ready fact.
    results = [
        _result("a.pdf", 1, "| x | y |", is_table=True),
        _result("a.pdf", 1, "Aggregate: total 600", is_summary=True),
    ]
    assert classify("what is the total?", False, results) is AnswerMode.ANSWER


# --- history / context summarisation -----------------------------------------


def test_history_is_passed_through_to_llm():
    llm = StubLLM()
    answerer = Answerer(llm)
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]

    answerer.answer("second question", [_result("a.pdf", 1, "x")], history=history)

    # The main answer call is the second generate() — the first was the
    # summarisation call. histories[1] is the history passed to the real
    # answer prompt.
    assert llm.histories[1] == history


def test_summary_is_injected_into_user_prompt():
    llm = StubLLM(reply="the user asked about contracts")
    answerer = Answerer(llm)
    history = [
        {"role": "user", "content": "what is the contract number?"},
        {"role": "assistant", "content": "SC-4471"},
    ]

    answerer.answer(
        "what was it again?", [_result("a.pdf", 1, "contract SC-4471")], history=history
    )

    # The second prompt (main answer) should contain the summary from the
    # first call.
    _system, user = llm.prompts[1]
    assert "Conversation so far:" in user
    assert "the user asked about contracts" in user


def test_empty_history_produces_no_summary():
    llm = StubLLM()
    answerer = Answerer(llm)

    answerer.answer("only question", [_result("a.pdf", 1, "x")], history=[])

    # No summarisation call was made — only the main answer.
    assert len(llm.prompts) == 1
    _system, user = llm.prompts[0]
    assert "Conversation so far:" not in user


def test_single_turn_history_produces_no_summary():
    # < 2 messages is too short to summarise — only the main answer call.
    llm = StubLLM()
    answerer = Answerer(llm)

    answerer.answer(
        "only question",
        [_result("a.pdf", 1, "x")],
        history=[{"role": "user", "content": "hi"}],
    )

    assert len(llm.prompts) == 1
    _system, user = llm.prompts[0]
    assert "Conversation so far:" not in user


def test_history_none_default_preserves_existing_behaviour():
    llm = StubLLM()
    answerer = Answerer(llm)

    answer = answerer.answer("q", [_result("a.pdf", 1, "text")])

    assert answer.refused is False
    assert answer.citations == ["a.pdf, p. 1"]
    # No summarisation call, main call has history=None.
    assert len(llm.prompts) == 1
    assert llm.histories[0] is None


def test_summarize_history_uses_the_requested_model():
    # The summarization call used to always hit cfg.llm_model regardless of
    # what the user picked in the UI, forcing Ollama to swap two different
    # models in and out on every turn. It must use the same model as the
    # answer it's summarizing context for.
    llm = StubLLM()
    answerer = Answerer(llm)
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    answerer.summarize_history(history, model="llama3")

    assert llm.opts[0][0] == "llama3"


def test_answer_threads_model_into_its_own_summarization_call():
    llm = StubLLM()
    answerer = Answerer(llm)
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]

    answerer.answer(
        "second question", [_result("a.pdf", 1, "x")], model="llama3", history=history
    )

    # opts[0] is the summarization call, opts[1] the main answer call —
    # both must target the model the caller asked for.
    assert llm.opts[0][0] == "llama3"
    assert llm.opts[1][0] == "llama3"


def test_verbatim_history_sent_to_answer_call_is_capped_but_summary_sees_all():
    # Unbounded history was being sent to the LLM twice per turn: once
    # condensed into the summary, once again verbatim and growing forever.
    # The verbatim copy should be windowed to recent turns — the summary
    # already carries the older context.
    llm = StubLLM()
    answerer = Answerer(llm)
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})

    answerer.answer("latest question", [_result("a.pdf", 1, "x")], history=history)

    # Summarization (first call) still sees the full transcript, including
    # the oldest exchange.
    _summary_system, summary_user = llm.prompts[0]
    assert "q0" in summary_user

    # The verbatim history replayed to the answer call (second call) is
    # capped to the most recent exchanges.
    verbatim_history = llm.histories[1]
    assert len(verbatim_history) == 6
    assert all(m["content"] not in ("q0", "a0") for m in verbatim_history)
    assert verbatim_history[-1]["content"] == "a4"


def test_short_history_is_sent_verbatim_unchanged():
    # Below the cap, nothing is trimmed — same behavior as before.
    llm = StubLLM()
    answerer = Answerer(llm)
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]

    answerer.answer("second question", [_result("a.pdf", 1, "x")], history=history)

    assert llm.histories[1] == history


def test_stream_with_history_includes_context_summary():
    llm = StubLLM(reply="summary text")
    answerer = Answerer(llm)
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]

    list(answerer.stream("second", [_result("a.pdf", 1, "x")], history=history))

    # First prompt = summarisation, second = main answer with summary injected.
    _system, user = llm.prompts[1]
    assert "Conversation so far:" in user
    assert "summary text" in user


# --- conversational fallback (no document results) ---------------------------


def test_converse_stream_uses_conversation_prompt_and_yields_tokens():
    llm = StubLLM(reply="Your name is Alex.")
    answerer = Answerer(llm)
    history = [
        {"role": "user", "content": "my name is Alex"},
        {"role": "assistant", "content": NO_RESULTS_MESSAGE},
    ]

    chunks = list(answerer.converse_stream("what is my name?", history=history))

    assert "".join(chunks) == "Your name is Alex."
    system, _user = llm.prompts[0]
    assert system is CONVERSATION_SYSTEM_PROMPT
    assert llm.histories[0] == history


def test_converse_stream_threads_model_and_temperature():
    llm = StubLLM()
    answerer = Answerer(llm)

    list(answerer.converse_stream("hi", model="llama3", temperature=0.5))

    assert llm.opts[0] == ("llama3", 0.5)


def test_converse_stream_with_no_history_still_calls_llm():
    # Unlike the NO_RESULTS path in answer(), converse_stream() always calls
    # the LLM — it's the caller's responsibility to decide when to use it.
    llm = StubLLM(reply="I don't have any context yet.")
    answerer = Answerer(llm)

    list(answerer.converse_stream("what is my name?", history=None))

    assert llm.prompts  # LLM was called
