import pytest
from core.models import Chunk, SearchResult
from generation.answerer import (
    Answerer,
    AnswerMode,
    AnswerStream,
    NO_RESULTS_MESSAGE,
    build_excerpts,
    build_citations,
    citation_labels,
    classify,
    prune_citations,
)
from generation.guards import NO_ANSWER_MESSAGE
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


def _result(filename, page, text, score=0.9, is_table=False,
            is_summary=False, chunk_index=0):
    return SearchResult(
        chunk=Chunk(
            doc_id="d1",
            filename=filename,
            text=text,
            chunk_index=chunk_index,
            page=page,
            is_table=is_table,
            is_summary=is_summary,
        ),
        score=score,
    )


def test_citations_derive_from_chunks_not_model_output():
    llm = StubLLM(reply="The answer is in doc_that_does_not_exist.pdf")
    answerer = Answerer(llm)

    # Chunk text overlaps with the reply so it survives citation pruning.
    answer = answerer.answer("q", [_result("real.pdf", 4, "answer doc pdf")])

    assert answer.citations == ["real.pdf, p. 4"]


def test_citations_are_deduplicated_by_file_and_page():
    answerer = Answerer(StubLLM())
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("a.pdf", 1, "contract details"),
        _result("a.pdf", 2, "number 4471 reference"),
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
    # One call: the streamed answer. answer() deliberately does not run the
    # grounding check — that is answer_async/ground_async's job, off the
    # critical path. See Answerer.answer.
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

    # No summarisation call was made — the answer stream is the only call,
    # and the answer prompt must not carry a conversation summary.
    assert len(llm.prompts) == 1
    _system, user = llm.prompts[0]
    assert "Conversation so far:" not in user


def test_single_turn_history_produces_no_summary():
    # < 2 messages is too short to summarise, so the answer stream is the
    # only call.
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

    # Chunk text overlaps with the stub reply so it survives pruning.
    answer = answerer.answer("q", [_result("a.pdf", 1, "contract number SC-4471")])

    assert answer.refused is False
    assert answer.citations == ["a.pdf, p. 1"]
    # history=None means no summarisation call, so the answer stream is
    # the only call.
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


def test_answer_marks_the_models_own_refusal_and_drops_citations():
    llm = StubLLM(reply="The excerpts do not contain that information.")
    answer = Answerer(llm).answer("q?", [_result("a.pdf", 1, "a")])

    assert answer.refused
    assert answer.citations == []


class SequencedLLM:
    """Streams one answer, then returns a fixed verdict from generate()."""

    def __init__(self, answer, verdict):
        self.answer = answer
        self.verdict = verdict
        self.prompts = []

    def generate(self, system, user, *, model=None, temperature=None, history=None):
        self.prompts.append((system, user))
        return self.verdict

    def stream(self, system, user, *, model=None, temperature=None, history=None):
        self.prompts.append((system, user))
        yield self.answer


# _grounded is deliberately NOT wired into answer() — on qwen2.5:3b it refused
# 23 of 25 good answers. It stays covered here because eval/replay_gate.py
# scores candidate models through it.


def test_grounded_reads_the_judges_verdict():
    llm = SequencedLLM("The SH-60 Seahawk cruises at Mach 2.2.", "UNSUPPORTED")
    assert not Answerer(llm)._grounded("answer", [_result("a.pdf", 1, "text")])

    llm = SequencedLLM("Multan sits on the Chenab River.", "SUPPORTED")
    assert Answerer(llm)._grounded("answer", [_result("a.pdf", 1, "text")])


def test_grounded_fails_open_when_the_judge_is_down():
    # A judge outage must never drop a valid answer.
    class FailingLLM(SequencedLLM):
        def generate(self, system, user, *, model=None, temperature=None, history=None):
            raise RuntimeError("judge down")

    assert Answerer(FailingLLM("x", "y"))._grounded("answer", [_result("a.pdf", 1, "t")])


def test_answer_async_returns_immediately_and_grounds_in_background():
    import time

    class SlowJudgeLLM(SequencedLLM):
        """Streams instantly, but the grounding generate() blocks briefly."""

        def generate(self, system, user, *, model=None, temperature=None, history=None):
            time.sleep(0.2)
            return self.verdict

    llm = SlowJudgeLLM("The contract number is SC-4471.", "SUPPORTED")
    answerer = Answerer(llm)

    answer = answerer.answer_async("q?", [_result("a.pdf", 1, "x")])

    # The answer is returned before the grounding check has run.
    assert answer.text == "The contract number is SC-4471."
    assert answer.grounded is None

    # The background thread lands the verdict shortly after.
    deadline = time.monotonic() + 5
    while answer.grounded is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert answer.grounded is True


def test_answer_async_skips_grounding_on_refusal():
    llm = SequencedLLM("The excerpts do not contain that information.", "UNSUPPORTED")
    answer = Answerer(llm).answer_async("q?", [_result("a.pdf", 1, "a")])

    assert answer.refused
    assert answer.citations == []
    assert answer.grounded is None  # no background thread spawned


def test_ground_async_runs_the_gate_on_existing_text():
    import time

    class SlowJudgeLLM(SequencedLLM):
        def generate(self, system, user, *, model=None, temperature=None, history=None):
            time.sleep(0.2)
            return self.verdict

    llm = SlowJudgeLLM("The contract number is SC-4471.", "UNSUPPORTED")
    answer = Answerer(llm).ground_async(
        "The contract number is SC-4471.", [_result("a.pdf", 1, "x")]
    )

    # Returns immediately with the verdict still pending.
    assert answer.text == "The contract number is SC-4471."
    assert answer.grounded is None

    deadline = time.monotonic() + 5
    while answer.grounded is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert answer.grounded is False  # UNSUPPORTED verdict lands in background


# AnswerStream: the NO_ANSWER sentinel must never reach the screen, but the
# refusal it signals must still reach the caller. st.write_stream renders each
# delta on arrival, so those two requirements pull in opposite directions.

def test_answer_stream_hides_the_sentinel_and_reports_the_refusal():
    stream = AnswerStream(iter(["NO_", "ANSWER", " The excerpts ", "stop at 2019."]))
    assert "".join(stream) == "The excerpts stop at 2019."
    assert stream.refused is True


def test_answer_stream_passes_a_real_answer_through_unchanged():
    deltas = ["Multan sits on ", "the Chenab ", "River."]
    stream = AnswerStream(iter(deltas))
    assert "".join(stream) == "".join(deltas)
    assert stream.refused is False


def test_answer_stream_handles_a_sentinel_split_across_many_deltas():
    """One character per delta is the worst case for the buffer, and the case
    a naive startswith() on the first delta gets wrong."""
    stream = AnswerStream(iter(list("NO_ANSWER") + [" nothing here."]))
    assert "".join(stream) == "nothing here."
    assert stream.refused is True


def test_answer_stream_survives_a_stream_shorter_than_the_sentinel():
    stream = AnswerStream(iter(["No."]))
    assert "".join(stream) == "No."
    assert stream.refused is False


def test_answer_strips_the_sentinel_but_still_marks_the_refusal():
    llm = StubLLM(reply="NO_ANSWER The excerpts do not cover 2020.")
    answer = Answerer(llm).answer("q", [_result("a.pdf", 1, "x")])
    assert answer.refused is True
    assert answer.citations == []
    assert answer.text == "The excerpts do not cover 2020."
    assert "NO_ANSWER" not in answer.text


# Qwen3.8-27 answers a non-answer with the bare sentinel and nothing else, so
# stripping the token leaves an empty string. A refusal that renders as blank
# is worse than no refusal at all — the user sees an empty reply.

def test_a_bare_sentinel_stream_still_shows_something():
    stream = AnswerStream(iter(["NO_", "ANSWER"]))
    assert "".join(stream) == NO_ANSWER_MESSAGE
    assert stream.refused is True


def test_answer_never_returns_a_blank_refusal():
    answer = Answerer(StubLLM(reply="NO_ANSWER")).answer(
        "q", [_result("a.pdf", 1, "x")])
    assert answer.refused is True
    assert answer.citations == []
    assert answer.text == NO_ANSWER_MESSAGE


def test_a_sentinel_with_its_own_explanation_keeps_that_explanation():
    answer = Answerer(StubLLM(reply="NO_ANSWER The excerpts stop at 2019.")).answer(
        "q", [_result("a.pdf", 1, "x")])
    assert answer.text == "The excerpts stop at 2019."


# ──────────────────────────────────────────────────────────────────────────────
# Citation pruning: answer-overlap filtering
# ──────────────────────────────────────────────────────────────────────────────

def test_prune_citations_drops_chunks_with_no_answer_overlap():
    """A chunk whose text shares no content words with the answer is noise
    from fusion — it was retrieved but the model never used it."""
    results = [
        _result("relevant.pdf", 1, "contract number SC-4471"),
        _result("irrelevant.pdf", 1, "weather forecast rain tomorrow"),
    ]
    pruned = prune_citations("The contract number is SC-4471.", results)
    labels = [r.chunk.citation_label() for r in pruned]
    assert labels == ["relevant.pdf, p. 1"]


def test_prune_citations_keeps_chunks_with_word_overlap():
    """Chunks that share content words with the answer are kept."""
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("b.pdf", 2, "SC-4471 reference document"),
    ]
    pruned = prune_citations("The contract number is SC-4471.", results)
    labels = [r.chunk.citation_label() for r in pruned]
    assert "a.pdf, p. 1" in labels
    assert "b.pdf, p. 2" in labels


def test_prune_citations_deduplicates_by_label():
    """Multiple chunks from the same page collapse to one citation."""
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("a.pdf", 1, "contract details SC-4471"),
    ]
    pruned = prune_citations("The contract number is SC-4471.", results)
    assert len(pruned) == 1
    assert pruned[0].chunk.citation_label() == "a.pdf, p. 1"


def test_prune_citations_preserves_first_seen_order():
    """Results are returned in the order they first appear."""
    results = [
        _result("z.pdf", 1, "contract number SC-4471"),
        _result("a.pdf", 1, "number SC-4471 details"),
        _result("m.pdf", 1, "weather rain forecast"),
    ]
    pruned = prune_citations("The contract number is SC-4471.", results)
    labels = [r.chunk.citation_label() for r in pruned]
    assert labels == ["z.pdf, p. 1", "a.pdf, p. 1"]


def test_prune_citations_returns_all_when_answer_too_short():
    """A very short answer can't produce meaningful overlap — safe default
    is to keep everything (preserves citation_accuracy)."""
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("b.pdf", 1, "weather forecast rain"),
    ]
    pruned = prune_citations("Yes.", results)
    assert len(pruned) == 2


def test_prune_citations_returns_all_for_empty_answer():
    results = [_result("a.pdf", 1, "anything")]
    assert prune_citations("", results) == results


def test_prune_citations_requires_min_two_overlap_words():
    """A single shared word is not enough — it could be coincidence."""
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("b.pdf", 1, "contract zzzz"),
    ]
    # a.pdf shares "contract", "sc", "4471" (3 words) — kept.
    # b.pdf shares only "contract" (1 word) — below _MIN_OVERLAP_WORDS=2.
    pruned = prune_citations("The contract is SC-4471.", results)
    labels = [r.chunk.citation_label() for r in pruned]
    assert "a.pdf, p. 1" in labels
    assert "b.pdf, p. 1" not in labels


def test_citation_labels_with_answer_text_prunes():
    """citation_labels accepts an optional answer_text for pruning."""
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("b.pdf", 1, "weather forecast rain"),
    ]
    labels = citation_labels(results, "The contract number is SC-4471.")
    assert labels == ["a.pdf, p. 1"]


def test_citation_labels_without_answer_text_no_pruning():
    """Backward-compatible: without answer_text, no pruning is applied."""
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("b.pdf", 1, "weather forecast rain"),
    ]
    labels = citation_labels(results)
    assert labels == ["a.pdf, p. 1", "b.pdf, p. 1"]


def test_build_citations_with_answer_text_prunes():
    """build_citations accepts an optional answer_text for pruning."""
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("b.pdf", 1, "weather forecast rain"),
    ]
    citations = build_citations(results, "The contract number is SC-4471.")
    assert len(citations) == 1
    assert citations[0].label == "a.pdf, p. 1"


def test_answer_prunes_citations_to_answer_overlap():
    """The answer() method should only cite chunks that overlap with the
    generated answer text."""
    llm = StubLLM(reply="The contract number is SC-4471.")
    answerer = Answerer(llm)
    results = [
        _result("relevant.pdf", 1, "contract number SC-4471"),
        _result("noise.pdf", 1, "weather forecast rain tomorrow"),
    ]
    answer = answerer.answer("q", results)
    assert answer.citations == ["relevant.pdf, p. 1"]


def test_answer_keeps_all_citations_when_answer_overlaps_all():
    """When the answer draws from all chunks, all are cited."""
    llm = StubLLM(reply="The contract number SC-4471 is found in document reference.")
    answerer = Answerer(llm)
    results = [
        _result("a.pdf", 1, "contract number SC-4471"),
        _result("b.pdf", 1, "document reference page"),
    ]
    answer = answerer.answer("q", results)
    assert set(answer.citations) == {"a.pdf, p. 1", "b.pdf, p. 1"}
