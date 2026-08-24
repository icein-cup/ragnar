import re
import threading
from dataclasses import dataclass, field
from enum import Enum

from core.models import SearchResult, Citation
from generation.guards import (NO_ANSWER, NO_ANSWER_MESSAGE, is_refusal,
                              refusal_text, should_refuse_aggregation,
                              strip_no_answer)
from generation.prompts import (
    SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    build_user_prompt,
    build_history_summary_prompt,
    build_grounding_prompt,
)

NO_RESULTS_MESSAGE = "I could not find anything relevant in the indexed documents."

# How many of the most recent messages are replayed verbatim to the answer
# call. Summarization (summarize_history) always sees the full transcript —
# only this verbatim copy is windowed, since sending the whole conversation
# twice (once condensed, once in full) grows the prompt without bound as a
# session gets longer.
MAX_HISTORY_MESSAGES = 6

# ──────────────────────────────────────────────────────────────────────────────
# Citation pruning: answer-overlap filtering
# ──────────────────────────────────────────────────────────────────────────────
# After the answer is generated, we filter the citation list to only those
# chunks whose text has meaningful word overlap with the answer. This drops
# the noise from fusion (chunks retrieved by multi-query/multi-hop that the
# model never actually used) without touching the result set sent to the LLM.
#
# Reranker-based pruning was tried and failed (see the comment on Answerer
# below): bge-reranker-v2-m3 scores 44% of needed chunks at the same neutral
# 0.500 as irrelevant ones, so every threshold that lifted precision halved
# source recall. Answer-overlap filtering sidesteps that entirely — it asks
# "did the model USE this chunk?" not "does the reranker THINK this chunk is
# relevant?".

# Stopwords excluded from overlap counting. These appear in almost every
# chunk and every answer, so matching on them would keep every citation
# (defeating the purpose). Only content words count.
_STOPWORDS = frozenset(
    # English
    "a an the and or but in on at to for of is are was were be been being "
    "by with from as it its this that these those has have had will would "
    "could should may might can do does did not no yes if then than also "
    "more most about into over under between within without which who whom "
    "what when where why how all each every both few many much some any "
    "such only own same so than too very just per via etc".split()
    # Polish — common function words that would inflate overlap counts
    + "i w na z o że się jest są nie od po dla jak tylko lub ale czy "
    "to co kto czym co do gdy gdyby aby ponieważ więc więc lecz jednak "
    "już jeszcze też również bardzo więcej mniej bez nad pod przed za "
    "przy przez podczas oraz orazże żeby aby żeby".split()
)

# Minimum number of distinct content-word overlaps for a chunk to be cited.
# Set to 2 — a single shared word could be a coincidence (a common noun,
# a number), but two distinct content words appearing in both the chunk and
# the answer is strong evidence the model drew from that chunk. This is
# deliberately conservative: the goal is to drop chunks with ZERO content
# overlap (pure fusion noise), not to aggressively trim borderline ones.
_MIN_OVERLAP_WORDS = 2


def _content_words(text: str) -> set[str]:
    """Alphanumeric words lowercased, minus stopwords.

    Uses a Unicode-aware regex so Polish diacritics (ł, ą, ę, ś, ż, ź, ć, ń, ó)
    are preserved rather than splitting words. Shares the tokeniser approach
    from eval/metrics.py._words but adds stopword filtering on top.
    """
    words = re.findall(r"\w+", text.lower(), re.UNICODE)
    return {w for w in words if w not in _STOPWORDS}


def prune_citations(answer_text: str,
                    results: list[SearchResult]) -> list[SearchResult]:
    """Filter results to only those whose chunk text overlaps the answer.

    Returns results in first-seen order, deduplicated by citation label.
    A chunk is kept when at least _MIN_OVERLAP_WORDS distinct content words
    from its text appear in the answer.

    If the answer is empty or too short to extract meaningful words, no
    filtering is applied — return all results (safe default that preserves
    citation_accuracy).
    """
    answer_words = _content_words(answer_text)
    if len(answer_words) < _MIN_OVERLAP_WORDS:
        # Answer too short to filter meaningfully — keep everything.
        return results

    seen_labels: set[str] = set()
    kept: list[SearchResult] = []
    for r in results:
        label = r.chunk.citation_label()
        if label in seen_labels:
            continue
        chunk_words = _content_words(r.chunk.text)
        overlap = answer_words & chunk_words
        if len(overlap) >= _MIN_OVERLAP_WORDS:
            seen_labels.add(label)
            kept.append(r)
    return kept


def _recent(history: list[dict] | None) -> list[dict] | None:
    if not history:
        return history
    return history[-MAX_HISTORY_MESSAGES:]


def citation_labels(results: list[SearchResult],
                     answer_text: str | None = None) -> list[str]:
    """Deduplicated citation labels from search results, first-seen order.

    Citations come from retrieved chunk metadata, never from model prose —
    the model can't cite a document that wasn't actually retrieved.

    When answer_text is provided, citations are pruned to only those chunks
    whose text has word overlap with the answer. See prune_citations.
    """
    if answer_text is not None:
        results = prune_citations(answer_text, results)
    seen: list[str] = []
    for result in results:
        label = result.chunk.citation_label()
        if label not in seen:
            seen.append(label)
    return seen


def build_citations(results: list[SearchResult],
                    answer_text: str | None = None) -> list[Citation]:
    """Rich citations with navigation metadata for the UI.

    Deduplicated by label, first-seen order. When answer_text is provided,
    citations are pruned to only those chunks whose text has word overlap
    with the answer. See prune_citations.
    """
    if answer_text is not None:
        results = prune_citations(answer_text, results)
    seen: set[str] = set()
    citations: list[Citation] = []
    for result in results:
        chunk = result.chunk
        label = chunk.citation_label()
        if label not in seen:
            seen.add(label)
            citations.append(Citation(
                label=label,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page=chunk.page,
                sheet=chunk.sheet,
                chunk_index=chunk.chunk_index,
            ))
    return citations


def build_excerpts(results: list[SearchResult]) -> list[tuple[str, str]]:
    """(citation_label, text) pairs for the prompt builder."""
    return [(r.chunk.citation_label(), r.chunk.text) for r in results]


class AnswerMode(Enum):
    NO_RESULTS = "no_results"
    AGGREGATION_REFUSED = "aggregation_refused"
    ANSWER = "answer"


def classify(question: str, refused: bool, results: list[SearchResult]) -> AnswerMode:
    """The single source of truth for the refuse / guard / answer decision.

    Both the UI and the eval harness route through this so the policy —
    "no candidates cleared the floor" vs "aggregation over a table we can't
    safely compute" vs "answer normally" — can never drift between them.
    """
    if refused:
        return AnswerMode.NO_RESULTS
    if should_refuse_aggregation(question, results):
        return AnswerMode.AGGREGATION_REFUSED
    return AnswerMode.ANSWER


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    refused: bool = False
    grounded: bool | None = None  # None = grounding check not run


class AnswerStream:
    """Answer deltas with the NO_ANSWER sentinel held back and remembered.

    st.write_stream paints each delta the moment it arrives, so the sentinel
    has to be removed before display or it flashes on screen. That leaves
    ``is_refusal`` unable to find it in the finished text, which is what
    ``refused`` is for — read it once the stream is exhausted.

    Only enough of the opening is buffered to recognise the token; the rest
    passes straight through, so this costs one delta of latency and nothing
    else.
    """

    def __init__(self, deltas):
        self._deltas = iter(deltas)
        self.refused = False

    def __iter__(self):
        head = ""
        for delta in self._deltas:
            head += delta
            if len(head.lstrip()) >= len(NO_ANSWER):
                break
        if head.lstrip().startswith(NO_ANSWER):
            self.refused = True
            head = strip_no_answer(head)
            # The buffer stops the moment the token is recognised, so the
            # space or newline the model put after it is usually still in the
            # next delta. Pull until there is something real to show, or the
            # stream ends, so the answer does not open with stray whitespace.
            while not head.strip():
                try:
                    head += next(self._deltas)
                except StopIteration:
                    break
            head = head.lstrip()
        shown = False
        if head:
            shown = True
            yield head
        for delta in self._deltas:
            if delta:
                shown = True
            yield delta
        if self.refused and not shown:
            # The whole reply was the bare sentinel. Yielding nothing would
            # render the refusal as an empty answer.
            yield NO_ANSWER_MESSAGE


class Answerer:
    def __init__(self, llm):
        self._llm = llm

    # Citations are pruned by answer-overlap after generation. Pruning them
    # by scoring each chunk against the finished answer was tried and measured:
    # bge-reranker-v2-m3 gives 44% of the chunks a question NEEDS the same
    # neutral 0.500 it gives unrelated ones, so every threshold that lifted
    # citation precision from 0.26 to ~0.5 dropped source recall from 0.74 to
    # ~0.42 — halving multi_hop_citation_accuracy to tidy the citation list.
    # The noise comes from retrieval handing over sixteen chunks; answer-
    # overlap filtering (prune_citations) fixes the citation list without
    # touching the context set the model sees, so answer quality is preserved.

    def summarize_history(
        self, history: list[dict], model: str | None = None
    ) -> str | None:
        """Condense previous Q&A into 2-3 sentences, or None if too short.

        One turn or an empty history has nothing useful to summarize — the
        model handles a single follow-up fine without a summary.

        model should match whatever will generate the actual answer — using
        a different model for this call forces Ollama to swap two models in
        and out on every turn, which is slow and pointless.
        """
        if len(history) < 2:
            return None
        system_prompt, user_prompt = build_history_summary_prompt(history)
        try:
            return self._llm.generate(system_prompt, user_prompt, model=model)
        except Exception:
            return None

    def answer(
        self,
        question: str,
        results: list[SearchResult],
        *,
        model: str | None = None,
        temperature: float | None = None,
        history: list[dict] | None = None,
        context_summary: str | None = None,
    ) -> Answer:
        if not results:
            # Skip the model entirely — a refusal it cannot embellish.
            return Answer(text=NO_RESULTS_MESSAGE, refused=True)

        text = "".join(self.stream(
            question, results, model=model, temperature=temperature,
            history=history, context_summary=context_summary,
        ))
        # is_refusal reads the raw text — the NO_ANSWER sentinel is the point
        # — and only then is the token stripped from what callers see.
        refused = is_refusal(text)
        if refused:
            # The model said the excerpts do not cover this. Citing sources
            # for a non-answer makes it read as a sourced one.
            return Answer(text=refusal_text(text), citations=[], refused=True)
        text = strip_no_answer(text)
        # No grounding check here, deliberately. This is the path the eval
        # harness takes (eval/run_eval.py), and _grounded is a second LLM
        # call per case whose verdict no report or metric records — it
        # doubled a 55-minute benchmark to populate a field nothing read.
        # answer_async and ground_async run it off the critical path for the
        # UI, which is where it belongs until it measures well enough to gate
        # on. See _grounded's docstring.
        return Answer(text=text, citations=citation_labels(results, text))

    def answer_async(
        self,
        question: str,
        results: list[SearchResult],
        *,
        model: str | None = None,
        temperature: float | None = None,
        history: list[dict] | None = None,
        context_summary: str | None = None,
    ) -> Answer:
        """Answer immediately, then run the grounding check in the background.

        The answer streams synchronously (the caller still waits for the text),
        but the grounding verdict — a second, slow LLM call — runs on a daemon
        thread and lands on ``answer.grounded`` whenever it finishes. The caller
        never blocks on it; a judge outage or a slow model simply leaves
        ``grounded`` as None.

        This is the latency-safe form of ``answer()``: the user sees the answer
        as fast as the model can produce it, and the fabrication check catches
        up in the background instead of adding a second model round-trip to the
        critical path.
        """
        if not results:
            return Answer(text=NO_RESULTS_MESSAGE, refused=True)

        text = "".join(self.stream(
            question, results, model=model, temperature=temperature,
            history=history, context_summary=context_summary,
        ))
        refused = is_refusal(text)
        if refused:
            return Answer(text=refusal_text(text), citations=[], refused=True)
        text = strip_no_answer(text)

        answer = Answer(text=text, citations=citation_labels(results, text))

        def _check() -> None:
            answer.grounded = self._grounded(text, results, model=model)

        threading.Thread(target=_check, daemon=True).start()
        return answer

    def ground_async(
        self,
        text: str,
        results: list[SearchResult],
        *,
        model: str | None = None,
    ) -> Answer:
        """Run the grounding check on already-produced text in the background.

        The UI streams the answer itself (st.write_stream over ``stream()``),
        so it already has the text and cannot use ``answer_async`` (which
        joins the stream internally). This spawns the same daemon-thread gate
        and returns an ``Answer`` whose ``grounded`` field lands whenever the
        judge finishes — the caller never blocks on it.

        Refusals are the caller's concern here: pass only non-refusal text,
        or the gate will judge a "the excerpts do not cover this" sentence
        against the excerpts and mark it UNSUPPORTED.
        """
        answer = Answer(text=text, citations=citation_labels(results, text))

        def _check() -> None:
            answer.grounded = self._grounded(text, results, model=model)

        threading.Thread(target=_check, daemon=True).start()
        return answer

    def _grounded(
        self,
        text: str,
        results: list[SearchResult],
        *,
        model: str | None = None,
    ) -> bool:
        """Is the answer fully supported by the retrieved excerpts?

        A second LLM pass that judges facts, not phrasing. The refusal-phrase
        detector only catches answers that *say* they lack the information;
        this catches answers that confidently state a fact the excerpts do not
        contain (a helicopter "cruising at Mach 2.2", a tuition figure from a
        different school, a number from the wrong year or column).

        ADVISORY ONLY, and never on the critical path: reached through
        ``answer_async`` / ``ground_async``, which run it on a daemon thread.
        ``answer()`` does not call it — that is the eval harness's path, and a
        second LLM call per case doubles a benchmark run to fill a field no
        metric reads.

        It does not gate anything, because it cannot yet. Best result so far
        is qwen2.5:7b with the accept rule added to GROUNDING_PROMPT: 9/20 bad
        answers caught for 7/25 correct answers destroyed. Before that rule it
        was 16/25 destroyed, and on qwen2.5:3b 23/25 — a gate that refuses
        everything. Even the best version would cost more correct answers than
        the hallucinations it catches are worth, while over-refusal is already
        the pipeline's largest failure. eval/replay_gate.py scores a candidate
        against a saved report in about two minutes; let this decide anything
        only once "correct answers lost" is near zero.

        Any failure to run the check (LLM error, malformed reply) resolves to
        True — the answer stands rather than being dropped on a judge outage.
        """
        system, user = build_grounding_prompt(text, build_excerpts(results))
        try:
            verdict = self._llm.generate(system, user, model=model).strip().upper()
        except Exception:
            return True
        return not verdict.startswith("UNSUPPORTED")

    def stream(
        self,
        question: str,
        results: list[SearchResult],
        *,
        model: str | None = None,
        temperature: float | None = None,
        history: list[dict] | None = None,
        context_summary: str | None = None,
    ):
        """Yield answer-text deltas for the UI's st.write_stream.

        Citations are not part of the stream — they come from
        citation_labels(results) and are known before generation starts.
        model/temperature are threaded through per call so a shared Answerer
        instance never has to mutate the underlying LLM's state.
        """
        if context_summary is None:
            context_summary = self.summarize_history(history or [], model=model)
        yield from self._llm.stream(
            SYSTEM_PROMPT,
            build_user_prompt(
                question, build_excerpts(results), context_summary=context_summary
            ),
            model=model,
            temperature=temperature,
            history=_recent(history),
        )

    def converse_stream(
        self,
        question: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        history: list[dict] | None = None,
    ):
        """Answer personal / chitchat questions from conversation history.

        Used when document retrieval found nothing but there is prior
        conversation context. The model may use what the user said earlier
        (e.g. their name) while still declining to fabricate document content.
        Yields deltas for the UI's st.write_stream; no citations, since no
        documents were retrieved.
        """
        yield from self._llm.stream(
            CONVERSATION_SYSTEM_PROMPT,
            question,
            model=model,
            temperature=temperature,
            history=_recent(history),
        )
