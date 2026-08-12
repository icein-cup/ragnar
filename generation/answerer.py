from dataclasses import dataclass, field
from enum import Enum

from core.models import SearchResult, Citation
from generation.guards import should_refuse_aggregation
from generation.prompts import (
    SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    build_user_prompt,
    build_history_summary_prompt,
)

NO_RESULTS_MESSAGE = "I could not find anything relevant in the indexed documents."

# How many of the most recent messages are replayed verbatim to the answer
# call. Summarization (summarize_history) always sees the full transcript —
# only this verbatim copy is windowed, since sending the whole conversation
# twice (once condensed, once in full) grows the prompt without bound as a
# session gets longer.
MAX_HISTORY_MESSAGES = 6


def _recent(history: list[dict] | None) -> list[dict] | None:
    if not history:
        return history
    return history[-MAX_HISTORY_MESSAGES:]


def citation_labels(results: list[SearchResult]) -> list[str]:
    """Deduplicated citation labels from search results, first-seen order.

    Citations come from retrieved chunk metadata, never from model prose —
    the model can't cite a document that wasn't actually retrieved.
    """
    seen: list[str] = []
    for result in results:
        label = result.chunk.citation_label()
        if label not in seen:
            seen.append(label)
    return seen


def build_citations(results: list[SearchResult]) -> list[Citation]:
    """Rich citations with navigation metadata for the UI.

    Deduplicated by label, first-seen order.
    """
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


class Answerer:
    def __init__(self, llm):
        self._llm = llm

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

        if context_summary is None:
            context_summary = self.summarize_history(history or [], model=model)
        text = self._llm.generate(
            SYSTEM_PROMPT,
            build_user_prompt(
                question, build_excerpts(results), context_summary=context_summary
            ),
            model=model,
            temperature=temperature,
            history=_recent(history),
        )

        return Answer(text=text, citations=citation_labels(results))

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
