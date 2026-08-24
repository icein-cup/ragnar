SYSTEM_PROMPT = """\
You answer questions strictly from the provided document excerpts.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- If the excerpts do not contain the answer, say so plainly. Do not guess.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language.
- Be concise and factual. Do not speculate or embellish.
- Citations are added separately after your answer — do not include your \
own citations or source references in the response text.
"""

CONVERSATION_SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using the conversation \
history when the indexed documents did not contain anything relevant.

Rules:
- For personal questions (name, preferences, prior statements) use what the \
user said earlier in the conversation.
- If the user asks about document content and nothing was found, explain \
plainly that the indexed documents do not cover that topic. Do not guess.
- Be concise, friendly, and factual. Do not speculate or embellish.
- Answer in the SAME LANGUAGE as the question.
- Citations are added separately after your answer — do not include your \
own citations or source references in the response text.
"""

SUMMARIZE_HISTORY_PROMPT = """\
Summarize this conversation in 2-3 concise sentences. Focus on what topics
the user has asked about so far and what information was provided. Do not
include citations or source references. Write in the same language as the
conversation.
"""


# Cap on the summarization transcript. The verbatim history copy is already
# windowed to MAX_HISTORY_MESSAGES in answerer.py; this bounds the *full*
# transcript the summarizer sees so a long session cannot grow it unbounded.
MAX_SUMMARY_CHARS = 8000

# Per-excerpt and total character caps so a large chunk or a fused pile of
# multi-query/multi-hop results cannot blow past the model's context window.
MAX_EXCERPT_CHARS = 2000
MAX_TOTAL_EXCERPT_CHARS = 12000


def build_history_summary_prompt(history: list[dict]) -> tuple[str, str]:
    """Build (system, user) prompts to summarize a conversation history.

    Returns the summarization system prompt and a formatted transcript of
    user/assistant turns, truncated to MAX_SUMMARY_CHARS (oldest turns first,
    so the most recent context is preserved).
    """
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        label = {"user": "User", "assistant": "Assistant"}.get(role, role)
        lines.append(f"{label}: {content}")
    transcript = "\n\n".join(lines)
    if len(transcript) > MAX_SUMMARY_CHARS:
        transcript = transcript[-MAX_SUMMARY_CHARS:]
    return SUMMARIZE_HISTORY_PROMPT, transcript


def format_excerpts(excerpts: list[tuple[str, str]]) -> str:
    """(source_label, text) pairs as labelled blocks for a prompt body.

    Each excerpt is capped at MAX_EXCERPT_CHARS and the whole body at
    MAX_TOTAL_EXCERPT_CHARS, so oversized chunks or a large fused result set
    cannot overflow the context window.
    """
    blocks: list[str] = []
    total = 0
    for label, text in excerpts:
        if total >= MAX_TOTAL_EXCERPT_CHARS:
            break
        text = text[:MAX_EXCERPT_CHARS]
        block = f"[{label}]\n{text}"
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str,
    excerpts: list[tuple[str, str]],
    *,
    context_summary: str | None = None,
) -> str:
    """excerpts: list of (source_label, text).

    When context_summary is provided it is prepended before the excerpts so
    the model can refer back to earlier questions in the conversation.
    """
    blocks = format_excerpts(excerpts)
    parts: list[str] = []
    if context_summary:
        parts.append(f"Conversation so far:\n{context_summary}")
    parts.append(f"Excerpts:\n\n{blocks}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


GROUNDING_PROMPT = """\
You are a fact-checker. Decide whether an answer is fully supported by the \
provided document excerpts.

The answer is UNSUPPORTED if it states any fact that is:
- absent from the excerpts, or
- contradicted by the excerpts, or
- about the wrong entity, year, or field (a number that belongs to a \
different city, a different year, or a different column).

Ignore wording differences; judge the facts, not the phrasing. If the answer \
correctly says the excerpts lack the information, that is SUPPORTED.

Output exactly one word: SUPPORTED or UNSUPPORTED.
"""


def build_grounding_prompt(
    answer: str, excerpts: list[tuple[str, str]]
) -> tuple[str, str]:
    """Build (system, user) prompts to verify an answer against the excerpts."""
    return GROUNDING_PROMPT, (
        f"Excerpts:\n\n{format_excerpts(excerpts)}\n\nAnswer:\n{answer}"
    )
