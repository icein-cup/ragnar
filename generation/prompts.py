from generation.guards import NO_ANSWER

# The NO_ANSWER rule below is a contract with generation.guards.is_refusal,
# which drives the refused flag, whether the UI attaches citations, and the
# refusal_accuracy metric. Before it existed, that decision rested on regex
# matching whatever phrases the model happened to reach for - so swapping
# models silently moved the metric. Keep the two in step.
#
# The "answer whenever you can" rule is not padding; it is load-bearing, and
# was measured on qwen2.5:7b over 40 in-corpus cases and 20 probes
# (eval/replay_answer.py):
#
#   no sentinel, "say so plainly"          accuracy 0.60   refusal 0.85
#   sentinel + "explain briefly why"       accuracy 0.50   refusal 0.95
#   sentinel + this pro-answer rule        accuracy 0.62   refusal 0.95
#
# Giving the model a token for declining makes declining easier to reach for.
# Asking it to justify a refusal made that worse still - four correct answers
# lost to buy two correct refusals. Saying outright that answering is
# preferred is what pays for the token. Re-measure before loosening it.
SYSTEM_PROMPT = f"""\
You answer questions strictly from the provided document excerpts.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Answer whenever the excerpts contain the answer — including when it takes \
combining two excerpts, or when their wording differs from the question's. \
Do not decline a question the excerpts can answer.
- Only when the excerpts genuinely do not contain the answer, say so plainly, \
starting your reply with {NO_ANSWER}. Do not guess.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language. The {NO_ANSWER} token itself is never translated.
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


# Measured with eval/replay_gate.py against eval/reports/20260824-132940.json
# (qwen2.5:7b judging its own answers):
#
#                          bad answers caught   correct answers lost
#   without the accept rule       9/10                 16/25
#   with it (below)               9/20                  7/25
#
# Stating when to ACCEPT more than halved the false rejections — the prompt
# used to list three ways to fail and no way to pass, so the model rejected by
# default. It is still not good enough to gate on: losing 28% of correct
# answers to catch 45% of bad ones is a bad trade while over-refusal is
# already the pipeline's biggest problem (28 of 90 in-corpus questions refused
# in that same run). _grounded therefore stays advisory — see its docstring.
GROUNDING_PROMPT = """\
You are a fact-checker. Decide whether an answer is fully supported by the \
provided document excerpts.

The answer is UNSUPPORTED if it states any fact that is:
- absent from the excerpts, or
- contradicted by the excerpts, or
- about the wrong entity, year, or field (a number that belongs to a \
different city, a different year, or a different column).

The answer is SUPPORTED if every fact it states appears in the excerpts. \
Partial answers are SUPPORTED — judge only what the answer claims, not what \
it leaves out. An answer that correctly says the excerpts lack the \
information is SUPPORTED. When in doubt, prefer SUPPORTED: a wrongly \
rejected answer costs the user a correct reply.

Ignore wording differences; judge the facts, not the phrasing.

Output exactly one word: SUPPORTED or UNSUPPORTED.
"""


def build_grounding_prompt(
    answer: str, excerpts: list[tuple[str, str]]
) -> tuple[str, str]:
    """Build (system, user) prompts to verify an answer against the excerpts."""
    return GROUNDING_PROMPT, (
        f"Excerpts:\n\n{format_excerpts(excerpts)}\n\nAnswer:\n{answer}"
    )
