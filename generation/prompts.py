import re

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
#
# The step-by-step reasoning block (steps 1-4) and the stronger anti-refusal
# language were added after an eval run showed 28 of 90 in-corpus questions
# refused and 18 of 62 answered questions wrong. The reasoning steps force
# the model to check every excerpt before deciding, and step 3 explicitly
# addresses multi-hop synthesis. The NO_ANSWER sentinel contract is
# preserved: when refusing, the model starts its reply with NO_ANSWER; when
# answering, it outputs the answer directly.
SYSTEM_PROMPT = f"""\
You answer questions strictly from the provided document excerpts.

Before answering, work through these steps:
1. Identify exactly what the question asks — the entity, the property, the \
time frame, and any implicit sub-questions.
2. Check EVERY excerpt one by one. Look for the answer even when the wording \
differs from the question, when the information is indirect, or when it is \
split across multiple excerpts.
3. If the answer requires combining facts from two or more excerpts, piece \
them together: one excerpt may name the entity, another may give the value, \
and a third may provide the date. Synthesize across all excerpts that are \
relevant.
4. Only after you have checked every excerpt, decide: can the question be \
answered from the excerpts alone?

If yes, give a concise, factual answer. Do not speculate or embellish.
If no — you have genuinely checked every excerpt and none contains the \
answer, even indirectly — start your reply with {NO_ANSWER} and briefly \
state what is missing. Do not guess.

Rules:
- Use ONLY information in the excerpts. Never use outside knowledge.
- Default to answering. Refusing is the last resort, not the first. Most \
questions that seem unanswered at first glance CAN be answered by combining \
or carefully reading the excerpts. Read difficult passages slowly and look \
for indirect mentions, synonyms, and information that implies the answer.
- When the answer requires synthesizing across excerpts, combine the facts \
explicitly. Do not give up because no single excerpt contains the full answer.
- Answer in the SAME LANGUAGE as the question, even when the excerpts are \
in a different language. The {NO_ANSWER} token itself is never translated.
- Be concise and factual.
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


def _table_to_sentences(text: str) -> str:
    """Convert markdown table rows to natural-language "Header: Value" sentences.

    Lines starting with ``|`` are treated as markdown table rows. The first
    such row is the header; subsequent rows are data rows. Separator lines
    (``|---|---|``) are skipped. Each data row becomes a single line of
    ``"Header: Value | Header: Value | ..."`` pairs. Non-table lines pass
    through unchanged.

    Example::

        | City | Population | Year |
        |------|-----------|------|
        | Multan | 1871843 | 2017 |

    becomes::

        City: Multan | Population: 1871843 | Year: 2017
    """
    lines = text.split("\n")
    headers: list[str] | None = None
    out: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            out.append(line)
            continue

        # Split on | and discard the leading/trailing empty cells produced by
        # the leading/trailing pipes.
        cells = [c.strip() for c in stripped.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        # Skip separator rows like |---|---|
        if all(re.match(r"^:?-+:?$", c) for c in cells if c):
            continue

        if headers is None:
            headers = cells
            continue

        if headers and len(cells) == len(headers):
            pairs = [f"{h}: {v}" for h, v in zip(headers, cells)]
            out.append(" | ".join(pairs))
        else:
            # Malformed row — keep the original line.
            out.append(line)

    return "\n".join(out)


def format_excerpts(excerpts: list[tuple[str, str]]) -> str:
    """(source_label, text) pairs as labelled blocks for a prompt body.

    Each excerpt is capped at MAX_EXCERPT_CHARS and the whole body at
    MAX_TOTAL_EXCERPT_CHARS, so oversized chunks or a large fused result set
    cannot overflow the context window.

    Markdown table rows in each excerpt are converted to natural-language
    "Header: Value" sentences via :func:`_table_to_sentences` so that a 7B
    model can bind entities to values without misreading a table grid.
    """
    blocks: list[str] = []
    total = 0
    for label, text in excerpts:
        if total >= MAX_TOTAL_EXCERPT_CHARS:
            break
        text = _table_to_sentences(text)
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
