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


def build_history_summary_prompt(history: list[dict]) -> tuple[str, str]:
    """Build (system, user) prompts to summarize a conversation history.

    Returns the summarization system prompt and a formatted transcript of
    user/assistant turns.
    """
    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        label = {"user": "User", "assistant": "Assistant"}.get(role, role)
        lines.append(f"{label}: {content}")
    return SUMMARIZE_HISTORY_PROMPT, "\n\n".join(lines)


def format_excerpts(excerpts: list[tuple[str, str]]) -> str:
    """(source_label, text) pairs as labelled blocks for a prompt body."""
    return "\n\n".join(f"[{label}]\n{text}" for label, text in excerpts)


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
