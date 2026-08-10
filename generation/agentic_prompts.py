"""Agentic RAG prompts for query rewriting, multi-query generation, multi-hop reasoning, and self-correction."""

# ── Query Rewriting ──────────────────────────────────────────────────────────
QUERY_REWRITE_PROMPT = """\
You are a query optimization assistant for a document retrieval system.

Your task: rewrite the user's question into a more precise, retrieval-friendly query.
Rules:
- Expand abbreviations and resolve ambiguous references using conversation context.
- Break down complex questions into explicit sub-questions if needed.
- Use keywords and phrases likely to appear in technical/legal documents.
- Preserve the original language.
- Output ONLY the rewritten query — no explanations, no quotes.
"""

# ── Multi-Query Generation ───────────────────────────────────────────────────
MULTI_QUERY_PROMPT = """\
You are a query diversification assistant for a document retrieval system.

Your task: generate {num_queries} different ways to ask the same question, covering:
- Different phrasings and synonyms
- Different angles or aspects of the question
- More specific vs. more general versions
- Technical vs. plain-language versions

Output exactly one query per line, no numbering, no bullets, no extra text.
"""

# ── Multi-Hop Reasoning ────────────────────────────────────────────────────────
MULTI_HOP_PROMPT = """\
You are analyzing retrieved document excerpts to answer a user's question.

Current question: {question}

Retrieved excerpts so far:
{excerpts}

Based on what we've found, assess:
1. Do we have enough information to fully answer the question? (yes/no/partial)
2. What key information is still missing?
3. What follow-up query would help find the missing pieces?

Output in this exact format:
Sufficient: <yes|no|partial>
Missing: <description of what's missing, or "none">
FollowUp: <follow-up query, or "none">
"""

# ── Self-Correction / Answer Evaluation ───────────────────────────────────────
SELF_CORRECTION_PROMPT = """\
You are evaluating whether an answer is complete and accurate based on retrieved document excerpts.

Question: {question}

Retrieved excerpts:
{excerpts}

Draft answer:
{answer}

Assess:
1. Does the answer fully address the question using only the excerpts? (yes/no/partial)
2. Are there contradictions or unsupported claims in the answer? (yes/no)
3. What additional information would improve the answer?

Output in this exact format:
Complete: <yes|no|partial>
Contradictions: <yes|no>
Improvement: <description of what's needed, or "none">
"""

def build_rewrite_prompt(question: str, context_summary: str | None = None) -> tuple[str, str]:
    """Build (system, user) prompts for query rewriting."""
    user = question
    if context_summary:
        user = f"Conversation context:\n{context_summary}\n\nQuestion: {question}"
    return QUERY_REWRITE_PROMPT, user


def build_multi_query_prompt(question: str, num_queries: int = 3) -> tuple[str, str]:
    """Build (system, user) prompts for generating multiple query variants."""
    return MULTI_QUERY_PROMPT.format(num_queries=num_queries), question


def build_multi_hop_prompt(question: str, excerpts: list[tuple[str, str]]) -> tuple[str, str]:
    """Build (system, user) prompts for multi-hop reasoning."""
    blocks = "\n\n".join(f"[{label}]\n{text}" for label, text in excerpts)
    return "", MULTI_HOP_PROMPT.format(question=question, excerpts=blocks)


def build_self_correction_prompt(question: str, excerpts: list[tuple[str, str]], answer: str) -> tuple[str, str]:
    """Build (system, user) prompts for self-correction evaluation."""
    blocks = "\n\n".join(f"[{label}]\n{text}" for label, text in excerpts)
    return "", SELF_CORRECTION_PROMPT.format(question=question, excerpts=blocks, answer=answer)
