"""LLM-assisted golden-set draft generator.

Reads the converted markdown of ingested documents and asks the eval judge
to produce candidate golden-set entries: a question, the expected answer,
and the verbatim source excerpt the answer must be grounded in. The output
is a DRAFT — every entry must be human-reviewed before it lands in
golden_set.yaml. The generator never writes directly to golden_set.yaml
for exactly that reason.

Usage:
    python eval/gen_golden.py --docs doc1 doc2 ... > eval/golden_draft.yaml
    # review, edit, then manually merge into eval/golden_set.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from core.config import Config

DRAFT_PROMPT = """\
You are helping build an evaluation set for a retrieval-augmented document
question-answering system.

For the document excerpt below, produce {n} question/answer/reference
entries. Rules:
- The question must be answerable ONLY from the excerpt.
- The answer must be a short, factual statement pulled from the excerpt.
- The reference must be a verbatim copy of the sentence(s) from the excerpt
  that support the answer. Do not paraphrase the reference.
- Vary difficulty: some direct lookup, some requiring connecting two facts.
- Preserve the original language of the excerpt.

Respond as a JSON array of objects, each with keys: question, answer, reference.
"""


def build_client() -> OpenAI:
    base_url = os.environ.get("RAGAS_JUDGE_BASE_URL", "").strip()
    api_key = os.environ.get("RAGAS_JUDGE_API_KEY", "").strip()
    if not base_url or not api_key:
        raise SystemExit(
            "RAGAS_JUDGE_BASE_URL and RAGAS_JUDGE_API_KEY must be set in the "
            "environment (see .env.example)."
        )
    return OpenAI(base_url=base_url, api_key=api_key)


def _extract_json(text: str) -> str:
    """Strip markdown code fences and leading/trailing non-JSON text."""
    import re
    # Try ```json ... ``` first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try bare ``` ... ```
    m = re.search(r"```\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: find first [ to last ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def read_excerpt(storage, doc_id: str, max_chars: int) -> str | None:
    markdown = storage.read_markdown(doc_id)
    if not markdown:
        return None
    return markdown[:max_chars]



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", required=True,
                        help="source doc_ids (from the registry) to draft from")
    parser.add_argument("--n", type=int, default=5,
                        help="entries to draft per document")
    parser.add_argument("--max-chars", type=int, default=12000,
                        help="max chars of document markdown to send per call")
    parser.add_argument("--model", default=None,
                        help="override the judge model")
    args = parser.parse_args()

    cfg = Config()
    from ingestion.storage import Storage
    storage = Storage(cfg.data_dir)

    client = build_client()
    model = args.model or os.environ.get("RAGAS_JUDGE_MODEL", "glm-5.2:cloud")

    entries = []
    for doc_id in args.docs:
        excerpt = read_excerpt(storage, doc_id, args.max_chars)
        if not excerpt:
            print(f"# skipped {doc_id}: markdown not found", file=sys.stderr)
            continue

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DRAFT_PROMPT.format(n=args.n)},
                {"role": "user", "content": excerpt},
            ],
        )
        text = response.choices[0].message.content
        cleaned = _extract_json(text)
        try:
            batch = json.loads(cleaned)
        except json.JSONDecodeError:
            print(f"# skipped {doc_id}: judge returned non-JSON", file=sys.stderr)
            print(f"# raw response:\n{text}", file=sys.stderr)
            continue

        for item in batch:
            item["source_doc"] = doc_id
            item["out_of_corpus"] = False
            entries.append(item)

    print(json.dumps(entries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()