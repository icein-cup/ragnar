"""HybridQA sampled multi-hop golden-set loader.

Fetches HybridQA ``dev.traced.json`` (questions carrying ``answer-node``
evidence traces) plus WikiTables-WithLinks table metadata, and emits a
review-ready sampled golden set where each entry spans the anchor table plus
one or more linked passages — genuine cross-document multi-hop over tabular +
textual data.

The loader never writes to ``golden_set.yaml``: its output is a DRAFT for
human review, mirroring the ``gen_golden.py`` convention.

Usage:
    python eval/load_hybridqa.py --n 25 --seed 42 --out eval/golden_hybridqa_draft.yaml
"""
import argparse
import random
import sys
from pathlib import Path

import httpx
import yaml

DEV_URL = (
    "https://raw.githubusercontent.com/wenhuchen/HybridQA/master/"
    "released_data/dev.traced.json"
)
TABLE_URL = (
    "https://raw.githubusercontent.com/wenhuchen/WikiTables-WithLinks/"
    "master/tables_tok/{table_id}.json"
)


def url2title(path: str) -> str:
    return path.removeprefix("/wiki/").replace("_", " ")


def fetch_json(client: httpx.Client, url: str):
    resp = client.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def node_source(node, table_title: str) -> str:
    """Citation label a single answer-node evidences.

    node = [answer_text, [row, col], path_or_null, kind]; kind is
    "passage" (path is /wiki/X) or "table" (path is null, source is the
    anchor table itself).
    """
    if node[3] == "table":
        return table_title
    return url2title(node[2])


def sources_for(question: dict, table_title: str) -> list[str]:
    """Ordered, deduplicated source titles: anchor table first, then passages."""
    srcs = [table_title]
    for node in question.get("answer-node") or []:
        src = node_source(node, table_title)
        if src not in srcs:
            srcs.append(src)
    return srcs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25,
                        help="golden entries to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="deterministic sample so review is stable")
    parser.add_argument("--min-hops", type=int, default=2,
                        help="minimum distinct sources per entry "
                             "(table = 1, plus passages)")
    parser.add_argument("--max-sources", type=int, default=None,
                        help="cap distinct sources per entry (table + passages). "
                             "2 gives a clean anchor-table + 1-passage sample.")
    parser.add_argument("--out", type=Path,
                        default=Path("eval/golden_hybridqa_draft.yaml"))
    args = parser.parse_args()

    with httpx.Client(follow_redirects=True) as client:
        questions = fetch_json(client, DEV_URL)

    multi: list[dict] = []
    for q in questions:
        nodes = q.get("answer-node") or []
        if not nodes:
            continue
        passage_titles = sorted({
            url2title(n[2]) for n in nodes if n[3] == "passage"
        })
        # Anchor table counts once; passages are the cross-document hops.
        n_sources = 1 + len(passage_titles)
        if n_sources < args.min_hops:
            continue
        if args.max_sources is not None and n_sources > args.max_sources:
            continue
        multi.append(q)

    rng = random.Random(args.seed)
    sample = rng.sample(multi, min(args.n, len(multi)))

    entries: list[dict] = []
    with httpx.Client(follow_redirects=True) as client:
        for q in sample:
            table_id = q["table_id"]
            try:
                table = fetch_json(client, TABLE_URL.format(table_id=table_id))
            except Exception as exc:
                print(f"# skipped {q['question_id']}: {exc}", file=sys.stderr)
                continue
            entries.append({
                "question": q["question"],
                "expected_answer": q["answer-text"],
                "expected_sources": sources_for(q, table["title"]),
                "out_of_corpus": False,
                "multihop": True,
                "table_id": q["table_id"],
            })

    header = (
        "# HybridQA sampled multi-hop golden set (DRAFT for review).\n"
        "# Source: wenhuchen/HybridQA dev.traced.json + "
        "wenhuchen/WikiTables-WithLinks tables_tok.\n"
        "# multihop: true = the answer requires the anchor table AND every "
        "listed passage.\n"
        "# expected_sources = citation labels (document titles) the answerer "
        "must surface.\n"
        "# Review each entry, then merge accepted ones into eval/golden_set.yaml.\n"
    )
    args.out.write_text(
        header + "\n" + yaml.safe_dump(entries, allow_unicode=True,
                                       sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {args.out}: {len(entries)} entries "
          f"from {len(multi)} multi-hop candidates", file=sys.stderr)


if __name__ == "__main__":
    main()