"""HybridQA sampled multi-hop golden-set loader.

Fetches HybridQA ``dev.traced.json`` (questions carrying ``answer-node``
evidence traces) plus WikiTables-WithLinks table metadata, and emits a
review-ready sampled golden set where each entry spans the anchor table plus
exactly one linked passage — genuine cross-document multi-hop over tabular +
textual data.

``answer-node`` is distant supervision, not curated evidence: it marks every
cell and every linked passage where the answer *string* happens to occur. Used
raw it yields sources that are not evidence at all (answer "Gothic Revival"
traced to the passage "Renaissance Revival architecture", which only mentions
Gothic Revival to say it is something else). ``clean_trace`` and
``answer_is_unique`` below drop those, so a kept entry's listed passage really
is where the answer lives.

The loader never writes to ``golden_set.yaml``: its output is a DRAFT for
human review, mirroring the ``gen_golden.py`` convention.

Usage:
    python eval/load_hybridqa.py --n 25 --seed 42 --out eval/golden_hybridqa_draft.yaml
"""
import argparse
import random
import re
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
REQUEST_URL = (
    "https://raw.githubusercontent.com/wenhuchen/WikiTables-WithLinks/"
    "master/request_tok/{table_id}.json"
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


def clean_trace(question: dict) -> bool:
    """Do all answer-nodes agree on one linked passage?

    A "table" node means the answer string also sits in a cell, so the
    passage nodes alongside it may be incidental links rather than the hop
    the question describes; several passage titles mean the trace cannot say
    which one the answer came from. Neither is usable as ground truth for
    ``multi_hop_citation_accuracy``, which demands every listed source.
    """
    nodes = question.get("answer-node") or []
    if not nodes or any(node[3] != "passage" for node in nodes):
        return False
    return len({url2title(node[2]) for node in nodes}) == 1


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def answer_is_unique(answer: str, passages: dict, cited: str) -> bool:
    """Is the cited passage the only one in this table carrying the answer?

    When a sibling passage of the same table also contains the answer string,
    an answer citing that sibling looks wrong to the metric while being
    perfectly defensible — the entry would score noise, not capability.
    """
    needle = _squash(answer)
    holders = [url2title(path) for path, text in passages.items()
               if needle in _squash(text)]
    return holders == [cited]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25,
                        help="golden entries to sample")
    parser.add_argument("--seed", type=int, default=42,
                        help="deterministic sample so review is stable")
    parser.add_argument("--out", type=Path,
                        default=Path("eval/golden_hybridqa_draft.yaml"))
    args = parser.parse_args()

    with httpx.Client(follow_redirects=True) as client:
        questions = fetch_json(client, DEV_URL)

    multi = [q for q in questions if clean_trace(q)]

    # Walk a shuffled candidate list rather than sampling a fixed slice: the
    # sibling-passage check needs the table's passages, which only the fetch
    # below has, so rejects have to be replaced as they appear.
    rng = random.Random(args.seed)
    rng.shuffle(multi)

    entries: list[dict] = []
    with httpx.Client(follow_redirects=True) as client:
        for q in multi:
            if len(entries) >= args.n:
                break
            table_id = q["table_id"]
            try:
                table = fetch_json(client, TABLE_URL.format(table_id=table_id))
                passages = fetch_json(client,
                                      REQUEST_URL.format(table_id=table_id))
            except Exception as exc:
                print(f"# skipped {q['question_id']}: {exc}", file=sys.stderr)
                continue
            sources = sources_for(q, table["title"])
            if not answer_is_unique(q["answer-text"], passages, sources[1]):
                continue
            entries.append({
                "question": q["question"],
                "expected_answer": q["answer-text"],
                "expected_sources": sources,
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
        "# Traces are filtered (single passage, answer nowhere else in the "
        "table), but the\n"
        "# questions are not: HybridQA phrases them against a table already "
        "on screen, so\n"
        "# some identify nothing on their own ('A 2009 title came out in what "
        "month ?').\n"
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