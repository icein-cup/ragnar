"""The aggregation guard measured against cases built to exercise it.

eval/golden_aggregation.yaml holds 16 questions that all trip
_has_aggregation_intent: 8 that genuinely need summing or counting across rows,
and 8 single-cell lookups phrased with the same vocabulary. The guard is
supposed to refuse the first group and leave the second alone.

These tests do not hit Qdrant. They feed the guard a synthetic result set with
a known table share, which is what isolates the two halves of the decision —
the live measurement (recorded in the YAML header) showed retrieval never
returns a table majority on this corpus, so the guard never fires there and a
live test would only ever confirm "never fires".
"""
from pathlib import Path

import yaml

from core.models import Chunk, SearchResult
from generation.guards import (TABLE_MAJORITY, _has_aggregation_intent,
                               should_refuse_aggregation)

GOLDEN = Path(__file__).parent.parent / "eval" / "golden_aggregation.yaml"
CASES = yaml.safe_load(GOLDEN.read_text())


def _results(n_tables: int, n_prose: int) -> list[SearchResult]:
    out = []
    for i in range(n_tables + n_prose):
        out.append(SearchResult(
            chunk=Chunk(doc_id="d", filename="t.md", text="| a | b |",
                        chunk_index=i, is_table=i < n_tables),
            score=0.7,
        ))
    return out


def test_the_probe_set_stays_balanced():
    kinds = [c["aggregation"] for c in CASES]
    assert kinds.count("true") == 8 and kinds.count("false") == 8


def test_the_vocabulary_separates_real_aggregations_from_lookups():
    """What the narrowing in guards.py bought.

    Every question here is phrased with aggregation vocabulary, so the old
    list matched all 16 and the guard could not tell a sum from a single cell.
    The rule is now scope, not the interrogative: a collective word ("total",
    "all", "combined") or a partitive ("how many OF", "how many DIFFERENT")
    ranges over rows; "how many stores does Netto have" names one entity.
    """
    caught = {c["question"] for c in CASES if _has_aggregation_intent(c["question"])}
    aggregations = {c["question"] for c in CASES if c["aggregation"] == "true"}
    lookups = {c["question"] for c in CASES if c["aggregation"] == "false"}

    assert aggregations - caught == set(), "missed a real aggregation"
    assert lookups & caught == set(), "a single-cell lookup would be refused"


def test_the_guard_refuses_aggregations_and_spares_lookups_on_a_table_heavy_set():
    """End to end, with the table share the guard needs in order to fire."""
    table_heavy = _results(n_tables=4, n_prose=1)
    refused = {c["question"] for c in CASES
               if should_refuse_aggregation(c["question"], table_heavy)}

    assert len(refused) == 8
    assert all(c["aggregation"] == "true"
               for c in CASES if c["question"] in refused)


def test_the_guard_stays_silent_at_the_table_share_retrieval_actually_returns():
    """Measured live: 1 table chunk in 5 (0.20) even for questions written to
    target a specific table. That is below TABLE_MAJORITY, so nothing fires."""
    assert 0.20 <= TABLE_MAJORITY
    realistic = _results(n_tables=1, n_prose=4)
    assert not any(should_refuse_aggregation(c["question"], realistic)
                   for c in CASES)
