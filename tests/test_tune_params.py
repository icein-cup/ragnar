from unittest.mock import patch

import pytest

from eval.tune_params import _run_eval, _run_grid


def test_run_eval_parses_report_json_before_trailing_wrote_line():
    """run_eval.py prints the report JSON, then a trailing
    "wrote eval/reports/<stamp>.json" line (see run_eval.py main()). Passing
    that combined stdout straight to json.loads raises "Extra data" — this
    is the bug that killed every tuning sweep at combo 1."""
    fake_stdout = (
        '{"answer_coverage": 0.5, "refusal_accuracy": 0.9}\n'
        "\nwrote eval/reports/20260825-000000.json\n"
    )
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        run.return_value.stdout = fake_stdout
        row = _run_eval({"top_k": 10}, collection=None, golden=None)

    assert row["answer_coverage"] == 0.5
    assert row["top_k"] == 10


def test_run_grid_override_pins_one_axis_and_leaves_the_module_grid_alone():
    """Phase 1 is coordinate descent (tuning-runbook.md), so a sweep pins
    one axis and holds the other at its default. The override must not
    mutate RETRIEVAL_GRID — the second step reads the untouched grid."""
    from eval.tune_params import RETRIEVAL_GRID
    before = {k: list(v) for k, v in RETRIEVAL_GRID.items()}

    with patch("eval.tune_params._run_eval", return_value=None) as run_eval:
        _run_grid("retrieval", None, None, {"candidates": [30], "top_k": [5, 8]})

    combos = [call.args[0] for call in run_eval.call_args_list]
    assert combos == [{"candidates": 30, "top_k": 5},
                      {"candidates": 30, "top_k": 8}]
    assert RETRIEVAL_GRID == before


def test_run_grid_rejects_an_unknown_axis():
    with pytest.raises(SystemExit):
        _run_grid("retrieval", None, None, {"topk": [5]})
