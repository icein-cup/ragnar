from unittest.mock import patch

from eval.tune_params import _run_eval


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
