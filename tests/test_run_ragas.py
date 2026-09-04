import json

from eval.run_ragas import build_samples


def test_build_samples_from_report_maps_fields_and_skips_refused(tmp_path):
    """--report reads a saved eval/run_eval.py report instead of re-running
    the pipeline. This pins the field mapping and the two skip conditions
    (out_of_corpus, refused) against a report shape, not a live run."""
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "cases": [
            {
                "question": "When was the bridge built?",
                "expected_answer": "1932",
                "out_of_corpus": False,
                "refused": False,
                "answer": "The bridge was built in 1932.",
                "contexts": ["The bridge opened in 1932."],
            },
            {
                "question": "What is the capital of Mars?",
                "expected_answer": "",
                "out_of_corpus": True,
                "refused": True,
                "answer": "",
                "contexts": [],
            },
        ],
    }))

    samples = build_samples(report=report)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.user_input == "When was the bridge built?"
    assert sample.response == "The bridge was built in 1932."
    assert sample.reference == "1932"
    assert sample.retrieved_contexts == ["The bridge opened in 1932."]
