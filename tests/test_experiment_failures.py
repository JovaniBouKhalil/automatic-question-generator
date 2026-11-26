"""A failed generator must never inherit the gold question as its prediction."""

import json
from types import SimpleNamespace

import pytest

from questionforge import experiment
from questionforge.data import write_jsonl


@pytest.mark.parametrize("failure", ["generation", "verification"])
def test_execution_errors_do_not_copy_reference_or_discard_a_real_prediction(tmp_path, monkeypatch, failure):
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [{"id": "one", "context": "Lina planted seeds.", "question": "Who planted seeds?",
                        "answer": "Lina", "answer_start": 0, "language": "en", "group": "one", "source": "test"}])
    monkeypatch.setattr(experiment, "load_verifier", lambda *args: None)
    monkeypatch.setattr(experiment, "load_generator", lambda *args: None)

    def generate(*args, **kwargs):
        if failure == "generation":
            raise RuntimeError("injected generation failure")
        return SimpleNamespace(question="Which person planted the seeds?", cropped=False, input_tokens=12, model_id="stub")

    def verify(*args, **kwargs):
        if failure == "verification":
            raise RuntimeError("injected verification failure")
        return {"qa_answer": "", "qa_score": 0, "answer_f1": 0, "answer_em": 0}

    monkeypatch.setattr(experiment, "generate_question", generate)
    monkeypatch.setattr(experiment, "verify_question", verify)
    output = tmp_path / "results"
    summary = experiment.run_experiment(path, output, methods=["beam4"], languages=["en"])
    rows = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    predicted = next(row for row in rows if row["method"] == "beam4")
    assert predicted["question"] == ("" if failure == "generation" else "Which person planted the seeds?")
    assert predicted["question"] != predicted["reference"]
    assert predicted["error"] and predicted["answer_f1"] == 0
    metrics = next(row for row in summary if row["method"] == "beam4")
    assert metrics["n"] == 1 and metrics["errors"] == 1
