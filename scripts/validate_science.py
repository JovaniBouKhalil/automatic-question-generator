"""Audit science split provenance, all trials, selection and paired evaluation records."""

from __future__ import annotations

import json
from pathlib import Path

from questionforge.data import assert_disjoint, load_jsonl
from questionforge.experiment import fingerprint_file
from scripts.run_learning_focus_study import source_hash, write_json


def main():
    root = Path("results/science_training")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    for name, digest in config["sources_sha256"].items():
        # The first archived trainer records its original absolute Windows source path.
        relative = "scripts/" + name.split("/scripts/", 1)[1] if "/scripts/" in name else name
        assert fingerprint_file(root / "source" / relative) == digest, relative
    trials = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("lr*.json")]
    assert len(trials) == 8 and len({r["trial"] for r in trials}) == 8
    assert len({r["initial_dev_loss"] for r in trials}) == 1
    for r in trials:
        assert r["complete"] and r["updates"] == 256 and not r["test_used_for_training_or_selection"]
        checks = [h for h in r["history"] if "dev_loss" in h]
        best = min(checks, key=lambda h: (h["dev_loss"], h["step"]))
        assert (r["best_dev_loss"], r["best_step"]) == (best["dev_loss"], best["step"])
        assert fingerprint_file(Path(r["adapter"]) / "adapter_model.safetensors") == r["adapter_sha256"]
    winner = min(trials, key=lambda r: (r["best_dev_loss"], r["best_step"], r["rank"]))
    assert winner["trial"] == selection["selected_trial"] and not selection["test_used_for_selection"]
    manifest = json.loads((root / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepared_sha256"] == config["data_sha256"]
    assert manifest["all_published_heldout_support_and_questions_protected"]
    data = Path("data/processed/science")
    if data.exists():
        partitions = {s: load_jsonl(data / f"{s}.jsonl") for s in ["train", "dev", "test"]}
        assert_disjoint(partitions)
        for s in partitions:
            assert fingerprint_file(data / f"{s}.jsonl") == config["data_sha256"][s]
    counts = {}
    for split, expected in [("dev", 64), ("test", 200)]:
        directory = Path(f"results/science_evaluation_{split}")
        evaluation = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        assert evaluation["selection_sha256"] == fingerprint_file(root / "selection.json")
        for name, digest in evaluation["sources_sha256_normalized_utf8"].items():
            assert source_hash(directory / "source" / name) == digest
        for name, digest in (evaluation["checkpoint_files"] or {}).items():
            assert fingerprint_file(Path(evaluation["selected_checkpoint"]) / name) == digest
        rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(rows) == expected * 3 and len({(r["id"], r["method"]) for r in rows}) == len(rows)
        assert not any(r["error"] for r in rows)
        for row in rows:
            start, answer = row["answer_start"], row["answer"]
            assert row["context"][start:start + len(answer)] == answer and 0 <= row["answer_f1"] <= 1
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        assert summary["prediction_records"] == len(rows) and summary["question_answer_pairs"] == expected
        counts[split] = len(rows)
    receipt = {"completed_trials": 8, "trained_updates_per_trial": 256, "selected_trial": winner["trial"],
               "source_and_adapter_hashes_checked": True, "validation_only_selection_verified": True,
               "prediction_records": counts, "execution_errors": 0, "human_usefulness_ratings": "pending",
               "interpretation": "Artifact consistency does not establish factual support or educational value."}
    write_json(Path("results/quality/science_validation.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
