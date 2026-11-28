"""Audit the sealed data separation, frozen inference sources and complete result records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from questionforge.classroom.material import prepare_material
from questionforge.data import context_hash, load_jsonl
from scripts.prepare_generalization_holdout import key
from scripts.run_learning_focus_study import source_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    root = Path("results/generalization_v1")
    manifest, config, sealed = (read(root / name) for name in
                                ["data_manifest.json", "config.json", "sealed_protection.json"])
    protected = set(sealed["paragraph_hashes"])
    questions = set(sealed["reference_question_hashes"])
    titles = set(map(tuple, sealed["known_titles"]))
    for name, digest in config["source_sha256_normalized_utf8"].items():
        assert source_hash(root / "source" / name) == digest, name
        assert source_hash(name) == digest, (name, "frozen inference source changed during test")
    for name, expected in config["checkpoint_files"].items():
        assert hashlib.sha256(Path(config["science_checkpoint"], name).read_bytes()).hexdigest() == expected
    assert source_hash(root / "data_manifest.json") == config["data_manifest_sha256"]
    automatic, science = (load_jsonl(root / f"{name}_data.jsonl") for name in ["automatic", "science"])
    assert len(automatic) == 12 and len(science) == 50
    for name in ["automatic", "science"]:
        assert source_hash(root / f"{name}_data.jsonl") == manifest["data_sha256"][name]
    for row in [*automatic, *science]:
        assert context_hash(row["context"]) not in protected
        assert hashlib.sha256(key(row["question"]).encode()).hexdigest() not in questions
        title = key(row.get("title", ""))
        if title:
            assert (row["language"], title) not in titles
    assert len({context_hash(row["context"]) for row in [*automatic, *science]}) == 62
    for language in ["en", "fr", "ar"]:
        chosen = [r for r in automatic if r["language"] == language]
        assert len(chosen) == 4
        known = [key(r.get("title", "")) for r in chosen if r.get("title")]
        assert len(known) == len(set(known))
        if language == "ar":
            assert all("arabic-" in r["id"] for r in chosen)
    cases = []
    for number, source in enumerate(automatic, 1):
        case = read(root / "automatic" / f"case_{number:02d}_{source['language']}.json")
        assert case["context"] == prepare_material(source["slide_text"])
        assert case["original_human_context"] == source["context"] and case["case_id"] == source["id"]
        for method, rows in case["outputs"].items():
            for row in rows:
                start, answer = row["answer_start"], row["answer"]
                assert case["context"][start:start + len(answer)] == answer, (number, method)
        cases.append(case)
    predictions = [json.loads(s) for s in (root / "science_predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == len({(r["id"], r["method"]) for r in predictions}) == 150
    assert {r["id"] for r in predictions} == {r["id"] for r in science}
    auto_summary, science_summary = read(root / "automatic_summary.json"), read(root / "science_summary.json")
    assert auto_summary["complete"] and auto_summary["case_records"] == 12
    assert auto_summary["execution_errors"] == sum(bool(c["error"]) for c in cases)
    assert science_summary["execution_errors"] == sum(bool(r["error"]) for r in predictions)
    assert not config["test_used_for_tuning"] and not manifest["test_used_for_tuning"]
    receipt = {"automatic_cases": 12, "science_fixed_answer_cases": 50,
               "science_prediction_rows": 150, "sealed_paragraph_and_question_overlap": 0,
               "known_article_overlap": 0, "french_article_identity": "unavailable",
               "frozen_source_snapshots_and_current_hashes_match": True,
               "automatic_execution_errors": auto_summary["execution_errors"],
               "science_execution_errors": science_summary["execution_errors"],
               "upstream_checkpoint_exposure": "unknown", "test_used_for_tuning": False,
               "interpretation": "Project-level data separation and provenance checked; this is not proof of question correctness or absence of upstream public-data exposure."}
    write_json(Path("results/quality/generalization_validation.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
