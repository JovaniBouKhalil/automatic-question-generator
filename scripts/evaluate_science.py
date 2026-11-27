"""Evaluate the validation-selected science checkpoint without reselection on test."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from questionforge.data import load_jsonl
from questionforge.experiment import bootstrap_mean_ci, fingerprint_file
from questionforge.metrics import aggregate
from questionforge.models import generate_question, verify_question
from scripts.run_learning_focus_study import snapshot_sources, source_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()
    selection_path = Path("results/science_training/selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["complete"] and selection["trials"] == 8 and not selection["test_used_for_selection"]
    sources = [Path(__file__).relative_to(Path.cwd()), Path("docs/SCIENCE_TRAINING_PROTOCOL.md"),
               *Path("questionforge").glob("*.py")]
    data_path = Path("data/processed/science") / f"{args.split}.jsonl"
    checkpoint = selection["selected_checkpoint"]
    output = Path(f"results/science_evaluation_{args.split}")
    output.mkdir(parents=True, exist_ok=True)
    config = {"split": args.split, "limit": 64 if args.split == "dev" else 200,
              "selection_sha256": fingerprint_file(selection_path), "data_sha256": fingerprint_file(data_path),
              "selected_checkpoint": checkpoint, "beams": 4, "seed": 42,
              "checkpoint_files": {p.name: fingerprint_file(p) for p in Path(checkpoint).iterdir() if p.is_file()} if checkpoint else None,
              "sources_sha256_normalized_utf8": {p.as_posix(): source_hash(p) for p in sources},
              "test_used_for_selection": False}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Science evaluation changed. Preserve the current folder and choose another version.")
    write_json(config_path, config)
    snapshot_sources(output, sources)
    records = load_jsonl(data_path)[:config["limit"]]
    path = output / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done = {(r["id"], r["method"]) for r in rows}
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            for method in ["reference_control", "original_t5", "selected_t5"]:
                if (record["id"], method) in done:
                    continue
                started = time.perf_counter()
                row = {**record, "reference": record["question"], "method": method, "error": None}
                try:
                    if method != "reference_control":
                        generated = generate_question(record["context"], record["answer"], "en", answer_start=record["answer_start"],
                                                      beams=4, checkpoint=checkpoint if method == "selected_t5" else None)
                        row.update(asdict(generated), method=method)
                    row.update(verify_question(row["question"], record["context"], record["answer"], "en"))
                except Exception as exc:
                    row.update(error=f"{type(exc).__name__}: {exc}", question="", answer_f1=0., answer_em=0., qa_answer="", qa_score=0.)
                row["seconds"] = time.perf_counter() - started
                # Unicode line separators in source passages must not split JSONL records.
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                rows.append(row)
            if len(rows) % 30 == 0:
                print(args.split, f"{len(rows)}/{len(records) * 3} saved predictions", flush=True)
    summary = {"scope": f"SciQ {args.split}; validation-only checkpoint selection, no completed human usefulness ratings.",
               "question_answer_pairs": len(records), "prediction_records": len(rows), "methods": {},
               "cases_with_execution_error": sum(bool(r["error"]) for r in rows), "human_usefulness_ratings": "pending"}
    for method in ["reference_control", "original_t5", "selected_t5"]:
        selected = [r for r in rows if r["method"] == method]
        summary["methods"][method] = {**aggregate(selected), "qa_f1_ci95": bootstrap_mean_ci(selected)}
    originals = {r["id"]: r for r in rows if r["method"] == "original_t5"}
    differences = [{"group": r["group"], "answer_f1": r["answer_f1"] - originals[r["id"]]["answer_f1"]}
                   for r in rows if r["method"] == "selected_t5"]
    summary["paired_qa_answer_f1_difference_pp"] = {"mean": 100 * sum(r["answer_f1"] for r in differences) / len(differences),
                                                  "ci95": bootstrap_mean_ci(differences)}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
