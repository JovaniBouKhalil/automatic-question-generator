"""Evaluate the frozen classroom workflow and selected T5 adapter on sealed holdout rows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from questionforge.classroom.material import (
    CLASSROOM_PLAN_SYSTEM,
    build_classroom_candidates,
    list_scope_issue,
    prepare_material,
)
from questionforge.data import load_jsonl
from questionforge.experiment import bootstrap_mean_ci, environment_info, fingerprint_file
from questionforge.learning.quality import QUALITY_MODEL, create_quality_quiz
from questionforge.metrics import aggregate
from questionforge.models import generate_question, verify_question
from scripts.run_learning_focus_study import (
    diagnostics,
    render_report,
    snapshot_sources,
    source_hash,
    write_json,
)

ROOT = Path("results/generalization_v1")


def science_audit(checkpoint):
    data = load_jsonl(ROOT / "science_data.jsonl")
    path = ROOT / "science_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    done = {(r["id"], r["method"]) for r in rows}
    with path.open("a", encoding="utf-8") as stream:
        for number, record in enumerate(data, 1):
            for method in ["reference_control", "original_t5", "selected_t5"]:
                if (record["id"], method) in done:
                    continue
                row = {**record, "reference": record["question"], "method": method, "error": None}
                start = time.perf_counter()
                try:
                    if method != "reference_control":
                        row.update(asdict(generate_question(record["context"], record["answer"], "en",
                                   answer_start=record["answer_start"], beams=4,
                                   checkpoint=checkpoint if method == "selected_t5" else None)), method=method)
                    row["generation_seconds"] = time.perf_counter() - start
                    row.update(verify_question(row["question"], record["context"], record["answer"], "en"))
                except Exception as exc:
                    row.update(error=f"{type(exc).__name__}: {exc}", question="", qa_answer="", qa_score=0., answer_f1=0., answer_em=0.)
                row["seconds"] = time.perf_counter() - start
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
                stream.flush()
                rows.append(row)
            print(f"Unused SciQ test: {number}/{len(data)} pairs saved", flush=True)
    summary = {"question_answer_pairs": len(data), "prediction_records": len(rows),
               "execution_errors": sum(bool(r["error"]) for r in rows), "test_used_for_selection": False,
               "methods": {method: aggregate([r for r in rows if r["method"] == method]) for method in
                           ["reference_control", "original_t5", "selected_t5"]}}
    before = {r["id"]: r for r in rows if r["method"] == "original_t5"}
    delta = [{"group": r["group"], "answer_f1": r["answer_f1"] - before[r["id"]]["answer_f1"]}
             for r in rows if r["method"] == "selected_t5"]
    summary["paired_qa_answer_f1_difference_pp"] = {
        "mean": 100 * sum(r["answer_f1"] for r in delta) / len(delta), "ci95": bootstrap_mean_ci(delta)}
    write_json(ROOT / "science_summary.json", summary)


def automatic_audit():
    records = load_jsonl(ROOT / "automatic_data.jsonl")
    directory = ROOT / "automatic"
    directory.mkdir(exist_ok=True)
    cases = []
    for number, record in enumerate(records, 1):
        path = directory / f"case_{number:02d}_{record['language']}.json"
        if path.exists():
            cases.append(json.loads(path.read_text(encoding="utf-8")))
            continue
        print(f"Sealed automatic holdout {number}/{len(records)}: {record['language']}", flush=True)
        source = prepare_material(record["slide_text"])
        case = {"case_id": record["id"], "language": record["language"], "context": source,
                "original_human_context": record["context"], "layout_variant": record["slide_text"],
                "source": record["source"], "title": record.get("title", ""),
                "human_reference_question_not_given_to_planner": record["question"],
                "outputs": {}, "error": None}
        start = time.perf_counter()
        try:
            returned, report = create_quality_quiz(source, record["language"], 3, objective="",
                                      candidate_builder=build_classroom_candidates, planning_system=CLASSROOM_PLAN_SYSTEM,
                                      progress=lambda s: print(s, flush=True))
            case["planning_report"] = report
            case["outputs"]["planned_t5"] = [diagnostics(r, source, record["language"]) for r in report["drafts"]]
            case["scope_rejected"] = []
            kept = []
            for row in returned:
                issue = list_scope_issue(row["question"], row["answer"], record["language"])
                if issue:
                    case["scope_rejected"].append({"row": row, "reason": issue})
                else:
                    kept.append(diagnostics(row, source, record["language"]))
            case["outputs"]["classroom_hybrid"] = kept
        except Exception as exc:
            case["error"] = f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "raw_text"):
                case["failed_model_response"] = exc.raw_text
            if hasattr(exc, "learning_report"):
                case["planning_report"] = exc.learning_report
            print(case["error"], flush=True)
        case["seconds"] = time.perf_counter() - start
        write_json(path, case)
        cases.append(case)
        render_report(directory, cases, methods=["planned_t5", "classroom_hybrid"])
    summary = {"case_records": len(cases), "planned_case_records": len(records), "complete": len(cases) == len(records),
               "execution_errors": sum(bool(c["error"]) for c in cases),
               "requested_maximum_questions": len(records) * 3, "test_used_for_tuning": False,
               "human_quality_ratings": "pending", "methods": {}}
    for method in ["planned_t5", "classroom_hybrid"]:
        rows = [r for c in cases for r in c["outputs"].get(method, [])]
        summary["methods"][method] = {"returned_questions": len(rows),
                    "mean_qa_answer_f1_diagnostic": sum(r["answer_f1"] for r in rows) / len(rows) if rows else None}
    render_report(directory, cases, write_template=True, methods=["planned_t5", "classroom_hybrid"])
    write_json(ROOT / "automatic_summary.json", summary)


def main():
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    assert source_hash("docs/GENERALIZATION_PROTOCOL.md") == manifest["protocol_sha256"]
    for name in ["automatic", "science"]:
        assert source_hash(ROOT / f"{name}_data.jsonl") == manifest["data_sha256"][name]
    selected = json.loads(Path("results/science_training/selection.json").read_text(encoding="utf-8"))
    checkpoint = selected["selected_checkpoint"]
    files = [Path("docs/GENERALIZATION_PROTOCOL.md"), Path(__file__).relative_to(Path.cwd()),
             Path("scripts/prepare_generalization_holdout.py"), Path("scripts/run_learning_focus_study.py"),
             *Path("questionforge").rglob("*.py")]
    config = {"seed": manifest["seed"], "beams": 4, "device": "cpu", "precision": "none-float32",
              "planner_model": asdict(QUALITY_MODEL), "science_checkpoint": checkpoint,
              "science_selection_sha256": source_hash("results/science_training/selection.json"),
              "checkpoint_files": {p.name: fingerprint_file(p) for p in Path(checkpoint).iterdir() if p.is_file()},
              "source_sha256_normalized_utf8": {p.as_posix(): source_hash(p) for p in files},
              "data_manifest_sha256": source_hash(ROOT / "data_manifest.json"), "test_used_for_tuning": False}
    target = ROOT / "config.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != config:
        raise ValueError("Frozen source/config changed. Preserve the run rather than reuse this holdout for tuning.")
    write_json(target, config)
    snapshot_sources(ROOT, files)
    if not (ROOT / "environment.json").exists():
        write_json(ROOT / "environment.json", environment_info())
    science_audit(checkpoint)
    automatic_audit()
    print("Both sealed audits complete; inspect every result without retuning on this test.", flush=True)


if __name__ == "__main__":
    main()
