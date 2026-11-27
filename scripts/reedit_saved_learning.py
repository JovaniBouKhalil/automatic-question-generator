"""Paired development re-edit: hold saved facts, offsets and T5 drafts fixed."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

from questionforge.experiment import environment_info
from scripts.run_learning_focus_study import (
    diagnostics,
    render_report,
    snapshot_sources,
    source_hash,
    write_json,
)


def main():
    from questionforge.learning.focus import FOCUS_SYSTEM, focused_payload
    from questionforge.learning.pipeline import validate_edits
    from questionforge.learning.quality import QUALITY_MODEL, QUALITY_PRECISION, quality_chat_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["edit", "draft-free"], required=True)
    args = parser.parse_args()
    original = json.loads((args.source / "config.json").read_text(encoding="utf-8"))
    completed = json.loads((args.source / "summary.json").read_text(encoding="utf-8"))
    if not completed["study_complete"] or completed["cases_with_execution_error"]:
        raise ValueError("The complete, error-free original comparison is required first.")
    input_paths = sorted(path for path in args.source.glob("*.json")
                         if "case_id" in json.loads(path.read_text(encoding="utf-8")))
    sources = [Path(__file__).relative_to(Path.cwd()), Path("scripts/run_learning_focus_study.py"),
               Path("docs/ANSWER_ROLE_PROTOCOL.md"), Path("results/learning_focus/refinement_selection.json"),
               *Path("questionforge/learning").glob("*.py"),
               *[Path("questionforge") / f"{name}.py" for name in
                 ["models", "registry", "text", "metrics", "verification", "experiment"]]]
    config = {"version": "answer-role-v1", "split": original["split"], "variant": args.variant,
              "planner_model": QUALITY_MODEL.model_id, "planner_revision": QUALITY_MODEL.revision,
              "cpu_precision": QUALITY_PRECISION, "greedy_edit": True,
              "source_study": args.source.as_posix(),
              "source_case_sha256": {p.as_posix(): source_hash(p) for p in input_paths},
              "source_sha256_normalized_utf8": {p.as_posix(): source_hash(p) for p in sources},
              "data_sha256": original["data_sha256"],
              "interpretation": "Development re-edit only. Facts, offsets, goals and T5 drafts are fixed; QA recovery is not educational usefulness."}
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Source/config changed. Preserve this run and choose a fresh output directory.")
    write_json(config_path, config)
    snapshot_sources(directory, sources)
    if not (directory / "environment.json").exists():
        write_json(directory / "environment.json", environment_info())
    cases = []
    methods = ["planned_t5", "baseline_hybrid", "role_hybrid"]
    for source_path in input_paths:
        target = directory / source_path.name
        if target.exists():
            cases.append(json.loads(target.read_text(encoding="utf-8")))
            continue
        source = json.loads(source_path.read_text(encoding="utf-8"))
        report = copy.deepcopy(source["planning_report"])
        case = {key: source[key] for key in ["case_id", "language", "group", "context"]}
        case.update(source_case=source_path.as_posix(), source_case_sha256=source_hash(source_path),
                    outputs={"planned_t5": source["outputs"]["planned_t5"],
                             "baseline_hybrid": source["outputs"]["learning_hybrid"]}, error=None)
        report.update(version="answer-role-v1", baseline_raw_edit=report["raw_edit"], raw_edit=None,
                      rejected_questions=[], refinement_variant=args.variant, edit_recipe="answer-role-v1")
        started = time.perf_counter()
        print("Re-editing", case["case_id"], args.variant, flush=True)
        try:
            payload = focused_payload(case["context"], case["language"], report["facts"], report["drafts"],
                                      report["learning_objective"], include_drafts=args.variant == "edit")
            case["edit_payload"] = payload
            edited, report["raw_edit"] = quality_chat_json(FOCUS_SYSTEM, payload)
            rows = validate_edits(case["context"], case["language"], edited, report, QUALITY_MODEL)
            case["outputs"]["role_hybrid"] = [diagnostics(row, case["context"], case["language"]) for row in rows]
        except Exception as exc:
            case["error"] = f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "raw_text"):
                report["raw_edit"] = exc.raw_text
            print(case["error"], flush=True)
        report["returned"] = len(case["outputs"].get("role_hybrid", []))
        case.update(planning_report=report, seconds=time.perf_counter() - started)
        write_json(target, case)
        cases.append(case)
        render_report(directory, cases, methods=methods)
    summary = {"scope": "Paired synthetic development re-edit. No fresh-test or human-quality claim.",
               "planned_passage_language_pairs": len(input_paths), "passage_language_pairs": len(cases),
               "study_complete": len(cases) == len(input_paths),
               "cases_with_execution_error": sum(bool(c["error"]) for c in cases),
               "fixed_planned_facts": sum(len(c["planning_report"]["facts"]) for c in cases),
               "human_learning_value_ratings": "pending", "methods": {}}
    for method in methods:
        rows = [r for c in cases for r in c["outputs"].get(method, [])]
        summary["methods"][method] = {
            "returned_questions": len(rows),
            "mean_qa_answer_f1_diagnostic": sum(r["answer_f1"] for r in rows) / len(rows) if rows else None,
            "interpretation": "Coverage and conditional QA recovery do not establish question quality."}
    write_json(directory / "summary.json", summary)
    render_report(directory, cases, write_template=True, methods=methods)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
