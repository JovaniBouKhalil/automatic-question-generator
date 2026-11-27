"""Inspect a CPU-only relation-targeting T5 alternative on synthetic development passages."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from questionforge.candidates import extract_candidates
from questionforge.learning.rules import create_learning_t5_quiz
from questionforge.models import generate_question
from scripts.run_learning_focus_study import diagnostics, render_report, source_hash, write_json


def main():
    folder = Path("results/learning_t5_educational_v1")
    folder.mkdir(parents=True, exist_ok=True)
    data = Path("data/authoring/learning_passages.json")
    paths = [*Path("questionforge/learning").glob("*.py"), Path(__file__).relative_to(Path.cwd()),
             Path("scripts/run_learning_focus_study.py"), Path("questionforge/candidates.py"),
             Path("questionforge/models.py"), Path("docs/LEARNING_FOCUS_PROTOCOL.md")]
    config = {"data_sha256": source_hash(data), "count": 3, "beams": 4, "device": "cpu", "native_adapters": False,
              "sources": {p.as_posix(): source_hash(p) for p in paths}, "split": "synthetic_educational_development"}
    target = folder / "config.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != config:
        raise ValueError("Preserve this study before changing its source or configuration.")
    write_json(target, config)
    cases = []
    for scenario in json.loads(data.read_text(encoding="utf-8")):
        for lang in ["en", "fr", "ar"]:
            case_id = f"learning:{scenario['topic']}:{lang}"
            file = folder / (case_id.replace(":", "_") + ".json")
            if file.exists():
                cases.append(json.loads(file.read_text(encoding="utf-8")))
                continue
            context = scenario[lang]["context"]
            case = {"case_id": case_id, "language": lang, "context": context, "outputs": {}, "error": None}
            started = time.perf_counter()
            print(case_id, flush=True)
            try:
                case["outputs"]["recall_t5"] = [diagnostics(asdict(generate_question(context, c.answer, lang,
                                 answer_start=c.start)), context, lang) for c in extract_candidates(context, lang, 3)]
                rows, report = create_learning_t5_quiz(context, lang, 3, objective=scenario[lang]["objective"])
                case["planning_report"] = report
                case["outputs"]["planned_t5"] = [diagnostics(row, context, lang) for row in rows]
            except Exception as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
            case["seconds"] = time.perf_counter() - started
            write_json(file, case)
            cases.append(case)
    summary = {"scope": "Synthetic development/demo passages; target selection changes. No measured human usefulness improvement.",
               "passage_language_pairs": len(cases), "requested_per_method": len(cases) * 3,
               "execution_errors": sum(bool(c["error"]) for c in cases),
               "human_learning_value_ratings": "pending",
               "returned": {method: sum(len(c["outputs"].get(method, [])) for c in cases)
                            for method in ["recall_t5", "planned_t5"]}}
    write_json(folder / "summary.json", summary)
    render_report(folder, cases, write_template=True)
    print(json.dumps(summary, indent=2), flush=True)
    # Fingerprint the inspected record set, not an invented educational-quality score.
    fingerprint = hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    print("Development comparison records SHA256:", fingerprint)


if __name__ == "__main__":
    main()
