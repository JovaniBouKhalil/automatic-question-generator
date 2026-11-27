"""Development-only comparison of recall T5, planned-target T5 and edited questions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import time
from dataclasses import asdict
from pathlib import Path

from questionforge.candidates import extract_candidates
from questionforge.data import load_jsonl
from questionforge.learning.pipeline import CPU_QUANTIZATION, PLANNER, VERSION, create_learning_quiz
from questionforge.metrics import leaks_answer, lexical_relevance, quality_decision
from questionforge.models import generate_question
from questionforge.verification import check_answer


def source_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def snapshot_sources(directory, sources):
    for path in sources:
        target = directory / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and source_hash(target) != source_hash(path):
            raise ValueError(f"Saved source snapshot differs: {path}")
        if not target.exists():
            target.write_bytes(path.read_bytes())


def diagnostics(row, context, language):
    row = {**row, "lexical_relevance": lexical_relevance(row["question"], row["evidence"], language),
           "answer_leak": leaks_answer(row["question"], row["answer"], language)}
    row.update(check_answer(row["question"], context, row["answer"], language, "original"))
    row["accepted"], row["review_reasons"] = quality_decision(row)
    return row


def render_report(directory, cases, write_template=False):
    cards, rating_rows = [], []
    for case in cases:
        cells = []
        for method in ["recall_t5", "planned_t5", "learning_hybrid"]:
            questions = []
            for index, row in enumerate(case.get("outputs", {}).get(method, [])):
                e = html.escape
                questions.append(f"<li dir='auto'><b>{e(row['question'])}</b><p>Answer: {e(row['answer'])}</p>"
                                 f"<p>{e(row.get('learning_goal', 'Original heuristic target selection'))}</p>"
                                 f"<details><summary>Evidence and diagnostic</summary><p>{e(row['evidence'])}</p>"
                                 f"<p>QA answer-token F1: {row['answer_f1']:.3f}. This does not measure usefulness or factual support.</p>"
                                 "</details></li>")
                rating_rows.append({"item_id": f"{case['case_id']}:{method}:{index}", "language": case["language"],
                                    "method": method, "question": row["question"], "answer": row["answer"],
                                    "learning_goal": row.get("learning_goal", ""), "context": case["context"]})
            cells.append(f"<div><h3>{method.replace('_', ' ')}</h3><ol>{''.join(questions)}</ol></div>")
        error = f"<p>Execution error: {html.escape(case['error'])}</p>" if case.get("error") else ""
        cards.append(f"<section><h2>{html.escape(case['case_id'])}</h2><p dir='auto'>{html.escape(case['context'])}</p>"
                     f"{error}<div class='grid'>{''.join(cells)}</div></section>")
    page = """<!doctype html><html><meta charset='utf-8'><title>QuestionForge learning-focused development comparison</title>
<style>body{font:17px/1.6 Arial,sans-serif;background:#f6f7f3;color:#173c3b;max-width:1450px;margin:30px auto;padding:20px}
section{background:white;padding:24px;margin:24px 0;border:1px solid #cad8cf;border-radius:10px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:28px}
li{margin-bottom:22px}p{overflow-wrap:anywhere}details{font-size:14px;color:#536b60}@media(max-width:950px){.grid{grid-template-columns:1fr}}</style>
<h1>Does the question help a student understand?</h1><p>Development examples; model methods are visible. Original recall targets differ from planned targets.
Planned T5 and edited questions share targets. This is an inspectable comparison, not a completed human study.</p>
<p>Review four aspects: source support, clear wording, importance to the passage, and whether the question tests a concept/function/process/relationship.
A longer question or a higher QA score need not be more useful. No human ratings are claimed.</p>""" + "".join(cards) + "</html>"
    (directory / "comparison.html").write_text(page, encoding="utf-8")
    fields = ["reviewer_id", "item_id", "language", "method", "question", "answer", "learning_goal", "context",
              "factual_support_0_2", "clarity_0_2", "learning_value_0_2", "comment"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(rating_rows)
    target = directory / "learning_review_template.csv"
    if write_template and not target.exists():
        target.write_bytes(stream.getvalue().encode("utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/learning_focus_dev_v1"))
    parser.add_argument("--source", choices=["classroom", "educational"], default="classroom")
    parser.add_argument("--languages", nargs="+", choices=["en", "fr", "ar"], default=["en", "fr", "ar"])
    args = parser.parse_args()
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    sources = [Path("docs/LEARNING_FOCUS_PROTOCOL.md"), Path(__file__).relative_to(Path.cwd()),
               *Path("questionforge/learning").glob("*.py"),
               *[Path("questionforge") / f"{name}.py" for name in
                 ["candidates", "models", "registry", "text", "metrics", "verification"]]]
    data_path = "data/novel/dev.jsonl" if args.source == "classroom" else "data/authoring/learning_passages.json"
    config = {"version": VERSION, "split": f"{args.source}_development", "count": 3,
              "languages": args.languages,
              "seed": 42, "beams": 4, "native_adapters": False, "device": "cpu",
              "planner_model": PLANNER.model_id, "planner_revision": PLANNER.revision,
              "cpu_quantization": CPU_QUANTIZATION,
              "source_sha256_normalized_utf8": {p.as_posix(): source_hash(p) for p in sources},
              "data_sha256": source_hash(data_path),
              "interpretation": "Target selection differs across recall/planned methods. QA F1 is not usefulness or accuracy."}
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Study source/config differs. Preserve this run and choose a new --output directory.")
    write_json(config_path, config)
    snapshot_sources(directory, sources)
    if args.source == "classroom":
        unique = {(r["language"], r["group"]): r for r in load_jsonl(data_path) if r["language"] in args.languages}
    else:
        scenarios = json.loads(Path(data_path).read_text(encoding="utf-8"))
        unique = {(language, "learning:" + s["topic"]): s[language] for s in scenarios for language in args.languages}
    cases = []
    for (language, group), source in sorted(unique.items()):
        case_id = f"{group}:{language}"
        filename = directory / (case_id.replace(":", "_") + ".json")
        if filename.exists():
            cases.append(json.loads(filename.read_text(encoding="utf-8")))
            continue
        started = time.perf_counter()
        case = {"case_id": case_id, "language": language, "group": group, "context": source["context"],
                "outputs": {}, "error": None}
        print(f"Starting {case_id}", flush=True)
        try:
            original = []
            for candidate in extract_candidates(source["context"], language, 3):
                original.append(diagnostics(asdict(generate_question(source["context"], candidate.answer, language,
                                        answer_start=candidate.start, beams=4)), source["context"], language))
            case["outputs"]["recall_t5"] = original
            edited, report = create_learning_quiz(source["context"], language, count=3,
                                                   objective=source.get("objective", ""),
                                                   progress=lambda s: print(f"{case_id}: {s}", flush=True))
            case["planning_report"] = report
            case["outputs"]["planned_t5"] = [diagnostics(row, source["context"], language) for row in report["drafts"]]
            case["outputs"]["learning_hybrid"] = [diagnostics(row, source["context"], language) for row in edited]
        except Exception as exc:
            case["error"] = f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "raw_text"):
                case["failed_model_response"] = exc.raw_text
            print(case["error"], flush=True)
        case["seconds"] = time.perf_counter() - started
        write_json(filename, case)
        cases.append(case)
        render_report(directory, cases)
    summary = {"scope": f"Development-only synthetic {args.source} passages. No fresh-test or human-quality claim.",
               "passage_language_pairs": len(cases), "requested_questions_per_method": len(cases) * 3,
               "cases_with_execution_error": sum(bool(c.get("error")) for c in cases),
               "human_learning_value_ratings": "pending", "methods": {}}
    for method in ["recall_t5", "planned_t5", "learning_hybrid"]:
        rows = [r for c in cases for r in c["outputs"].get(method, [])]
        summary["methods"][method] = {"returned_questions": len(rows),
                       "mean_qa_answer_f1_diagnostic": sum(r["answer_f1"] for r in rows) / len(rows) if rows else None,
                       "interpretation": "QA recovery and output count do not establish educational usefulness."}
    write_json(directory / "summary.json", summary)
    render_report(directory, cases, write_template=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
