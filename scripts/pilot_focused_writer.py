"""Check independent, evidence-focused questions without conditioning on faulty T5 drafts."""

import json
import time
from pathlib import Path

from questionforge.learning.pipeline import chat_json, question_issue
from questionforge.learning.rules import focused_evidence
from scripts.pilot_learning_editor import SYSTEM
from scripts.run_learning_focus_study import source_hash, write_json


def main():
    system = SYSTEM.replace("احتفظ بالسؤال المقترح إذا كان صحيحا؛ أصلح فقط الأخطاء أو الغموض.\n", "")
    probes = [("learning_text_representation_ar", i) for i in range(3)] + [("learning_light_and_voltage_ar", 0)]
    results = {"scope": "Development diagnostic; T5 drafts are excluded from writer input. No human ratings.",
               "system": system, "script_sha256": source_hash(Path(__file__)), "probes": []}
    target = Path("results/quality/learning_focused_writer_pilot.json")
    for name, index in probes:
        case_file = Path("results/learning_focus_educational_v3") / (name + ".json")
        case = json.loads(case_file.read_text(encoding="utf-8"))
        fact = case["planning_report"]["facts"][index]
        window, _ = focused_evidence(case["context"], fact["answer"], fact["answer_start"], "ar")
        payload = {"language": "Arabic (العربية)", "passage": window,
                   "facts": [{"index": 0, "answer": fact["answer"], "evidence": fact["evidence"]}]}
        row = {"source_case": str(case_file), "source_case_sha256": source_hash(case_file),
               "source_fact_index": index, "payload": payload}
        print(name, index, "starting", flush=True)
        started = time.perf_counter()
        try:
            parsed, raw = chat_json(system, payload)
            row.update(raw_response=raw, questions=parsed)
            row["format_issues"] = [question_issue(item.get("question"), fact["answer"], "ar")
                                    for item in parsed if isinstance(item, dict)]
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", raw_response=getattr(exc, "raw_text", None))
        row["seconds"] = time.perf_counter() - started
        results["probes"].append(row)
        write_json(target, results)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
