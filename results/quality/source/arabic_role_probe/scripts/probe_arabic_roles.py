"""Conditional development probe for Arabic conversion syntax, with an English control."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from questionforge.learning.focus import FOCUS_SYSTEM, focused_payload
from questionforge.learning.pipeline import question_issue
from questionforge.learning.quality import QUALITY_MODEL, quality_chat_json
from scripts.run_learning_focus_study import source_hash, write_json

ARABIC_ROLE_INSTRUCTION = """
For Arabic conversion OUTPUT answers, use the active-voice question frame
'إلى ماذا يحوّل ... ...؟': after يحوّل put the source converting subject, then its input.
The question asks for the resulting output. Do not switch to يتحول or ask which device
performs the conversion. Do not state the target output in the question.
For other Arabic purpose answers, ask 'ما الغرض من استخدام ...؟' or 'فيم تستخدم ...؟'.
These are grammatical frames, not text to copy with placeholders. For English/French,
continue writing naturally in the requested language; the Arabic frames do not apply.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/quality/learning_4b_arabic_role_probe.json"))
    args = parser.parse_args()
    original = Path("results/quality/learning_4b_role_focus_pilot.json")
    while True:
        try:
            saved = json.loads(original.read_text(encoding="utf-8"))
            if sum(len(c["variants"]) for c in saved["cases"]) == 12:
                break
        except (OSError, ValueError):
            pass
        time.sleep(10)
    if args.output.exists():
        raise ValueError("Preserve the existing probe; use a fresh --output.")
    system = FOCUS_SYSTEM + ARABIC_ROLE_INSTRUCTION
    report = {"scope": "Purposive development probe after observed pilot syntax errors. Not human ratings or a fresh test.",
              "model": QUALITY_MODEL.model_id, "revision": QUALITY_MODEL.revision,
              "precision": "none-float32", "system": system,
              "source_sha256": {p: source_hash(p) for p in
                                ["scripts/probe_arabic_roles.py", "questionforge/learning/focus.py",
                                 "docs/ANSWER_ROLE_PROTOCOL.md"]}, "cases": []}
    for name in ["learning_light_and_voltage_ar", "learning_training_and_evaluation_ar", "learning_text_representation_en"]:
        source = Path("results/learning_focus_educational_4b_v1") / (name + ".json")
        case = json.loads(source.read_text(encoding="utf-8"))
        plan = case["planning_report"]
        payload = focused_payload(case["context"], case["language"], plan["facts"], plan["drafts"], plan["learning_objective"])
        row = {"source_case": source.as_posix(), "source_sha256": source_hash(source), "payload": payload}
        start = time.monotonic()
        print("Arabic-role probe", name, flush=True)
        try:
            parsed, raw = quality_chat_json(system, payload)
            row.update(parsed=parsed, raw=raw)
            row["format_issues"] = [question_issue(item.get("question"), plan["facts"][item["index"]]["answer"],
                                                   case["language"]) for item in parsed]
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", raw=getattr(exc, "raw_text", None))
        row["seconds"] = time.monotonic() - start
        report["cases"].append(row)
        write_json(args.output, report)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
