"""Development-only prompt diagnostic on a saved failed Arabic question-editing case."""

import json
import time
from pathlib import Path

from questionforge.learning.pipeline import chat_json, question_issue
from scripts.run_learning_focus_study import source_hash, write_json

SYSTEM = """اكتب جميع الأسئلة باللغة العربية فقط. اكتب سؤالا واضحا ومباشرا لكل حقيقة.
الجواب المطلوب محدد؛ يجب أن يجيب هذا الجواب عن السؤال. لا تضع الجواب في السؤال.
استعمل النص فقط. لا تسأل لماذا إلا إذا ذكر النص سببا أو هدفا.
احتفظ بالسؤال المقترح إذا كان صحيحا؛ أصلح فقط الأخطاء أو الغموض.

Examples of correct question roles:
Passage: يحول المحرك الكهرباء إلى حركة. Answer: حركة
Question: إلى ماذا يحول المحرك الكهرباء؟
Passage: لا يحتفظ التنسيق بالألوان. Answer: الألوان
Question: ما المعلومات التي لا يحتفظ بها التنسيق؟
Passage: يستخدم الميزان لقياس الكتلة. Answer: قياس الكتلة
Question: لأي غرض يستخدم الميزان؟

Return ONLY a JSON array of objects with integer "index" and Arabic "question".
Use the same index as each supplied fact. No explanations."""


def main():
    case_file = Path("results/learning_focus_educational_v3/learning_text_representation_ar.json")
    case = json.loads(case_file.read_text(encoding="utf-8"))
    report = case["planning_report"]
    payload = {"language": "Arabic (العربية)", "passage": case["context"],
               "facts": [{"index": i, "answer": fact["answer"], "evidence": fact["evidence"],
                          "draft_question": report["drafts"][i]["question"]}
                         for i, fact in enumerate(report["facts"])]}
    output = {"scope": "Development prompt diagnostic, not a human rating or fresh test.",
              "case_path": str(case_file), "system": SYSTEM, "payload": payload,
              "script_sha256": source_hash(Path(__file__)), "case_sha256": source_hash(case_file)}
    started = time.perf_counter()
    print("Running a localized Arabic question-editing diagnostic", flush=True)
    try:
        edited, raw = chat_json(SYSTEM, payload)
        output.update(raw_response=raw, questions=edited)
        output["format_issues"] = [{"index": item.get("index"),
             "issue": question_issue(item.get("question"), report["facts"][item["index"]]["answer"], "ar")}
             for item in edited if isinstance(item, dict) and type(item.get("index")) is int
             and 0 <= item["index"] < len(report["facts"])]
    except Exception as exc:
        output.update(error=f"{type(exc).__name__}: {exc}", raw_response=getattr(exc, "raw_text", None))
    output["seconds"] = time.perf_counter() - started
    write_json(Path("results/quality/learning_localized_editor_pilot.json"), output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
