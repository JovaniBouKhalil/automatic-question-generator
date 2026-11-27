"""Paired FP32/FP16-packed controls for the stronger multilingual editor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from questionforge.learning.pipeline import EDIT_SYSTEM, parse_array, question_issue
from questionforge.learning.quality import QUALITY_MODEL, load_quality_model
from scripts.run_learning_focus_study import source_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/quality_model_control.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the prior control; choose a new --output.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    case_path = Path("results/learning_focus_educational_v3/learning_text_representation_ar.json")
    case = json.loads(case_path.read_text(encoding="utf-8"))
    facts = [{"index": i, **fact, "draft_question": case["planning_report"]["drafts"][i]["question"]}
             for i, fact in enumerate(case["planning_report"]["facts"])]
    english = {"language": "English", "passage": "A mesh catches leaves before they reach the pump. It protects the pump from blockage.",
               "facts": [{"index": 0, "answer": "protects the pump from blockage", "evidence": "It protects the pump from blockage.",
                          "learning_goal": "Understand the function of the mesh.", "draft_question": "Who invented the pump?"}]}
    probes = [("json_control", 'Return only a JSON array containing the string "light".', {}, 40),
              ("english_function", EDIT_SYSTEM, english, 150),
              ("arabic_relations", EDIT_SYSTEM, {"language": "Arabic (العربية)", "passage": case["context"],
                                               "learning_objective": case["planning_report"]["learning_objective"], "facts": facts}, 540)]
    tokenizer, model, torch, device = load_quality_model("cpu", "none-float32")
    results = {"model": QUALITY_MODEL.model_id, "revision": QUALITY_MODEL.revision,
               "script_sha256": source_hash(Path(__file__)), "source_case_sha256": source_hash(case_path),
               "interpretation": "Paired runtime controls, not human educational-quality ratings.", "probes": []}
    for precision in ["none-float32", "dynamic-fp16-linear"]:
        if precision == "dynamic-fp16-linear":
            torch.backends.quantized.engine = "fbgemm"
            torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16, inplace=True)
        for name, system, payload, budget in probes:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(device)
            print(precision, name, "starting", flush=True)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, max_new_tokens=budget,
                                           pad_token_id=tokenizer.pad_token_id, use_cache=True)
            raw = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            row = {"probe": name, "precision": precision, "system": system, "payload": payload,
                   "raw_response": raw, "seconds": time.perf_counter() - started,
                   "new_tokens": int(generated.shape[1] - encoded["input_ids"].shape[1])}
            try:
                parsed = parse_array(raw)
                row["parsed"] = parsed
                if name == "json_control":
                    row["json_control_pass"] = parsed == ["light"]
                else:
                    row["format_issues"] = [question_issue(item.get("question"), payload["facts"][item["index"]]["answer"],
                                                          "ar" if name == "arabic_relations" else "en") for item in parsed]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                row["error"] = str(exc)
            results["probes"].append(row)
            write_json(args.output, results)
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
