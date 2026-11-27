"""Replay saved controls with FP16-packed linear weights and FP32 activations."""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from questionforge.learning.pipeline import PLANNER, parse_array, validate_plan
from questionforge.models import configure_torch, model_source
from scripts.run_learning_focus_study import source_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/learning_fp16_replay.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the earlier diagnostic; choose another --output.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch, device = configure_torch()
    original_file = Path("results/quality/learning_editor_runtime_diagnostic.json")
    original = json.loads(original_file.read_text(encoding="utf-8"))
    assert original["model"] == PLANNER.model_id and original["revision"] == PLANNER.revision
    source, kwargs = model_source(PLANNER)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32,
                                              trust_remote_code=False, **kwargs).to(device).eval()
    torch.backends.quantized.engine = "fbgemm"
    torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16, inplace=True)
    output = {"model": PLANNER.model_id, "revision": PLANNER.revision, "precision": "FP16 packed linear weights, FP32 activations",
              "original_control_sha256": source_hash(original_file), "script_sha256": source_hash(Path(__file__)), "probes": []}
    for probe in [p for p in original["probes"] if p["precision"] == "float32"]:
        messages = [{"role": "system", "content": probe["system"]}, {"role": "user", "content": probe["user"]}]
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors="pt").to(device)
        print("FP16 packing control:", probe["probe"], flush=True)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=40 if probe["probe"] == "json_control" else 180,
                                       do_sample=False, pad_token_id=tokenizer.pad_token_id)
        raw = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        row = {"probe": probe["probe"], "raw_response": raw, "seconds": time.perf_counter() - started,
               "new_tokens": int(generated.shape[1] - encoded["input_ids"].shape[1])}
        try:
            parsed = parse_array(raw)
            if probe["probe"] == "json_control":
                row["format_grounding_control_pass"] = parsed == ["light"]
            else:
                payload = json.loads(probe["user"])
                context = " ".join(s["text"] for s in payload["passage_sentences"])
                accepted, rejected = validate_plan(context, parsed, "ar", 1)
                row.update(format_grounding_control_pass=bool(accepted) and not rejected,
                           accepted_facts=accepted, rejected_facts=rejected)
        except ValueError:
            row["format_grounding_control_pass"] = False
        output["probes"].append(row)
        write_json(args.output, output)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
