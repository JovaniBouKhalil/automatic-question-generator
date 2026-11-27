"""Paired FP32/int8 development controls before trusting the optional local editor."""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from questionforge.learning.pipeline import PLANNER
from questionforge.models import configure_torch, model_source
from scripts.run_learning_focus_study import source_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/learning_runtime_replay.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the earlier diagnostic; choose another --output.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch, device = configure_torch()
    source, kwargs = model_source(PLANNER)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
    print("Loading unquantized FP32 editor for diagnosis", flush=True)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32,
                                              trust_remote_code=False, **kwargs).to(device).eval()
    original_file = Path("results/quality/learning_editor_runtime_diagnostic.json")
    original = json.loads(original_file.read_text(encoding="utf-8"))
    assert original["model"] == PLANNER.model_id and original["revision"] == PLANNER.revision
    probes = [(p["probe"], p["system"], p["user"], 40 if p["probe"] == "json_control" else 180)
              for p in original["probes"] if p["precision"] == "float32"]
    results = {"model": PLANNER.model_id, "revision": PLANNER.revision,
               "source_sha256": source_hash(Path(__file__)), "original_control_sha256": source_hash(original_file), "probes": []}
    target = args.output
    for precision in ["float32", "dynamic_int8_fbgemm"]:
        if precision != "float32":
            print("Quantizing the same model", flush=True)
            torch.backends.quantized.engine = "fbgemm"
            torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        for name, system, user, budget in probes:
            encoded = tokenizer.apply_chat_template([{"role": "system", "content": system},
                        {"role": "user", "content": user}], tokenize=True, add_generation_prompt=True,
                        return_tensors="pt", return_dict=True).to(device)
            print(precision, name, "starting", flush=True)
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**encoded, max_new_tokens=budget, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            row = {"precision": precision, "probe": name, "system": system, "user": user,
                   "raw_response": text, "seconds": time.perf_counter() - started,
                   "new_tokens": int(output.shape[1] - encoded["input_ids"].shape[1])}
            results["probes"].append(row)
            write_json(target, results)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    print("Paired runtime diagnostic saved:", target, flush=True)


if __name__ == "__main__":
    main()
