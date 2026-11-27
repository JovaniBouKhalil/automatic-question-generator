"""Paired FP32/int8 development controls before trusting the optional local editor."""

import json
import time
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from questionforge.learning.pipeline import PLAN_SYSTEM, PLANNER
from questionforge.models import configure_torch, model_source
from questionforge.text import sentences
from scripts.run_learning_focus_study import source_hash, write_json


def main():
    torch, device = configure_torch()
    source, kwargs = model_source(PLANNER)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
    print("Loading unquantized FP32 editor for diagnosis", flush=True)
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32,
                                              trust_remote_code=False, **kwargs).to(device).eval()
    scenarios = json.loads(Path("data/authoring/learning_passages.json").read_text(encoding="utf-8"))
    solar = next(s["ar"] for s in scenarios if s["topic"] == "light_and_voltage")
    probes = [("json_control", "Follow the user's instruction exactly.",
               'Return ONLY a JSON array containing the word light. No other text.', 40),
              ("arabic_single_fact", PLAN_SYSTEM, json.dumps({"language": "العربية", "count": 1,
                "learning_objective": solar["objective"], "passage_sentences": [
                    {"index": i, "text": p.text} for i, p in enumerate(sentences(solar["context"]))]}, ensure_ascii=False), 180)]
    results = {"model": PLANNER.model_id, "revision": PLANNER.revision,
               "source_sha256": source_hash(Path(__file__)), "probes": []}
    target = Path("results/quality/learning_editor_runtime_diagnostic.json")
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
