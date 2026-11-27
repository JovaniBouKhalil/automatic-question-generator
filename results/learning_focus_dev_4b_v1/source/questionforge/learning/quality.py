"""Pinned stronger model for the learning-focused development comparison.

The original T5 and smaller editor loaders remain available for reproduction.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from questionforge.registry import ModelSpec

QUALITY_MODEL = ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554",
                          "chat", "Apache-2.0")
QUALITY_PRECISION = "none-float32"


def local_quality_source() -> Path:
    root = Path(__file__).resolve().parents[2] / "models" / QUALITY_MODEL.model_id.replace("/", "--")
    try:
        receipt = json.loads((root / "download_receipt.json").read_text(encoding="utf-8"))
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        valid = (receipt["revision"] == QUALITY_MODEL.revision and receipt["all_weights_verified"]
                 and shards == set(receipt["weight_sha256"]) and all((root / p).is_file() for p in shards))
    except (OSError, KeyError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError("The stronger local model is not ready. Run: python -m scripts.download_quality_model")
    return root


@lru_cache(maxsize=1)
def load_quality_model(device: str = "cpu", precision: str = QUALITY_PRECISION):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from questionforge.models import configure_torch

    source = local_quality_source()
    torch, device = configure_torch(device=device)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                              torch_dtype=torch.float32).to(device).eval()
    if precision == "dynamic-fp16-linear" and device == "cpu":
        torch.backends.quantized.engine = "fbgemm"
        torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.float16, inplace=True)
    elif precision != "none-float32":
        raise ValueError("Unsupported quality model precision/device combination.")
    return tokenizer, model, torch, device


def quality_chat_json(system: str, payload: dict, device: str = "cpu", max_new_tokens: int | None = None,
                      *, precision: str = QUALITY_PRECISION) -> tuple[list, str]:
    from questionforge.learning.pipeline import StructuredOutputError, parse_array

    tokenizer, model, torch, actual_device = load_quality_model(device, precision)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                            return_dict=True, return_tensors="pt").to(actual_device)
    if encoded["input_ids"].shape[1] > 6000:
        raise ValueError("Select a shorter passage for learning-focused generation (up to 6,000 input tokens).")
    requested = payload.get("count", len(payload.get("facts", [])))
    budget = max_new_tokens or min(1500, max(256, requested * 180))
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=budget,
                                pad_token_id=tokenizer.pad_token_id, use_cache=True)
    raw = tokenizer.decode(output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        return parse_array(raw), raw
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(raw) from exc


def create_quality_quiz(*args, **kwargs):
    from questionforge.learning.pipeline import create_learning_quiz

    return create_learning_quiz(*args, **kwargs, complete=quality_chat_json, planner_spec=QUALITY_MODEL,
                                cpu_precision=QUALITY_PRECISION)
