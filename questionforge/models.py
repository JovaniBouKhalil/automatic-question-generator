"""Lazy model loading and bounded, answer-aware sequence-to-sequence inference."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .registry import GENERATORS, VERIFIER, ModelSpec
from .text import answer_f1, evidence_for, normalize_answer, validate_language

ROOT = Path(__file__).resolve().parents[1]


def model_source(spec: ModelSpec) -> tuple[str, dict]:
    local = ROOT / "models" / spec.model_id.replace("/", "--")
    weights = (local / "model.safetensors").is_file() or (local / "pytorch_model.bin").is_file()
    metadata_file = local / "hub_metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
    if weights and (local / "config.json").is_file() and metadata.get("sha") == spec.revision:
        return str(local), {"local_files_only": True}
    return spec.model_id, {"revision": spec.revision}


def configure_torch(seed: int = 42, device: str = "cpu"):
    import torch

    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("Device must be cpu, cuda or auto.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Choose CPU or install a compatible CUDA PyTorch build.")
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    return torch, device


@lru_cache(maxsize=3)
def load_generator(language: str, device: str = "cpu", checkpoint: str | None = None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    validate_language(language)
    torch, device = configure_torch(device=device)
    adapter = checkpoint is not None and (Path(checkpoint) / "adapter_config.json").is_file()
    source, kwargs = model_source(GENERATORS[language]) if checkpoint is None or adapter else (checkpoint, {})
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(source, trust_remote_code=False, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    model.to(device).eval()
    return tokenizer, model, torch, device


def format_prompt(context: str, answer: str, start: int, style: str) -> str:
    if start < 0 or context[start:start + len(answer)] != answer:
        raise ValueError("Invalid answer offset.")
    if style == "context_answer":
        return f"context: {context} answer: {answer} </s>"
    highlighted = context[:start] + "<hl> " + answer + " <hl>" + context[start + len(answer):]
    return ("generate question: " if style == "en_highlight" else "") + highlighted


def answer_window(context: str, answer: str, start: int, tokenizer, style: str, max_tokens: int):
    """Crop around the answer, never silently truncate away the fact being queried.

    Character-radius binary search measures the *actual prompted* token count, so
    Arabic's different subword fertility and model prefixes are included in the budget.
    """
    def make(radius):
        left = max(0, start - radius)
        right = min(len(context), start + len(answer) + radius)
        return context[left:right], start - left

    def size(text, offset):
        return len(tokenizer(format_prompt(text, answer, offset, style), verbose=False)["input_ids"])

    if size(context, start) <= max_tokens:
        return context, start, False
    minimal, offset = make(0)
    if size(minimal, offset) > max_tokens:
        raise ValueError("The target answer alone exceeds the model's input budget.")
    low, high = 0, len(context)
    while low < high:
        mid = (low + high + 1) // 2
        text, offset = make(mid)
        if size(text, offset) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    text, offset = make(low)
    return text, offset, True


@dataclass
class GeneratedQuestion:
    question: str
    answer: str
    answer_start: int
    evidence: str
    language: str
    method: str
    model_id: str
    input_tokens: int = 0
    cropped: bool = False


def generate_question(context: str, answer: str, language: str = "en", *,
                      answer_start: int | None = None, beams: int = 4,
                      max_input_tokens: int = 384, max_new_tokens: int = 64,
                      device: str = "cpu", checkpoint: str | None = None) -> GeneratedQuestion:
    validate_language(language)
    if not 1 <= beams <= 8:
        raise ValueError("Beam count must be between 1 and 8.")
    if not 32 <= max_input_tokens <= 512 or not 8 <= max_new_tokens <= 128:
        raise ValueError("Input budget must be 32–512 tokens and output budget 8–128 tokens.")
    start = context.find(answer) if answer_start is None else answer_start
    evidence = evidence_for(context, answer, start)
    tokenizer, model, torch, actual_device = load_generator(language, device, checkpoint)
    spec = GENERATORS[language]
    window, offset, cropped = answer_window(context, answer, start, tokenizer, spec.prompt_style, max_input_tokens)
    prompt = format_prompt(window, answer, offset, spec.prompt_style)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False).to(actual_device)
    generation = {"num_beams": beams, "do_sample": False, "max_new_tokens": max_new_tokens,
                  "no_repeat_ngram_size": 3}
    if beams > 1:
        generation["early_stopping"] = True
    with torch.inference_mode():
        output = model.generate(**encoded, **generation)
    question = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if question.casefold().startswith("question:"):
        question = question[len("question:"):].strip()
    return GeneratedQuestion(question, answer, start, evidence.text, language,
                             f"transformer_beam{beams}", checkpoint or spec.model_id,
                             int(encoded["input_ids"].shape[1]), cropped)


@lru_cache(maxsize=1)
def load_verifier(device: str = "cpu"):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    _, device = configure_torch(device=device)
    source, kwargs = model_source(VERIFIER)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
    model = AutoModelForQuestionAnswering.from_pretrained(source, trust_remote_code=False, **kwargs)
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)


def verify_question(question: str, context: str, answer: str, language: str, device: str = "cpu") -> dict:
    """Independent extractive QA; answer agreement is a proxy, never a factuality proof.

    The verifier is multilingual-pretrained but fine-tuned on English SQuAD2. Report
    reference-question controls per language to expose cross-lingual judge weaknesses.
    """
    validate_language(language)
    if not question.strip():
        return {"qa_answer": "", "qa_score": 0.0, "answer_f1": 0.0, "answer_em": 0.0}
    result = load_verifier(device)(question=question, context=context, handle_impossible_answer=True,
                                   max_seq_len=384, doc_stride=96, max_question_len=64,
                                   max_answer_len=64)
    extracted = result["answer"]
    return {"qa_answer": extracted, "qa_score": float(result["score"]) if extracted else 0.0,
            "answer_f1": answer_f1(extracted, answer, language),
            "answer_em": float(normalize_answer(extracted, language) == normalize_answer(answer, language))}
