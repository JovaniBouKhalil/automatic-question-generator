"""Optional multilingual-QA ablation; the original verifier remains unchanged."""

from __future__ import annotations

from functools import lru_cache

from .models import configure_torch, model_source, verify_question
from .registry import ModelSpec
from .text import answer_f1, normalize_answer, validate_language

MULTILINGUAL_QA = ModelSpec(
    "mrm8488/bert-multi-cased-finetuned-xquadv1",
    "610161fac355eac3e06ac622d376e178a948a5d0",
    "extractive_qa_answerable_only",
    "Unspecified; do not redistribute weights",
)


@lru_cache(maxsize=1)
def load_multilingual_verifier(device="cpu"):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

    _, actual_device = configure_torch(device=device)
    source, options = model_source(MULTILINGUAL_QA)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **options)
    model = AutoModelForQuestionAnswering.from_pretrained(source, trust_remote_code=False, **options)
    return pipeline("question-answering", tokenizer=tokenizer, model=model, device=actual_device)


def check_answer(question, context, answer, language, verifier="original", device="cpu"):
    """Keep model choice explicit and use the same answer normalization for both judges.

    The alternative was not trained on impossible questions. Its span score is not an
    answerability probability, so deliberately do not enable a fake no-answer decision.
    """
    validate_language(language)
    if verifier == "original":
        return verify_question(question, context, answer, language, device)
    if verifier != "multilingual":
        raise ValueError("Choose original or multilingual verifier.")
    if not question.strip():
        return {"qa_answer": "", "qa_score": 0.0, "answer_f1": 0.0, "answer_em": 0.0}
    result = load_multilingual_verifier(device)(
        question=question, context=context, handle_impossible_answer=False,
        max_seq_len=384, doc_stride=96, max_question_len=64, max_answer_len=64,
    )
    extracted = result["answer"]
    return {"qa_answer": extracted, "qa_score": float(result["score"]) if extracted else 0.0,
            "answer_f1": answer_f1(extracted, answer, language),
            "answer_em": float(normalize_answer(extracted, language) == normalize_answer(answer, language))}
