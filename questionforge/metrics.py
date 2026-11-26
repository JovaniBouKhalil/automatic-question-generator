"""Generation metrics with Unicode tokenization and explicit interpretation."""

from __future__ import annotations

from collections import Counter
import math

from .text import STOPWORDS, normalize_answer, tokens


def rouge_l(prediction: str, reference: str) -> float:
    """Token LCS F1 in [0,1]; no English-only stemming or ASCII filtering."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for word in pred:
        current = [0]
        for j, other in enumerate(ref, 1):
            current.append(previous[j - 1] + 1 if word == other else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(pred) + len(ref))


def lexical_relevance(question: str, evidence: str, language: str) -> float:
    """Content-token cosine; a lexical signal, not semantic entailment or factuality."""
    q = Counter(w for w in tokens(question) if w not in STOPWORDS[language])
    e = Counter(w for w in tokens(evidence) if w not in STOPWORDS[language])
    denominator = math.sqrt(sum(v*v for v in q.values()) * sum(v*v for v in e.values()))
    return sum(v * e[k] for k, v in q.items()) / denominator if denominator else 0.0


def leaks_answer(question: str, answer: str, language: str) -> bool:
    q, a = normalize_answer(question, language), normalize_answer(answer, language)
    return bool(a) and f" {a} " in f" {q} "


def quality_decision(row: dict, min_f1: float = 0.5, min_score: float = 0.1) -> tuple[bool, list[str]]:
    """Fixed review rules; QA scores are not calibrated probabilities of correctness."""
    reasons = []
    if not row["question"].strip():
        reasons.append("empty question")
    if row.get("answer_leak", False):
        reasons.append("answer appears in question")
    if row.get("answer_f1", 0) < min_f1:
        reasons.append("verifier answer differs from target")
    if row.get("qa_score", 0) < min_score:
        reasons.append("weak verifier span score")
    return not reasons, reasons


def aggregate(rows: list[dict]) -> dict:
    from sacrebleu.metrics import BLEU, CHRF

    if not rows:
        return {"n": 0}
    predictions, references = [r["question"] for r in rows], [r["reference"] for r in rows]
    bleu = BLEU(tokenize="intl", effective_order=True)
    chrf = CHRF(word_order=2)
    result = {"n": len(rows), "bleu": bleu.corpus_score(predictions, [references]).score,
              "chrf_pp": chrf.corpus_score(predictions, [references]).score,
              "rouge_l": 100 * sum(rouge_l(p, r) for p, r in zip(predictions, references)) / len(rows),
              "bleu_signature": str(bleu.get_signature()), "chrf_signature": str(chrf.get_signature())}
    for key in ["answer_f1", "answer_em", "lexical_relevance", "answer_leak", "accepted", "cropped"]:
        if all(key in r for r in rows):
            result[key] = 100 * sum(float(r[key]) for r in rows) / len(rows)
    result["generation_seconds_mean"] = sum(r.get("generation_seconds", 0) for r in rows) / len(rows)
    result["unique_question_pct"] = 100 * len(set(predictions)) / len(predictions)
    return result
