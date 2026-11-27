"""Conservative fact selection from explicit relation markers; native T5 writes questions.

These transparent heuristics identify possible learning targets, not educational merit.
They may abstain when a passage lacks a supported pattern.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict

from questionforge.metrics import leaks_answer, lexical_relevance
from questionforge.models import generate_question
from questionforge.text import evidence_for, normalize_answer, sentences, validate_language

from .pipeline import question_issue

PATTERNS = {
    "en": [
        ("cause_effect", r"\b(?:because(?: of)?|so that|in order to|to avoid)\s+(?P<answer>[^.!?;]+)", 5),
        ("function", r"\b(?:is used to|are used to|used to|designed to|purpose .{0,30}? was to|helps)\s+(?P<answer>[^.!?;]+)", 4),
        ("concept", r"\bdoes not preserve\s+(?P<answer>[^.!?;]+)", 4),
        ("function", r"\brecords\s+(?P<answer>[^.!?;]+)", 3),
        ("function", r"\bconverts .{1,60}? into\s+(?P<answer>[^.!?;]+)", 4),
    ],
    "fr": [
        ("cause_effect", r"\b(?:parce que|afin que|afin de|pour éviter)\s+(?P<answer>[^.!?;]+)", 5),
        ("function", r"\b(?:servent à|sert à|servait à|permet de|permettent de)\s+(?P<answer>[^.!?;]+)", 4),
        ("concept", r"\bne conserve pas\s+(?P<answer>[^.!?;]+)", 4),
        ("function", r"\benregistre\s+(?P<answer>[^.!?;]+)", 3),
        ("function", r"\bconvertit .{1,60}? en\s+(?P<answer>[^.!?;]+)", 4),
    ],
    "ar": [
        ("cause_effect", r"(?:بسبب|لأن)\s+(?P<answer>[^.؟!؛]+)", 5),
        ("cause_effect", r"حتى\s+(?P<answer>لا\s+[^.؟!؛]+)", 5),
        ("cause_effect", r"ل(?P<answer>تجنب\s+[^.؟!؛]+)", 5),
        ("function", r"(?:تستخدم|يستخدم|تستعمل|يستعمل)\s+.+?\s+ل(?P<answer>[^.؟!؛]+)", 4),
        ("function", r"(?:الهدف من .{0,40}?|يساعد ذلك)\s+(?:على|ل|هو)?\s*(?P<answer>[^.؟!؛]+)", 3),
        ("concept", r"لا يحتفظ ب(?P<answer>[^.؟!؛]+)", 4),
        ("function", r"بحسب\s+(?P<answer>[^.؟!؛]+)", 3),
        ("function", r"(?:يحول|تحول)\s+.{1,70}?\s+إلى\s+(?P<answer>[^.؟!؛]+)", 4),
    ],
}
GOALS = {
    "en": {"cause_effect": "Understand an explicitly stated reason or purpose.",
           "function": "Understand the function described in the passage.",
           "concept": "Understand an important property or limitation."},
    "fr": {"cause_effect": "Comprendre une raison ou un objectif explicitement indiqué.",
           "function": "Comprendre la fonction décrite dans le passage.",
           "concept": "Comprendre une propriété ou une limite importante."},
    "ar": {"cause_effect": "فهم سبب أو هدف مذكور صراحة في النص.",
           "function": "فهم الوظيفة الموصوفة في النص.",
           "concept": "فهم خاصية أو قيد مهم."},
}


def plan_learning_facts(context: str, language: str, count: int = 3) -> list[dict]:
    validate_language(language)
    if not 1 <= count <= 8:
        raise ValueError("Choose between one and eight questions.")
    proposed = []
    for sentence in sentences(context):
        for kind, pattern, priority in PATTERNS[language]:
            for match in re.finditer(pattern, sentence.text, re.IGNORECASE):
                raw = match.group("answer")
                answer = raw.strip(" ,،:")
                if not 4 <= len(answer) <= 240:
                    continue
                start = sentence.start + match.start("answer") + raw.index(answer)
                proposed.append({"answer": answer, "answer_start": start, "evidence": sentence.text,
                                 "question_type": kind, "learning_goal": GOALS[language][kind], "priority": priority})
    proposed.sort(key=lambda row: (-row["priority"], row["answer_start"]))
    chosen, seen, used_sentences = [], set(), set()
    for row in proposed:
        key = normalize_answer(row["answer"], language)
        if key in seen or row["evidence"] in used_sentences:
            continue
        seen.add(key)
        used_sentences.add(row["evidence"])
        chosen.append(row)
        if len(chosen) == count:
            break
    return chosen


def create_learning_t5_quiz(context: str, language: str, count: int = 3, *, objective: str = "",
                             manual_answers: list[str] | None = None, beams: int = 4,
                             checkpoint: str | None = None, device: str = "cpu", progress=None):
    validate_language(language)
    if not context.strip() or not 1 <= count <= 8:
        raise ValueError("Add a passage and request between one and eight questions.")
    if len(objective) > 500:
        raise ValueError("Keep the learning objective within 500 characters.")
    started = time.perf_counter()
    notice = progress or (lambda message: None)
    facts = plan_learning_facts(context, language, count)
    if manual_answers:
        facts = [{"answer": answer, "answer_start": context.find(answer),
                  "evidence": evidence_for(context, answer).text, "question_type": "concept",
                  "learning_goal": GOALS[language]["concept"]} for answer in dict.fromkeys(manual_answers[:count])]
    report = {"version": "learning-t5-v1", "planner": "explicit_relation_rules", "requested": count,
              "planned": len(facts), "facts": facts, "drafts": [], "rejected_questions": [],
              "learning_objective": objective, "educational_quality_status": "not_human_rated"}
    rows = []
    for index, fact in enumerate(facts):
        notice(f"Generating T5 question {index + 1}/{len(facts)}…")
        generated = asdict(generate_question(context, fact["answer"], language, answer_start=fact["answer_start"],
                                             beams=beams, checkpoint=checkpoint, device=device))
        report["drafts"].append(generated)
        issue = question_issue(generated["question"], fact["answer"], language)
        if issue:
            report["rejected_questions"].append({"index": index, "reason": issue})
            continue
        rows.append({**generated, "method": "learning_focused_t5", "learning_goal": fact["learning_goal"],
                     "question_type": fact["question_type"], "planner": "explicit_relation_rules",
                     "answer_leak": leaks_answer(generated["question"], fact["answer"], language),
                     "lexical_relevance": lexical_relevance(generated["question"], fact["evidence"], language),
                     "educational_quality_status": "not_human_rated"})
    report.update(returned=len(rows), seconds=time.perf_counter() - started)
    return rows, report
