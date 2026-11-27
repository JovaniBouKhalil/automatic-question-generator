"""Source-derived answer-role cues for the controlled question-editing refinement."""

from __future__ import annotations

import re

from questionforge.text import LANGUAGES

from .pipeline import EDIT_SYSTEM

FOCUS_SYSTEM = EDIT_SYSTEM + """
Before writing, identify the role of the exact target answer in its evidence.
Follow answer_focus: the question must request that role, rather than the other endpoint
of the relation. Treat a faulty draft as untrusted; rewrite it completely when necessary.
For Arabic purpose answers, use a natural purpose question such as 'ما الغرض من استخدام ...؟'.
Keep Arabic active/passive roles and French agreement correct. The source determines the subject.
Do not copy an answer-focus hint or its placeholders into the question."""


def answer_focus(fact: dict) -> str:
    evidence, answer = fact["evidence"], fact["answer"]
    pos = evidence.find(answer)
    before = evidence[:pos].strip() if pos >= 0 else ""
    if re.search(r"\bwhile\b|\btandis que\b|بينما", evidence, re.I) and re.search(
            r"\bless\b|\bmore\b|\bmoins\b|davantage|أقل|أكبر", evidence, re.I):
        return "Explicit comparison: ask how the stated conditions or outcomes differ. The target describes the contrast, not its explanation; do not ask a why-question for this target."
    if re.search(r"(?:convert\w*|يحول|تحول).+(?:into|en|إلى)$", before, re.I):
        return "Conversion output: ask what the source device converts its input into. The target names the resulting output, not the converting device."
    if fact["question_type"] == "cause_effect":
        return "Explicit reason or prevention purpose: ask why the stated action is taken. Preserve negation and the action as described."
    if (re.search(r"(?:used to|used for|designed to|purpose .{0,50} was to|servent à|sert à|servait à|permet de)$", before, re.I)
            or re.search(r"(?:تستخدم|يستخدم|تستعمل|يستعمل).+\sل$", before)
            or re.search(r"(?:الهدف من|الغرض من)", before)):
        return "Intended use or purpose: ask what the thing/data is used for. Do not phrase the data as an agent using another object. Arabic pattern: ما الغرض من استخدام ...؟"
    if re.search(r"(?:does not preserve|ne conserve pas|لا يحتفظ ب)$", before, re.I):
        return "Missing information: ask which information/property is NOT retained. Preserve the negative relation and name the representation."
    if re.search(r"(?:records|enregistre|يسجل)$", before, re.I):
        return "Recorded information: ask what information the representation records, rather than what records that information."
    if re.search(r"(?:بحسب|according to|selon)$", before, re.I):
        return "Criterion: ask on what basis the described adjustment is made."
    return "Ask for the specific stated fact supplied as the target; keep subjects, predicates, scope and exceptions faithful."


def focused_payload(context: str, language: str, facts: list[dict], drafts: list[dict], objective: str,
                    *, include_drafts: bool = True) -> dict:
    if len(facts) != len(drafts):
        raise ValueError("Every planned fact must retain its corresponding T5 draft.")
    payload = {"language": LANGUAGES[language], "learning_objective": objective, "passage": context,
               "facts": [{**fact, "index": index, "answer_focus": answer_focus(fact)}
                         for index, fact in enumerate(facts)]}
    if include_drafts:
        for index, fact in enumerate(payload["facts"]):
            fact["draft_question"] = drafts[index]["question"]
    return payload
