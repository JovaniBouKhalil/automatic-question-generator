"""A transparent, untrained baseline: replace the target answer with a WH phrase.

The baseline has the same passage and target answer as answer-aware Transformers.
It establishes whether a neural model improves on cheap source-grounded rewriting.
"""

import re

from .text import evidence_for, validate_language


def template_question(context: str, answer: str, language: str, answer_start: int | None = None) -> str:
    validate_language(language)
    evidence = evidence_for(context, answer, answer_start)
    start = context.find(answer) if answer_start is None else answer_start
    offset = start - evidence.start
    numeric = bool(re.fullmatch(r"[\d\s.,%٠-٩٫٬]+", answer.strip()))
    wh = {
        "en": "how many" if numeric else "what",
        "fr": "combien" if numeric else "quoi",
        "ar": "كم" if numeric else "ما",
    }[language]
    question = evidence.text[:offset] + wh + evidence.text[offset + len(answer):]
    question = question.rstrip(" .!?؟")
    if language != "ar":
        question = question[0].upper() + question[1:]
    return question + ("؟" if language == "ar" else "?")
