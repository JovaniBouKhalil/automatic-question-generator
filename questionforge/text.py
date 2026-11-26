"""Unicode-aware text handling. Preserve accents, Arabic letters and original offsets."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

LANGUAGES = {"en": "English", "fr": "Français", "ar": "العربية"}
STOPWORDS = {
    "en": set("a an the is are was were be been of in on to for from and or at by with as this that it its".split()),
    "fr": set("le la les un une des du de d l est sont était étaient dans sur à au aux et ou en pour par ce cette ces il elle son sa ses".split()),
    "ar": set("في من إلى على عن مع هو هي هذا هذه ذلك تلك التي الذي و أو كان كانت أن إن ما لا ال تم وقد حيث بين خلال".split()),
}


def validate_language(language: str) -> str:
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}; choose en, fr or ar.")
    return language


def tokens(text: str) -> list[str]:
    """Python's Unicode word matching retains Arabic and French, unlike ASCII ROUGE."""
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold(), re.UNICODE)


def normalize_answer(text: str, language: str = "en") -> str:
    validate_language(language)
    text = unicodedata.normalize("NFKC", text).casefold()
    if language == "ar":
        # Remove Arabic vowel marks and elongation; retain letters (no aggressive stemming).
        text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)
    words = tokens(text)
    if language == "en":
        words = [word for word in words if word not in {"a", "an", "the"}]
    return " ".join(words)


def answer_f1(prediction: str, reference: str, language: str = "en") -> float:
    pred = normalize_answer(prediction, language).split()
    ref = normalize_answer(reference, language).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    return 2 * common / (len(pred) + len(ref))


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


def sentences(text: str) -> list[Sentence]:
    """Split at sentence punctuation/newlines while keeping exact source character spans."""
    result = []
    for match in re.finditer(r"[^.!?؟\n]+(?:[.!?؟]+|$)", text):
        raw = match.group()
        stripped = raw.strip()
        if stripped:
            start = match.start() + len(raw) - len(raw.lstrip())
            result.append(Sentence(stripped, start, start + len(stripped)))
    return result


def evidence_for(context: str, answer: str, answer_start: int | None = None) -> Sentence:
    if not answer.strip():
        raise ValueError("Answer cannot be empty.")
    start = context.find(answer) if answer_start is None else answer_start
    if start < 0 or context[start:start + len(answer)] != answer:
        raise ValueError("The answer must be an exact span in the passage.")
    for sentence in sentences(context):
        if sentence.start <= start and start + len(answer) <= sentence.end:
            return sentence
    # Decimal values/abbreviations may cross the simple splitter; preserve the whole span.
    left = max(0, context.rfind("\n", 0, start) + 1)
    right = context.find("\n", start + len(answer))
    right = len(context) if right == -1 else right
    return Sentence(context[left:right], left, right)
