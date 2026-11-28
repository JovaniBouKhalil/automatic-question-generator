"""Prepare slide/PDF text and select source-supported classroom targets.

Offsets refer to the displayed prepared passage; the application retains the raw input.
These structural checks do not establish entailment or educational usefulness.
"""

from __future__ import annotations

import re

from questionforge.learning.pipeline import PLAN_SYSTEM
from questionforge.learning.rules import plan_learning_facts
from questionforge.text import STOPWORDS, normalize_answer, sentences, tokens, validate_language

VERSION = "classroom-layout-v1"
CLASSROOM_PLAN_SYSTEM = PLAN_SYSTEM + """
Cover distinct lesson concepts rather than repeated formulations of the same fact.
Prefer complete explanatory answers and complete lists when a question would request a list.
Candidate answers and evidence come from the prepared source, not from headings alone.
Never treat a question copied from the source as an answered fact."""


def _list_start(line: str) -> bool:
    return bool(re.match(r"^(?:[•▪●◦]|\d+\s*[.)—–-]|\d+\s{2,})", line))


def prepare_material(raw: str) -> str:
    """Join clear physical line wraps, keeping headings, lists and paragraphs separate."""
    output, pending = [], ""
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            if pending:
                output.append(pending)
                pending = ""
            if output and output[-1]:
                output.append("")
            continue
        letter = next((c for c in line if c.isalpha()), "")
        last = tokens(pending)[-1] if tokens(pending) else ""
        connector = last in set.union(*STOPWORDS.values())
        prose = len(tokens(pending)) >= 4 or connector or pending.endswith((",", "،", "-"))
        capitals = [c for c in pending if c.isalpha() and (c.isupper() or c.islower())]
        heading = bool(capitals) and all(c.isupper() for c in capitals)
        wrap = (pending and prose and not heading and not re.search(r"[.!?؟;:»”]$", pending)
                and not _list_start(line) and (bool(letter) and not letter.isupper()
                                              or pending.endswith((",", "،", "-")) or connector))
        if wrap:
            pending += ("" if pending.endswith("-") else " ") + line
        else:
            if pending:
                output.append(pending)
            pending = line
    if pending:
        output.append(pending)
    return "\n".join(output).strip()


def is_fact_statement(text: str) -> bool:
    text = text.strip()
    words = tokens(text)
    if len(words) < 3 or text.endswith(("?", "؟")):
        return False
    # A heading can mention a central concept without asserting a fact about it.
    if not text.endswith((".", "!")):
        return False
    return True


def _spread(items: list, limit: int) -> list:
    if len(items) <= limit:
        return items
    if limit == 1:
        return items[:1]
    return [items[round(i * (len(items) - 1) / (limit - 1))] for i in range(limit)]


def build_classroom_candidates(context: str, language: str) -> list[dict]:
    validate_language(language)
    proposed, seen = [], set()

    def add(answer, start, evidence, kind, source):
        key = normalize_answer(answer, language)
        if key in seen or not 4 <= len(answer) <= 240 or not is_fact_statement(evidence):
            return
        if context[start:start + len(answer)] != answer:
            raise ValueError("Prepared target is not an exact source span.")
        seen.add(key)
        proposed.append({"answer": answer, "answer_start": start, "evidence": evidence,
                         "question_type": kind, "selection_source": source})

    for fact in plan_learning_facts(context, language, 8):
        add(fact["answer"], fact["answer_start"], fact["evidence"], fact["question_type"], "explicit_relation")
    for part in sentences(context):
        if not is_fact_statement(part.text):
            continue
        patterns = []
        if language == "fr":
            patterns = [("function", "purpose_span", r"\bpour\s+(?P<answer>[^\W\d_]+(?:er|ir|re)\b[^.!?;]+)"),
                        ("function", "transformation_output", r"\b(?:transforme|traduit|traduisent|transforment)\b.+?\ben\s+(?P<answer>[^.!?;]+)")]
        for kind, source, pattern in patterns:
            for match in re.finditer(pattern, part.text, re.I):
                answer = match.group("answer").strip()
                add(answer, part.start + match.start("answer"), part.text, kind, source)
        if ":" in part.text:
            suffix = part.text.split(":", 1)[1].strip().rstrip(".!?؟")
            if len(re.findall(r"[,،;•]", suffix)) >= 2:
                add(suffix, part.start + part.text.index(suffix), part.text, "concept", "enumerated_list")
    relations = proposed[:]
    for part in sentences(context):
        add(part.text.rstrip(".!?؟"), part.start, part.text, "concept", "source_statement")
    statements = proposed[len(relations):]
    selected_relations = _spread(relations, min(24, len(relations)))
    selected = [*selected_relations, *_spread(statements, 48 - len(selected_relations))]
    return [{"candidate_id": i, **fact} for i, fact in enumerate(selected)]


def select_classroom_targets(context: str, language: str, count: int) -> list[dict]:
    """Prefer relation/list targets; broad statement targets stay with the hybrid planner."""
    from questionforge.candidates import extract_candidates

    facts = build_classroom_candidates(context, language)
    relations = [f for f in facts if f["selection_source"] != "source_statement"]
    chosen, evidence, categories = [], set(), set()
    for diverse in [True, False]:
        for fact in relations:
            if fact["evidence"] in evidence or diverse and fact["selection_source"] in categories:
                continue
            chosen.append(fact)
            evidence.add(fact["evidence"])
            categories.add(fact["selection_source"])
            if len(chosen) == count:
                return chosen
    for candidate in extract_candidates(context, language, 20):
        if not is_fact_statement(candidate.evidence) or candidate.evidence in evidence:
            continue
        chosen.append({"answer": candidate.answer, "answer_start": candidate.start, "evidence": candidate.evidence})
        evidence.add(candidate.evidence)
        if len(chosen) == count:
            break
    return chosen


def list_scope_issue(question: str, answer: str, language: str) -> str | None:
    """Catch explicit multi-item requests paired with an obviously incomplete target."""
    numbers = {"two": 2, "three": 3, "four": 4, "five": 5, "deux": 2, "trois": 3,
               "quatre": 4, "cinq": 5, "اثنين": 2, "ثلاثة": 3, "أربعة": 4}
    patterns = {
        "en": r"\b(?:what|which)\s+are\s+(?:the\s+)?(?P<n>\d{1,2}|two|three|four|five)\b",
        "fr": r"\bquell?e?s?\s+sont\s+(?:les\s+)?(?P<n>\d{1,2}|deux|trois|quatre|cinq)\b",
        "ar": r"(?:اذكر|عدد)\s+(?P<n>\d{1,2}|اثنين|ثلاثة|أربعة)\b",
    }
    match = re.search(patterns[language], question, re.I)
    if not match:
        return None
    count = numbers.get(match["n"], int(match["n"]) if match["n"].isdigit() else 0)
    if not 2 <= count <= 20:
        return None
    separator = r"[,،;•\n]+|\s+(?:and|et)\s+"
    if language == "ar":
        separator += r"|\s+و(?=[\u0621-\u064a])"
    items = [item for item in re.split(separator, answer) if item.strip()]
    if len(items) < count and (len(items) > 1 or len(tokens(answer)) <= 6):
        return f"Question requests {count} items but the target does not supply that list."
    return None
