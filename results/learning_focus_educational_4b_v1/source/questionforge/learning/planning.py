"""Resolve model-selected IDs to exact source facts; never trust generated answer text."""

from questionforge.text import sentences, validate_language

from .rules import plan_learning_facts

GOALS = {
    "en": {"concept": "Understand the selected fact in context.",
           "function": "Understand the function described in the passage.",
           "cause_effect": "Understand a stated reason or purpose."},
    "fr": {"concept": "Comprendre le fait sélectionné dans son contexte.",
           "function": "Comprendre la fonction décrite dans le passage.",
           "cause_effect": "Comprendre une raison ou un objectif indiqué."},
    "ar": {"concept": "فهم الحقيقة المختارة في سياقها.",
           "function": "فهم الوظيفة الموصوفة في النص.",
           "cause_effect": "فهم سبب أو هدف مذكور في النص."},
}


def build_fact_candidates(context: str, language: str) -> list[dict]:
    validate_language(language)
    candidates, seen = [], set()

    def add(answer, start, evidence, kind, source):
        key = (start, answer)
        if key in seen or not 4 <= len(answer) <= 240:
            return
        seen.add(key)
        candidates.append({"candidate_id": len(candidates), "answer": answer, "answer_start": start,
                           "evidence": evidence, "question_type": kind, "selection_source": source})

    for fact in plan_learning_facts(context, language, 8):
        add(fact["answer"], fact["answer_start"], fact["evidence"], fact["question_type"], "explicit_relation")
    for part in sentences(context):
        answer = part.text.rstrip(".!?؟")
        add(answer, part.start, part.text, "concept", "source_statement")
    return candidates[:48]


def resolve_candidate_choices(choices, candidates, language, count, objective=""):
    proposed, rejected, seen_evidence = [], [], set()
    for index, choice in enumerate(choices):
        candidate_id = choice.get("candidate_id") if isinstance(choice, dict) else choice
        if type(candidate_id) is not int or not 0 <= candidate_id < len(candidates):
            rejected.append({"index": index, "item": choice, "reason": "unknown source candidate ID"})
            continue
        fact = candidates[candidate_id]
        if fact["evidence"] in seen_evidence:
            rejected.append({"index": index, "item": choice, "reason": "duplicate evidence statement"})
            continue
        seen_evidence.add(fact["evidence"])
        proposed.append({**fact, "learning_goal": objective or GOALS[language][fact["question_type"]]})
        if len(proposed) == count:
            break
    return proposed, rejected
