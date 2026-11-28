"""Run the existing quiz workflow over source-aligned, token-bounded sections.

Small requests retain the original inference path. Longer inputs are scanned locally;
the requested question count determines how many eligible sections reach the models.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from questionforge.classroom.material import CLASSROOM_PLAN_SYSTEM, build_classroom_candidates
from questionforge.learning.focus import REFINEMENT_SYSTEM, focused_payload
from questionforge.metrics import lexical_relevance
from questionforge.text import LANGUAGES, evidence_for, normalize_answer

MAX_QUESTIONS = 20
SECTION_TOKENS = 1000
INPUT_TOKENS = 6000


@lru_cache(maxsize=1)
def quality_tokenizer():
    from transformers import AutoTokenizer

    from questionforge.learning.quality import local_quality_source

    return AutoTokenizer.from_pretrained(local_quality_source(), local_files_only=True, trust_remote_code=False)


def message_tokens(tokenizer, system, payload):
    return len(tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        add_generation_prompt=True, tokenize=True))


def request_fits(context, language, count, objective, candidates, tokenizer, manual=False):
    if not 1 <= count <= 8:
        return False
    planning = 0 if manual else message_tokens(tokenizer, CLASSROOM_PLAN_SYSTEM, {
        "language": LANGUAGES[language], "count": count, "learning_objective": objective,
        "passage": context, "candidates": [{"candidate_id": f.get("candidate_id", i),
            "answer": f["answer"], "evidence": f["evidence"], "kind": f["question_type"]}
            for i, f in enumerate(candidates)]})
    # Bound the largest possible selected fact payload, plus room for native T5 drafts.
    largest = sorted(candidates, key=lambda f: len(tokenizer.encode(json.dumps(f, ensure_ascii=False))), reverse=True)[:count]
    facts = [{**f, "learning_goal": objective or "Understand the selected fact in context."} for f in largest]
    editing = message_tokens(tokenizer, REFINEMENT_SYSTEM,
                            focused_payload(context, language, facts, [{"question": ""}] * len(facts), objective))
    return max(planning, editing + 512 * len(facts)) + 256 <= INPUT_TOKENS


def split_sections(context, tokenizer, budget=SECTION_TOKENS):
    """Keep every source character; prefer paragraph/sentence boundaries over hard cuts."""
    sections, start = [], 0
    while start < len(context):
        low, high = start + 1, min(len(context), start + 16000)
        while low < high:
            middle = (low + high + 1) // 2
            if len(tokenizer.encode(context[start:middle], add_special_tokens=False)) <= budget:
                low = middle
            else:
                high = middle - 1
        end = low
        if end < len(context):
            piece = context[start:end]
            for pattern in [r"\n\s*\n", r"[.!?؟]\s+", r"\n", r"\s+"]:
                cuts = [m.end() for m in re.finditer(pattern, piece) if m.end() >= len(piece) // 2]
                if cuts:
                    end = start + cuts[-1]
                    break
        sections.append({"index": len(sections), "start": start, "end": end,
                         "text": context[start:end]})
        start = end
    return sections


def section_candidates(context, language):
    pool = build_classroom_candidates(context, language)
    if len(pool) > 12:
        pool = [pool[round(i * (len(pool) - 1) / 11)] for i in range(12)]
    return [{**fact, "candidate_id": i} for i, fact in enumerate(pool)]


def _manual_facts(context, answers):
    return [{"answer": answer, "evidence": evidence_for(context, answer).text,
             "question_type": "concept"} for answer in answers]


def create_document_quiz(context, language, count=3, *, objective="", manual_answers=None,
                         progress=None, tokenizer=None, create=None):
    from questionforge.learning.quality import create_quality_quiz

    if not context.strip() or not 1 <= count <= MAX_QUESTIONS:
        raise ValueError(f"Add a document and request between one and {MAX_QUESTIONS} questions.")
    if len(objective) > 500:
        raise ValueError("Keep the learning objective within 500 characters.")
    tokenizer = tokenizer or quality_tokenizer()
    create = create or create_quality_quiz
    notice = progress or (lambda message: None)
    answers = list(dict.fromkeys(manual_answers or []))[:count]
    pool = _manual_facts(context, answers) if answers else build_classroom_candidates(context, language)

    def generate(text, requested, targets, builder):
        return create(text, language, requested, objective=objective, manual_answers=targets,
                      beams=4, checkpoint=None, progress=notice, role_focus=True,
                      candidate_builder=builder, planning_system=CLASSROOM_PLAN_SYSTEM)

    if request_fits(context, language, count, objective, pool, tokenizer, bool(answers)):
        rows, report = generate(context, count, answers, build_classroom_candidates)
        report["document"] = {"mode": "single", "source_characters": len(context),
                              "sections_total": 1, "sections_used": 1}
        return rows, report

    sections = split_sections(context, tokenizer)
    jobs = []
    if answers:
        for answer in answers:
            position = context.index(answer)
            section = next((s for s in sections if s["start"] <= position and
                            position + len(answer) <= s["end"]), None)
            if section is None:
                start, end = max(0, position - 1200), min(len(context), position + len(answer) + 1200)
                section = {"index": -1, "start": start, "end": end, "text": context[start:end]}
            radius = 1200
            while not request_fits(section["text"], language, 1, objective,
                                   _manual_facts(section["text"], [answer]), tokenizer, True):
                if radius == 0:
                    raise ValueError("Use a shorter exact target answer; this answer alone exceeds the model request budget.")
                radius //= 2
                start, end = max(0, position - radius), min(len(context), position + len(answer) + radius)
                section = {"index": -1, "start": start, "end": end, "text": context[start:end]}
            if jobs and jobs[-1][0]["start"] == section["start"] and jobs[-1][0]["end"] == section["end"] and jobs[-1][1] < 3:
                previous, n, targets = jobs[-1]
                combined = [*targets, answer]
                if request_fits(previous["text"], language, n + 1, objective,
                                _manual_facts(previous["text"], combined), tokenizer, True):
                    jobs[-1] = (previous, n + 1, combined)
                    continue
            jobs.append((section, 1, [answer]))
    else:
        eligible = []
        for section in sections:
            candidates = section_candidates(section["text"], language)
            while len(candidates) > 1 and not request_fits(section["text"], language, 3, objective, candidates, tokenizer):
                limit = max(1, len(candidates) // 2)
                candidates = [candidates[round(i * (len(candidates) - 1) / max(1, limit - 1))] for i in range(limit)]
                candidates = [{**f, "candidate_id": i} for i, f in enumerate(candidates)]
            section = {**section, "candidates": candidates}
            capacity = len({f["evidence"] for f in candidates})
            if capacity and request_fits(section["text"], language, min(3, capacity), objective, candidates, tokenizer):
                eligible.append((section, capacity))
        size = min(count, len(eligible))
        if size and objective.strip():
            chosen = sorted(eligible, key=lambda pair: lexical_relevance(objective, pair[0]["text"], language), reverse=True)[:size]
        elif size:
            chosen = [eligible[round(i * (len(eligible) - 1) / max(1, size - 1))] for i in range(size)]
        else:
            chosen = []
        chosen.sort(key=lambda pair: pair[0]["start"])
        allocations = [0] * len(chosen)
        while sum(allocations) < count:
            before = sum(allocations)
            for i, (_, capacity) in enumerate(chosen):
                if allocations[i] < capacity and sum(allocations) < count:
                    allocations[i] += 1
            if sum(allocations) == before:
                break
        jobs = [(section, min(3, n - start), []) for (section, _), n in zip(chosen, allocations)
                for start in range(0, n, 3)]

    notice(f"Document divided into {len(sections)} sections; using {len(jobs)} section request(s) for this quiz.")
    rows, records, seen, used_evidence = [], [], set(), set()
    for number, (section, requested, targets) in enumerate(jobs, 1):
        notice(f"Section {number}/{len(jobs)}: creating up to {requested} question(s)…")
        record = {k: section[k] for k in ["index", "start", "end"]}
        record.update(requested=requested, error=None)
        try:
            def remaining_candidates(text, selected_language):
                pool = section.get("candidates", section_candidates(text, selected_language))
                available = [f for f in pool if f["evidence"] not in used_evidence]
                return [{**f, "candidate_id": i} for i, f in enumerate(available)]

            local, report = generate(section["text"], requested, targets, remaining_candidates)
            record["planning_report"] = report
            used_evidence.update(f["evidence"] for f in report.get("facts", local))
            for generated in local:
                row = dict(generated)
                row["answer_start"] += section["start"]
                assert context[row["answer_start"]:row["answer_start"] + len(row["answer"])] == row["answer"]
                key = normalize_answer(row["question"], language)
                if key not in seen:
                    seen.add(key)
                    row["source_section"] = {k: section[k] for k in ["index", "start", "end"]}
                    rows.append(row)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if hasattr(exc, "raw_text"):
                record["failed_model_response"] = exc.raw_text
            if hasattr(exc, "learning_report"):
                record["planning_report"] = exc.learning_report
        records.append(record)
    return rows, {"document": {"mode": "sectioned", "source_characters": len(context),
                   "sections_total": len(sections),
                   "sections_used": len({(s["start"], s["end"]) for s, _, _ in jobs}),
                   "section_requests": len(jobs),
                   "coverage": "All sections scanned locally; model inference uses the selected sections, not every fact in the document."},
                  "requested": count, "returned": len(rows), "sections": records,
                  "section_errors": sum(bool(r["error"]) for r in records),
                  "section_offset_space": "Prepared document; nested planning reports use section-local offsets.",
                  "educational_quality_status": "not_human_rated"}
