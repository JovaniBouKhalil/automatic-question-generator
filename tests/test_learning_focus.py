"""Grounding, abstention and language regressions for the experimental learning pipeline."""

import json

import pytest

from questionforge.learning import pipeline


def fact(answer, evidence, goal="Understand the mechanism."):
    return {"answer": answer, "evidence": evidence, "learning_goal": goal, "question_type": "function"}


def test_plan_rejects_invented_evidence_and_noncontiguous_answers():
    passage = "A mesh catches leaves. Water enters a tank."
    chosen, rejected = pipeline.validate_plan(passage, [
        fact("leaves", "A mesh catches leaves."),
        fact("bacteria", "A mesh removes bacteria."),
        fact("mesh leaves", "A mesh catches leaves."),
    ], "en", 3)
    assert [r["answer"] for r in chosen] == ["leaves"]
    assert len(rejected) == 2


def test_answer_offset_is_resolved_within_the_evidence_not_first_occurrence():
    passage = "Water is collected. The mesh filters water."
    chosen, _ = pipeline.validate_plan(passage, [fact("water", "The mesh filters water.")], "en", 1)
    assert chosen[0]["answer_start"] == passage.rfind("water")
    assert passage[chosen[0]["answer_start"]:][:5] == "water"


def test_duplicate_targets_are_not_extra_questions():
    passage = "The mesh catches leaves."
    chosen, rejected = pipeline.validate_plan(passage, [fact("leaves", passage)] * 2, "en", 3)
    assert len(chosen) == len(rejected) == 1


def test_numbered_evidence_is_resolved_from_source_instead_of_generated_quotes():
    passage = "A mesh catches leaves. A tank stores water."
    item = {"answer": "water", "sentence_index": 1, "learning_goal": "Understand storage.",
            "question_type": "function", "evidence": "This generated evidence is ignored."}
    chosen, rejected = pipeline.validate_plan(passage, [item], "en", 1)
    assert not rejected
    assert chosen[0]["evidence"] == "A tank stores water."
    assert chosen[0]["answer_start"] == passage.index("water")


@pytest.mark.parametrize("index", [-1, 2, "0", True])
def test_invalid_source_indices_cannot_be_rescued_by_a_generated_quote(index):
    item = {**fact("leaves", "A mesh catches leaves."), "sentence_index": index}
    chosen, rejected = pipeline.validate_plan("A mesh catches leaves.", [item], "en", 1)
    assert not chosen and rejected[0]["reason"] == "invalid source sentence index"


@pytest.mark.parametrize("response", ['{"answer":"leaves"}', '```python\nprint(1)\n```', '[] trailing'])
def test_generated_code_or_non_json_is_not_accepted(response):
    with pytest.raises(ValueError):
        pipeline.parse_array(response)


def test_arabic_questions_cannot_silently_be_english_or_leak_the_answer():
    assert pipeline.question_issue("What catches leaves?", "شبكة", "ar")
    assert pipeline.question_issue("ما وظيفة الشبكة؟", "الشبكة", "ar")
    assert pipeline.question_issue("ما الذي يحتجز الأوراق؟", "الشبكة", "ar") is None
    assert pipeline.question_issue("ما الذي يحتجز الأوراق？", "الشبكة", "ar") is None


def test_empty_plan_abstains_without_invoking_t5(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Abstention must not generate filler questions")

    monkeypatch.setattr(pipeline, "generate_question", unexpected)
    rows, report = pipeline.create_learning_quiz("Lina arrived.", "en", complete=lambda *args: ([], "[]"))
    assert rows == [] and report["returned"] == report["planned"] == 0


def test_missing_editor_weights_give_a_download_instruction_without_network(monkeypatch):
    pipeline.load_planner.cache_clear()
    monkeypatch.setattr(pipeline, "model_source", lambda spec: (spec.model_id, {"revision": spec.revision}))
    with pytest.raises(RuntimeError, match="scripts.download_models planner"):
        pipeline.load_planner()


def test_editor_cannot_introduce_answers_or_duplicate_fact_indices(monkeypatch):
    from questionforge.models import GeneratedQuestion

    passage = "A mesh catches leaves."
    monkeypatch.setattr(pipeline, "generate_question", lambda context, answer, language, **kwargs:
                        GeneratedQuestion("Who invented the mesh?", answer, context.find(answer), passage, "en", "beam4", "test"))
    answers = iter([
        ([0], "[0]"),
        ([{"index": 0, "question": "What does the mesh catch?"},
          {"index": 0, "question": "Which objects does the mesh catch?"},
          {"index": 3, "question": "What happens next?"}], "edit"),
    ])
    rows, report = pipeline.create_learning_quiz(passage, "en", complete=lambda *args: next(answers))
    assert len(rows) == 1 and len(report["rejected_questions"]) == 2
    assert rows[0]["question"] == "What does the mesh catch?"
    assert rows[0]["draft_question"] == "Who invented the mesh?"
    assert rows[0]["educational_quality_status"] == "not_human_rated"
    assert json.loads(json.dumps(report))["returned"] == 1


def test_editor_failure_preserves_the_valid_plan_and_seq2seq_draft(monkeypatch):
    from questionforge.models import GeneratedQuestion

    passage = "A mesh catches leaves."
    monkeypatch.setattr(pipeline, "generate_question", lambda context, answer, language, **kwargs:
                        GeneratedQuestion("What does the mesh catch?", answer, context.find(answer), passage, "en", "beam4", "test"))

    def complete(system, payload, device):
        if system == pipeline.PLAN_SYSTEM:
            return [0], "valid plan"
        raise pipeline.StructuredOutputError('[{"index": 0, "question":')

    with pytest.raises(pipeline.StructuredOutputError) as caught:
        pipeline.create_learning_quiz(passage, "en", complete=complete)
    report = caught.value.learning_report
    assert report["raw_plan"] == "valid plan"
    assert report["raw_edit"].endswith('"question":')
    assert report["drafts"][0]["question"] == "What does the mesh catch?"


def test_partial_report_does_not_freeze_an_incomplete_review_template(tmp_path):
    import csv
    import io

    from scripts.run_learning_focus_study import render_report

    def case(case_id, question):
        return {"case_id": case_id, "language": "fr", "context": "=1+1\nLe filtre retient les feuilles.",
                "outputs": {"learning_hybrid": [{"question": question, "answer": "les feuilles",
                     "evidence": "Le filtre retient les feuilles.", "learning_goal": "Comprendre le filtre.",
                     "answer_f1": 1.0}]}}

    first, second = case("one", "Que retient le filtre ?"), case("two", "À quoi sert ce filtre ?")
    render_report(tmp_path, [first])
    target = tmp_path / "learning_review_template.csv"
    assert not target.exists()
    render_report(tmp_path, [first, second], write_template=True)
    rows = list(csv.DictReader(io.StringIO(target.read_text(encoding="utf-8-sig"), newline="")))
    assert len(rows) == 2
    assert all(not row["learning_value_0_2"] and not row["reviewer_id"] for row in rows)
    assert rows[0]["context"] == "'=1+1\nLe filtre retient les feuilles."
    original = target.read_bytes()
    render_report(tmp_path, [first], write_template=True)
    assert target.read_bytes() == original


@pytest.mark.parametrize("language,context,expected", [
    ("en", "The article stays in one split to avoid sharing facts.", "avoid sharing facts"),
    ("fr", "L'article reste dans une partition pour éviter de partager ses faits.", "éviter de partager ses faits"),
    ("ar", "تبقى مجموعة الاختبار منفصلة حتى التقييم النهائي حتى لا تؤثر نتيجتها في الاختيار.", "لا تؤثر نتيجتها في الاختيار"),
])
def test_purpose_selection_preserves_prevention_and_negation(language, context, expected):
    from questionforge.learning.rules import plan_learning_facts

    facts = plan_learning_facts(context, language, 1)
    assert facts[0]["answer"] == expected
    start = facts[0]["answer_start"]
    assert context[start:start + len(expected)] == expected


def test_focus_keeps_antecedent_but_drops_later_distracting_subject():
    from questionforge.learning.rules import focused_evidence

    passage = "Bag-of-words counts words. It does not preserve word order. TF-IDF weights terms."
    answer = "word order"
    window, left = focused_evidence(passage, answer, passage.index(answer), "en")
    assert "Bag-of-words" in window and "TF-IDF" not in window
    offset = passage.index(answer) - left
    assert window[offset:offset + len(answer)] == answer


@pytest.mark.parametrize("language,context", [
    ("en", "Training data is used to update model weights. The test stays separate to avoid bias."),
    ("fr", "Les données servent à choisir les paramètres. Le test reste séparé pour éviter un biais."),
    ("ar", "تستخدم بيانات التدريب لتحديث الأوزان. يبقى الاختبار منفصلا حتى لا يؤثر في الاختيار."),
])
def test_candidate_ids_resolve_to_exact_source_spans_in_each_language(language, context):
    from questionforge.learning.planning import build_fact_candidates, resolve_candidate_choices

    candidates = build_fact_candidates(context, language)
    assert candidates
    for candidate in candidates:
        start, answer = candidate["answer_start"], candidate["answer"]
        assert context[start:start + len(answer)] == answer
    proposed, rejected = resolve_candidate_choices([0], candidates, language, 3)
    assert not rejected and proposed[0]["answer"] == candidates[0]["answer"]


def test_selector_cannot_supply_new_answer_wording_or_duplicate_the_same_fact():
    from questionforge.learning.planning import build_fact_candidates, resolve_candidate_choices

    passage = "Training data is used to update weights."
    candidates = build_fact_candidates(passage, "en")
    choices = [{"candidate_id": 0, "answer": "invented answer"}, 1, -1, "new wording"]
    proposed, rejected = resolve_candidate_choices(choices, candidates, "en", 3)
    assert len(proposed) == 1 and len(rejected) == 3
    assert proposed[0]["answer"] == "update weights"
