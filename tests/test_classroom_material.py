"""Regressions for heading targets, wrapped source facts and requested answer scope."""

import pytest

from questionforge.classroom.material import (
    build_classroom_candidates,
    list_scope_issue,
    prepare_material,
    select_classroom_targets,
)


@pytest.mark.parametrize("raw,expected", [
    ("A filter is used to remove large\nparticles from water.", "A filter is used to remove large particles from water."),
    ("Une initiative permet de renforcer les pratiques\nresponsables.", "Une initiative permet de renforcer les pratiques responsables."),
    ("تستخدم هذه الشبكة في التجربة لحجز الأوراق\nالكبيرة في الماء.", "تستخدم هذه الشبكة في التجربة لحجز الأوراق الكبيرة في الماء."),
])
def test_clear_physical_wraps_are_joined_in_each_language(raw, expected):
    assert prepare_material(raw) == expected


def test_headings_bullets_and_paragraphs_keep_their_boundaries():
    raw = "ENVIRONNEMENT\nUne mesure réduit les déchets.\n\nLes actions possibles\n• réduire les déchets\n• mesurer les résultats"
    assert prepare_material(raw) == raw


def test_automatic_targets_skip_outline_labels_and_keep_exact_prepared_spans():
    raw = "Les axes du programme\n7–9 ENVIRONNEMENT\nSynthèse\nUne initiative vise à agir pour encourager des pratiques\nresponsables."
    source = prepare_material(raw)
    selected = select_classroom_targets(source, "fr", 3)
    assert selected and selected[0]["answer"] == "encourager des pratiques responsables"
    assert not any(f["answer"] in {"ENVIRONNEMENT", "Synthèse", "7–9"} for f in selected)
    for fact in selected:
        assert source[fact["answer_start"]:][:len(fact["answer"])] == fact["answer"]


def test_late_explanations_are_not_hidden_by_many_earlier_statements():
    source = "\n".join(f"Sensor {i} measures the current water level." for i in range(70))
    source += "\nThe four priorities are: safety, quality, reliability and transparency."
    facts = build_classroom_candidates(source, "en")
    assert len(facts) <= 48
    assert any(f["answer"] == "safety, quality, reliability and transparency" for f in facts)
    assert all(source[f["answer_start"]:][:len(f["answer"])] == f["answer"] for f in facts)


@pytest.mark.parametrize("language,question,answer", [
    ("fr", "Quels sont les 4 domaines du programme ?", "ENVIRONNEMENT"),
    ("en", "What are the four project priorities?", "safety"),
    ("ar", "اذكر 4 مجالات للمشروع؟", "البيئة"),
])
def test_a_single_label_cannot_answer_an_explicit_multi_item_question(language, question, answer):
    assert list_scope_issue(question, answer, language)


@pytest.mark.parametrize("language,question,answer", [
    ("fr", "Quels sont les 4 axes ?", "sécurité, qualité, fiabilité et transparence"),
    ("en", "What are the four priorities?", "safety, quality, reliability and transparency"),
    ("ar", "اذكر 4 مجالات؟", "السلامة، الجودة، الكفاءة والاستدامة"),
    ("fr", "Quel est l'objectif du principe 4 ?", "éliminer le travail forcé"),
    ("en", "Which of the four priorities concerns risk?", "safety"),
])
def test_complete_lists_and_single_answers_to_numbered_references_are_allowed(language, question, answer):
    assert list_scope_issue(question, answer, language) is None


def test_source_questions_are_not_proposed_as_facts():
    source = "Which priority is most difficult, and why?\nThe project reduces waste to avoid unnecessary costs."
    facts = build_classroom_candidates(source, "en")
    assert facts and all("Which priority" not in f["answer"] for f in facts)
