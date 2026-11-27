"""Regressions for educational data alignment and source preservation."""

from questionforge.data import validate_record
from scripts.prepare_science_data import convert


def test_science_alignment_preserves_source_case_and_rejects_ambiguous_answers():
    rows, excluded = convert([
        {"support": "Chlorophyll captures the light energy that plants use.", "correct_answer": "chlorophyll",
         "question": "Which pigment captures light energy?"},
        {"support": "Heat moves from a hotter body. Heat can be transferred.", "correct_answer": "heat",
         "question": "What moves from a hotter body?"},
        {"support": "A starting motor powers the vehicle before driving.", "correct_answer": "star",
         "question": "What powers the vehicle?"},
    ], "train")
    assert len(rows) == 1 and rows[0]["answer"] == "Chlorophyll"
    validate_record(rows[0])
    assert excluded["answer_missing_or_ambiguous"] == 2


def test_duplicate_questions_with_different_support_are_not_extra_training_examples():
    question = "Which pigment captures light energy?"
    rows, excluded = convert([
        {"support": "Chlorophyll captures the light energy that plants use.", "correct_answer": "chlorophyll", "question": question},
        {"support": "The green pigment chlorophyll captures energy from light.", "correct_answer": "chlorophyll", "question": question},
    ], "train")
    assert len(rows) == 1 and excluded["duplicate_support_group_or_question"] == 1
