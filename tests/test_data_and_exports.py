import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from questionforge.data import (
    assert_disjoint,
    context_hash,
    grouped_split,
    load_jsonl,
    normalize_records,
    sample_groups,
)
from questionforge.export import as_csv, as_json, moodle_xml, worksheet_html
from questionforge.metrics import quality_decision


def record(group, number=0):
    return {"group": group, "id": f"{group}:{number}"}


def test_group_split_removes_train_test_paragraph_leakage():
    train = [record("same", i) for i in range(3)] + [record(f"new{i}") for i in range(10)]
    splits, audit = grouped_split(train, [record("same", 99)], seed=42)
    assert_disjoint(splits)
    assert audit["removed_training_rows_overlapping_test"] == 3
    assert len(splits["dev"]) >= 1


def test_group_sampler_is_reproducible_and_uses_one_question_per_paragraph():
    rows = [record(str(i), j) for i in range(10) for j in range(3)]
    first = sample_groups(rows, 6, 42)
    assert first == sample_groups(rows, 6, 42)
    assert len(first) == len({row["group"] for row in first}) == 6


def test_normalization_does_not_silently_fix_invalid_translated_answer_offsets():
    raw = [{"id": "a", "context": "Paris est une ville.", "question": "Quelle ville ?",
            "answers": {"text": ["Paris"], "answer_start": [4]}}]
    rows, excluded = normalize_records(raw, "fr")
    assert not rows
    assert excluded["invalid_span_or_record"] == 1


def test_context_hash_normalizes_only_case_whitespace_and_unicode_forms():
    assert context_hash("Café  PARIS") == context_hash("cafe\u0301 paris")
    assert context_hash("école") != context_hash("ecole")


@pytest.mark.parametrize("starts", [[], [None], ["not an offset"]])
def test_missing_or_invalid_offsets_are_counted_without_aborting_the_dataset(starts):
    raw = [{"id": "a", "context": "Paris est une ville.", "question": "Quelle ville ?",
            "answers": {"text": ["Paris"], "answer_start": starts}}]
    rows, excluded = normalize_records(raw, "fr")
    assert not rows
    assert excluded["invalid_span_or_record"] == 1


def test_novel_translations_share_topics_without_split_leakage():
    root = Path(__file__).resolve().parents[1] / "data/novel"
    splits = {split: load_jsonl(root / f"{split}.jsonl") for split in ["dev", "test"]}
    assert_disjoint(splits)
    assert len(splits["test"]) == 72
    for topic in {r["group"] for r in splits["test"]}:
        assert {r["language"] for r in splits["test"] if r["group"] == topic} == {"en", "fr", "ar"}


def test_exports_preserve_arabic_escape_html_and_make_valid_moodle_xml():
    rows = [{"question": "أين <script>alert(1)</script>؟", "answer": "بيروت & باريس", "evidence": "<b>المصدر</b>", "language": "ar"}]
    assert json.loads(as_json(rows))["questions"][0]["answer"] == "بيروت & باريس"
    worksheet = worksheet_html(rows, True)
    assert "<script>" not in worksheet and "&lt;script&gt;" in worksheet
    root = ET.fromstring(moodle_xml(rows))
    assert root.find("question/answer/text").text == "بيروت & باريس"
    assert "بيروت" in as_csv(rows)


def test_student_worksheet_omits_answers():
    rows = [{"question": "Where?", "answer": "Secret answer", "evidence": "Secret source"}]
    assert "Secret answer" not in worksheet_html(rows)
    assert "Secret answer" in worksheet_html(rows, True)


def test_spreadsheet_formula_cells_are_escaped():
    assert "'=HYPERLINK" in as_csv([{"question": '=HYPERLINK("example")'}])


def test_quality_rules_reject_answer_leaks_even_with_high_qa_agreement():
    accepted, reasons = quality_decision({"question": "Is Paris Paris?", "answer_leak": True,
                                         "answer_f1": 1.0, "qa_score": 0.9})
    assert not accepted
    assert "answer appears in question" in reasons


def test_explicit_split_overlap_is_an_error():
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint({"train": [record("a")], "test": [record("a")]})
