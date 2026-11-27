"""Protect against invalid human-evaluation claims and misleading control comparisons."""

import pytest

from questionforge.review import ratings_csv, summarize_ratings, validate_ratings
from scripts.study_verifiers import build_controls, summarize


def review_fixture():
    items = [{"item_id": method} for method in ["baseline", "greedy", "beam4"]]
    mapping = [{"item_id": i["item_id"], "method": i["item_id"], "dataset": "classroom",
                "source_id": "topic:en:0", "group": "topic", "language": "en"} for i in items]
    return items, mapping


def test_empty_review_is_pending_and_has_no_fabricated_means():
    items, mapping = review_fixture()
    summary = summarize_ratings([], items, mapping)
    assert summary["status"] == "pending_human_annotation"
    assert summary["method_means_0_to_2"] == []
    assert summary["inter_rater_agreement"] == []


@pytest.mark.parametrize("bad", ["3", "-1", "1.0", "yes"])
def test_invalid_ordinal_scores_rejected(bad):
    items, _ = review_fixture()
    text = ratings_csv([{"reviewer_id": "r1", "item_id": "baseline", "relevance": bad}])
    with pytest.raises(ValueError, match="must be"):
        validate_ratings(text, items)


def test_unknown_item_and_duplicate_votes_rejected():
    items, _ = review_fixture()
    row = {"reviewer_id": "r1", "item_id": "unknown"}
    with pytest.raises(ValueError, match="known item"):
        validate_ratings(ratings_csv([row]), items)
    row["item_id"] = "baseline"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_ratings(ratings_csv([row, row]), items)


def test_partial_annotations_do_not_create_unpaired_method_comparisons():
    items, mapping = review_fixture()
    row = {"reviewer_id": "r1", "item_id": "baseline", "relevance": 2, "answerability": 2, "fluency": 1}
    rows = validate_ratings(ratings_csv([row]), items)
    result = summarize_ratings(rows, items, mapping)
    assert result["complete_rating_rows"] == 1
    assert result["complete_paired_blocks"] == 0
    assert result["method_means_0_to_2"] == []


def test_two_reviewers_and_zero_ratings_are_counted_correctly():
    items, mapping = review_fixture()
    rows = [{"reviewer_id": reviewer, "item_id": item["item_id"],
             "relevance": score, "answerability": score, "fluency": score}
            for reviewer in ["r1", "r2"] for score, item in enumerate(items)]
    result = summarize_ratings(validate_ratings(ratings_csv(rows), items), items, mapping)
    assert result["complete_paired_blocks"] == 2
    assert len(result["method_means_0_to_2"]) == 6
    assert all(r["quadratic_weighted_kappa"] == 1 for r in result["inter_rater_agreement"])
    assert next(r["relevance"] for r in result["method_means_0_to_2"] if r["method"] == "baseline") == 0


def test_constant_identical_ratings_have_undefined_kappa_not_fake_one():
    items, mapping = review_fixture()
    rows = [{"reviewer_id": reviewer, "item_id": item["item_id"],
             "relevance": 2, "answerability": 2, "fluency": 2}
            for reviewer in ["r1", "r2"] for item in items]
    result = summarize_ratings(validate_ratings(ratings_csv(rows), items), items, mapping)
    assert all(r["quadratic_weighted_kappa"] is None for r in result["inter_rater_agreement"])
    assert all(r["exact_agreement"] == 1 for r in result["inter_rater_agreement"])


def test_controls_keep_mismatch_distinct_from_unanswerability():
    records = [{"id": f"t:en:{i}", "language": "en", "group": "t", "context": "Lina planted nine seeds.",
                "question": question, "answer": answer} for i, (question, answer) in enumerate([
                    ("Who planted the seeds?", "Lina"), ("How many seeds were planted?", "nine")])]
    controls = build_controls(records)
    assert len(controls) == 8
    mismatch = next(r for r in controls if r["source_id"] == "t:en:0" and r["condition"] == "target_mismatch")
    assert mismatch["answer"] == "nine"
    assert mismatch["question"] == "Who planted the seeds?"
    assert not mismatch["target_consistent_by_construction"]


def test_failed_control_inference_cannot_earn_true_negative_credit():
    with pytest.raises(ValueError, match="failures"):
        summarize([{"error": "inference failed"}])


@pytest.mark.integration
@pytest.mark.parametrize("language,context,question,answer", [
    ("en", "Lina planted nine seeds.", "Who planted the seeds?", "Lina"),
    ("fr", "Lina a planté neuf graines.", "Qui a planté les graines ?", "Lina"),
    ("ar", "زرعت لينا تسع بذور.", "من زرع البذور؟", "لينا"),
])
def test_optional_multilingual_verifier_runs_on_all_languages(language, context, question, answer):
    from questionforge.verification import check_answer

    result = check_answer(question, context, answer, language, "multilingual")
    assert isinstance(result["qa_answer"], str)
    assert 0 <= result["answer_f1"] <= 1
    assert 0 <= result["qa_score"] <= 1
