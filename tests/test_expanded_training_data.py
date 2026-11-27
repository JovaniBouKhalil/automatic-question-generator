"""Prevent larger-data training from leaking held-out paragraphs or article facts."""

from scripts.prepare_expanded_data import exclude_published_sources, sample_diverse, split_expanded
from scripts.train_expanded_lora import training_order


def row(identifier, group, article=""):
    return {"id": identifier, "group": group, "title": article}


def test_every_published_evaluation_paragraph_is_protected_not_only_old_test_sample():
    training = [row("a", "hidden-eval"), row("b", "safe1"), row("c", "safe2")]
    evaluation = [row("unseen-before", "hidden-eval")]
    selected, development, _ = split_expanded(training, evaluation, [], 10, 1)
    assert all(r["group"] != "hidden-eval" for r in selected + development)


def test_invalid_evaluation_answer_offsets_do_not_release_articles_for_training():
    # Protection records deliberately have no usable answer annotation at all.
    evaluation = [{"group": "bad-offset-source", "title": "Held article"}]
    training = [row("same-text", "bad-offset-source"), row("same-article", "different-text", "HELD article"),
                row("safe", "safe-text", "Another article")]
    kept, removed = exclude_published_sources(training, evaluation)
    assert [r["id"] for r in kept] == ["safe"]
    assert {r["id"] for r in removed} == {"same-text", "same-article"}


def test_article_isolation_removes_different_paragraphs_from_same_protected_article():
    training = [row("a", "different-paragraph", "Protected Article"),
                row("b", "safe1", "Other1"), row("c", "safe2", "Other2")]
    held = [row("old-dev", "dev-paragraph", "protected article")]
    selected, development, audit = split_expanded(training, [], held, 10, 2)
    assert "a" not in {r["id"] for r in selected}
    assert held[0] in development
    assert audit["known_article_title_intersection"] == []


def test_validation_budget_counts_distinct_paragraphs_instead_of_repeated_questions():
    training = [row(f"{article}:{paragraph}:{q}", f"{article}:{paragraph}", f"article {article}")
                for article in range(8) for paragraph in range(4) for q in range(6)]
    selected, development, _ = split_expanded(training, [], [], 30, 10)
    assert len(development) == len({r["group"] for r in development}) == 10
    assert not ({r["title"] for r in selected} & {r["title"] for r in development})


def test_sampling_exhausts_paragraph_diversity_before_repeating_a_passage():
    rows = [row(f"{group}:{q}", str(group)) for group in range(5) for q in range(4)]
    selected = sample_diverse(rows, 8)
    assert len({r["group"] for r in selected[:5]}) == 5
    assert len({r["id"] for r in selected}) == 8
    assert selected == sample_diverse(list(reversed(rows)), 8)


def test_resumable_epoch_order_is_independent_of_other_random_draws():
    import random

    first = training_order(50, 42, 0)
    random.seed(999)
    random.random()
    assert first == training_order(50, 42, 0)
    assert sorted(first) == list(range(50))
    assert first != training_order(50, 42, 1)
