"""A heldout row must not slip past paragraph, question or known-article protection."""

from questionforge.data import context_hash
from scripts.prepare_generalization_holdout import key, sample_unseen


def records():
    rows = []
    for i in range(8):
        text = f"Device {i} measures the water level and sends each recorded value to the controller. The controller stores those readings for later comparison."
        rows.append({"id": str(i), "context": text, "group": context_hash(text),
                     "question": f"What does device {i} measure?", "title": f"Article {i}"})
    return rows


def test_sampling_excludes_all_three_protection_types_and_duplicate_articles():
    rows = records()
    rows[4]["title"] = rows[3]["title"]
    selected, _ = sample_unseen(rows, "en", {rows[0]["group"]}, {key(rows[1]["question"])},
                                {("en", key(rows[2]["title"]))}, 4, 20261128)
    assert not {"0", "1", "2"} & {r["id"] for r in selected}
    assert len({key(r["title"]) for r in selected}) == len(selected)


def test_sample_order_is_reproducible_independently_of_input_order():
    rows = records()
    first, _ = sample_unseen(rows, "en", set(), set(), set(), 4, 20261128)
    second, _ = sample_unseen(rows[::-1], "en", set(), set(), set(), 4, 20261128)
    assert [r["id"] for r in first] == [r["id"] for r in second]
