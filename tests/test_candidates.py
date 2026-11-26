import pytest

pytest.importorskip("sklearn")
from questionforge.candidates import extract_candidates  # noqa: E402


@pytest.mark.parametrize("context,language", [
    ("Lina planted nine seeds. Omar built a robot.", "en"),
    ("Lina cultive des haricots. Omar construit un robot.", "fr"),
    ("زرعت لينا تسع بذور. صنع عمر روبوتا صغيرا.", "ar"),
])
def test_candidates_are_exact_nonoverlapping_source_spans(context, language):
    candidates = extract_candidates(context, language, 3)
    assert candidates
    for candidate in candidates:
        assert context[candidate.start:candidate.start + len(candidate.answer)] == candidate.answer
        assert candidate.answer in candidate.evidence
    assert len({c.evidence for c in candidates}) == len(candidates)


def test_all_stopwords_and_empty_context_have_no_candidates():
    assert extract_candidates("the a an", "en") == []
    assert extract_candidates("", "ar") == []


def test_arabic_name_suggestion_does_not_attach_the_following_numeral():
    choices = extract_candidates("زرعت لينا تسع بذور في وعاء.", "ar", 1)
    assert choices[0].answer in {"لينا", "تسع", "تسع بذور"}


def test_quantity_does_not_end_with_a_cut_compound_modifier():
    choices = extract_candidates("Lina planted twelve bean seeds in a tray.", "en", 1)
    assert choices[0].answer != "twelve bean"
