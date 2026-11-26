import pytest

from questionforge.baseline import template_question
from questionforge.metrics import leaks_answer, lexical_relevance, rouge_l
from questionforge.models import answer_window, format_prompt
from questionforge.text import answer_f1, evidence_for, normalize_answer, sentences, tokens


@pytest.mark.parametrize("text,answer,language", [
    ("Lina planted nine seeds.", "Lina", "en"),
    ("L'énergie solaire alimente la lampe.", "la lampe", "fr"),
    ("زرعت لينا تسع بذور.", "لينا", "ar"),
])
def test_template_removes_answer_and_retains_other_source_words(text, answer, language):
    result = template_question(text, answer, language)
    assert answer not in result
    assert result.endswith("؟" if language == "ar" else "?")


def test_repeated_answers_use_requested_occurrence():
    text = "Lina read a book. Omar lent Lina a pen."
    start = text.rfind("Lina")
    assert evidence_for(text, "Lina", start).text == "Omar lent Lina a pen."
    assert template_question(text, "Lina", "en", start) == "Omar lent what a pen?"


def test_invalid_spans_are_rejected():
    with pytest.raises(ValueError, match="exact span"):
        evidence_for("Paris is a city.", "London")
    with pytest.raises(ValueError):
        evidence_for("Paris is a city.", "Paris", 1)


def test_unicode_metrics_keep_arabic_and_french():
    assert tokens("école العربية") == ["école", "العربية"]
    assert rouge_l("أين تقع المدرسة؟", "أين تقع المدرسة؟") == 1
    assert rouge_l("Où est l'école ?", "Où est l'école ?") == 1
    assert rouge_l("أين تقع المدرسة؟", "ما اسم المعلم؟") == 0
    assert answer_f1("the red book", "red book", "en") == 1
    assert answer_f1("الكِتاب", "الكتاب", "ar") == 1
    assert normalize_answer("école", "fr") != normalize_answer("ecole", "fr")


def test_answer_leak_respects_token_boundaries():
    assert leaks_answer("What is Paris known for?", "Paris", "en")
    assert not leaks_answer("What is Parisian culture?", "Paris", "en")


def test_lexical_relevance_does_not_reward_empty_or_unrelated_questions():
    assert lexical_relevance("", "The lamp uses sunlight.", "en") == 0
    assert lexical_relevance("What powers the lamp?", "The lamp uses sunlight.", "en") > 0
    assert lexical_relevance("Who wrote poetry?", "The lamp uses sunlight.", "en") == 0


def test_sentence_offsets_include_unpunctuated_lines():
    text = "  First line\nDeuxième phrase.\nجملة عربية؟"
    parts = sentences(text)
    assert len(parts) == 3
    assert all(text[part.start:part.end] == part.text for part in parts)


def test_decimal_answer_is_never_split_out_of_evidence():
    text = "The meter measured 3.5 volts."
    assert "3.5" in evidence_for(text, "3.5").text


def test_prompt_highlights_only_the_requested_occurrence():
    text = "Paris and Paris"
    assert format_prompt(text, "Paris", 10, "fr_highlight") == "Paris and <hl> Paris <hl>"
    with pytest.raises(ValueError):
        format_prompt(text, "Paris", -1, "fr_highlight")


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}


def test_long_input_crop_keeps_answer_and_prompt_within_budget():
    context = "x " * 500 + "بيروت" + " y" * 500
    answer = "بيروت"
    window, offset, cropped = answer_window(context, answer, context.find(answer), CharacterTokenizer(), "fr_highlight", 80)
    assert cropped
    assert window[offset:offset + len(answer)] == answer
    assert len(format_prompt(window, answer, offset, "fr_highlight")) <= 80
    with pytest.raises(ValueError, match="alone exceeds"):
        answer_window(context, answer, context.find(answer), CharacterTokenizer(), "context_answer", 4)
