"""Optional real-model checks: python -m pytest -m integration -o addopts='' ."""

import pytest

from questionforge.models import generate_question, verify_question


@pytest.mark.integration
@pytest.mark.parametrize("language,text,answer", [
    ("en", "Lina planted nine bean seeds in a tray.", "Lina"),
    ("fr", "Lina a planté neuf graines de haricot dans un bac.", "Lina"),
    ("ar", "زرعت لينا تسع بذور فاصوليا في وعاء.", "لينا"),
])
def test_real_generator_and_independent_verifier(language, text, answer):
    result = generate_question(text, answer, language, beams=4)
    assert result.question.strip()
    assert result.input_tokens <= 384
    checked = verify_question(result.question, text, answer, language)
    assert 0 <= checked["answer_f1"] <= 1
    assert 0 <= checked["qa_score"] <= 1
    # Quality is measured in the experiment; do not force a pretrained model to pass a fabricated threshold.
