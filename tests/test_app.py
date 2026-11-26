"""Streamlit workflow smoke tests use the honest offline baseline; no model downloads."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_create_quiz_with_baseline_and_no_verifier():
    app = AppTest.from_file("app.py").run()
    assert not app.exception
    app.sidebar.selectbox[1].select("Template baseline")
    app.sidebar.toggle[0].set_value(False)
    app.run()
    app.text_area[1].input("Lina")
    app.button[1].click().run(timeout=20)
    assert not app.exception
    assert app.session_state["questions"][0]["answer"] == "Lina"
    assert app.session_state["questions"][0]["method"] == "baseline"
    assert "qa_score" not in app.session_state["questions"][0]


def test_arabic_workflow_and_invalid_answer_feedback():
    app = AppTest.from_file("app.py").run()
    app.sidebar.selectbox[0].select("ar")
    app.sidebar.selectbox[1].select("Template baseline")
    app.sidebar.toggle[0].set_value(False)
    app.run()
    app.text_area[1].input("لينا")
    app.button[1].click().run(timeout=20)
    assert not app.exception
    assert app.session_state["questions"][0]["question"].endswith("؟")
    app.text_area[1].input("This answer is absent")
    app.button[1].click().run(timeout=20)
    assert app.error
