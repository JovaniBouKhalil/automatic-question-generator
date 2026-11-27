"""Streamlit workflow smoke tests use the honest offline baseline; no model downloads."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_create_quiz_with_baseline_and_no_verifier():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
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
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
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


@pytest.mark.integration
def test_real_english_model_through_the_application():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    app.sidebar.toggle[0].set_value(False)
    app.run()
    app.text_area[1].input("Lina")
    app.button[1].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    row = app.session_state["questions"][0]
    assert row["question"].strip()
    assert row["model_id"] == "lmqg/t5-small-squad-qg"
    assert row["input_tokens"] <= 384


@pytest.mark.integration
def test_development_selected_adapter_through_the_application():
    import json

    root = Path(__file__).resolve().parents[1]
    study = json.loads((root / "results/seed_study/summary.json").read_text(encoding="utf-8"))
    app = AppTest.from_file(root / "app.py", default_timeout=120).run()
    app.sidebar.toggle[0].set_value(False)
    app.run()
    assert app.sidebar.toggle[1].label == "Use adapted English generator"
    app.sidebar.toggle[1].set_value(True)
    app.run()
    app.text_area[1].input("Lina")
    app.button[1].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    row = app.session_state["questions"][0]
    assert row["question"].strip()
    assert Path(row["model_id"]) == root / study["optional_app_adapter"]
