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
    pick = next(box for box in app.checkbox if box.label == "Include in quiz exports")
    assert pick.value
    pick.uncheck().run()
    assert any("Select at least one question" in element.value for element in app.info)
    assert len(app.session_state["questions"]) == 1


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
    assert app.session_state["questions"] == []


@pytest.mark.parametrize("stronger", [False, True])
def test_learning_mode_passes_objective_and_shows_the_seq2seq_draft(monkeypatch, stronger):
    from questionforge.learning import pipeline, quality

    calls = []

    def fake(context, language, count, **kwargs):
        calls.append((context, language, count, kwargs))
        return [{"question": "What does the mesh catch?", "answer": "leaves", "language": "en",
                 "evidence": "A mesh catches leaves.", "method": "learning_focused_hybrid",
                 "learning_goal": "Understand the mesh's function.",
                 "draft_question": "What catches leaves?"}], {"returned": 1}

    monkeypatch.setattr(quality if stronger else pipeline, "create_quality_quiz" if stronger else "create_learning_quiz", fake)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.sidebar.selectbox[1].select("Learning-focused hybrid 4B (experimental)" if stronger else "Learning-focused hybrid (experimental)")
    app.sidebar.toggle[0].set_value(False)
    app.run()
    app.text_area[0].input("A mesh catches leaves.")
    app.text_area[2].input("Understand how the filter works.")
    app.button[1].click().run(timeout=20)
    assert not app.exception and not app.error
    assert calls[0][3]["objective"] == "Understand how the filter works."
    assert app.session_state["questions"][0]["method"] == "learning_focused_hybrid"
    assert any("Understand the mesh" in element.value for element in app.markdown)


def test_switching_learning_mode_keeps_the_users_passage_and_target():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.sidebar.selectbox[1].select("Pretrained Transformer")
    app.run()
    app.text_area[0].input("A mesh catches leaves.")
    app.text_area[1].input("leaves")
    app.run()
    app.sidebar.selectbox[1].select("Learning-focused hybrid (experimental)")
    app.run()
    assert not app.exception
    assert app.text_area[0].value == "A mesh catches leaves."
    assert app.text_area[1].value == "leaves"


def test_demo_goal_updates_without_overwriting_a_users_goal():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.sidebar.selectbox[1].select("Learning-focused hybrid (experimental)")
    app.run()
    assert "training, validation and test" in app.text_area[2].value
    app.text_area[0].input("A mesh catches leaves.")
    app.run()
    assert app.text_area[2].value == ""
    app.text_area[2].input("Understand the filter's function.")
    app.text_area[0].input("A tank stores water.")
    app.run()
    assert app.text_area[2].value == "Understand the filter's function."


def test_qa_flagged_questions_start_excluded_and_can_be_included_after_review(monkeypatch):
    from questionforge import verification

    monkeypatch.setattr(verification, "check_answer", lambda *args, **kwargs:
                        {"qa_answer": "unrelated", "qa_score": .01, "answer_f1": 0., "answer_em": 0.})
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.sidebar.selectbox[1].select("Template baseline")
    app.run()
    app.text_area[1].input("Lina")
    app.button[1].click().run()
    assert not app.exception and not app.error
    pick = next(box for box in app.checkbox if box.label == "Include in quiz exports")
    assert not pick.value and not app.session_state["questions"][0]["accepted"]
    pick.check().run()
    assert any("Selected 1 of 1" in element.value for element in app.markdown)
    assert not app.session_state["questions"][0]["accepted"]
    app.button[1].click().run()
    assert not next(box for box in app.checkbox if box.label == "Include in quiz exports").value


def test_printable_quiz_includes_the_passage_and_escapes_user_text():
    from app import classroom_worksheet

    rows = [{"question": "What does the mesh catch?", "answer": "leaves", "evidence": "The mesh catches leaves.",
             "learning_objective": "Understand <the filter>."}]
    passage = "<script>bad()</script>\nالمرشح يحجز الأوراق."
    student = classroom_worksheet(rows, passage)
    key = classroom_worksheet(rows, passage, True)
    assert "&lt;script&gt;bad()&lt;/script&gt;" in student and "<script>" not in student
    assert "المرشح يحجز الأوراق." in student and "Understand &lt;the filter&gt;." in student
    assert "<strong>Answer /" not in student and "<strong>Answer /" in key


@pytest.mark.integration
def test_real_english_model_through_the_application():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    app.sidebar.selectbox[1].select("Pretrained Transformer")
    app.sidebar.toggle[0].set_value(False)
    if len(app.sidebar.toggle) > 1:
        app.sidebar.toggle[1].set_value(False)
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
    expanded = root / "results/expanded_training/summary.json"
    if expanded.exists():
        study = json.loads(expanded.read_text(encoding="utf-8"))
        selected_path = study["languages"]["en"]["selected_app_adapter"]
    else:
        study = json.loads((root / "results/seed_study/summary.json").read_text(encoding="utf-8"))
        selected_path = study["optional_app_adapter"]
    app = AppTest.from_file(root / "app.py", default_timeout=120).run()
    app.sidebar.selectbox[1].select("Pretrained Transformer")
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
    assert Path(row["model_id"]) == root / selected_path


@pytest.mark.integration
@pytest.mark.parametrize("language,answer", [("fr", "Lina"), ("ar", "لينا")])
def test_expanded_native_adapter_through_the_application(language, answer):
    import json

    root = Path(__file__).resolve().parents[1]
    path = root / "results/expanded_training/summary.json"
    if not path.exists():
        pytest.skip("Expanded training has not completed.")
    study = json.loads(path.read_text(encoding="utf-8"))
    selected = study["languages"][language]["selected_app_adapter"]
    if selected is None:
        pytest.skip("Development selection preferred the unadapted generator.")
    app = AppTest.from_file(root / "app.py", default_timeout=120).run()
    app.sidebar.selectbox[1].select("Pretrained Transformer")
    app.sidebar.selectbox[0].select(language)
    app.sidebar.toggle[0].set_value(False)
    app.run()
    app.sidebar.toggle[1].set_value(True)
    app.run()
    app.text_area[1].input(answer)
    app.button[1].click().run(timeout=120)
    assert not app.exception
    assert not app.error
    row = app.session_state["questions"][0]
    assert row["question"].strip()
    assert row["language"] == language
    assert Path(row["model_id"]) == root / selected


@pytest.mark.integration
def test_science_adapter_through_the_application():
    import json

    root = Path(__file__).resolve().parents[1]
    selection_path = root / "results/science_training/selection.json"
    if not selection_path.exists():
        pytest.skip("Science training is still running.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection["selected_checkpoint"]:
        pytest.skip("Validation selected the unadapted model.")
    app = AppTest.from_file(root / "app.py", default_timeout=120).run()
    app.sidebar.selectbox[1].select("Pretrained Transformer")
    app.sidebar.toggle[0].set_value(False)
    app.run()
    app.sidebar.toggle[1].set_value(True)
    app.run()
    next(box for box in app.sidebar.selectbox if box.label == "English adaptation data").select("Educational science")
    app.run()
    app.text_area[0].input("Chlorophyll captures the light energy that plants use to make food.")
    app.text_area[1].input("Chlorophyll")
    app.button[1].click().run(timeout=120)
    assert not app.exception and not app.error
    row = app.session_state["questions"][0]
    assert Path(row["model_id"]) == root / selection["selected_checkpoint"]
    assert row["answer"] == "Chlorophyll" and row["question"].strip()


@pytest.mark.integration
def test_stronger_editor_through_the_application():
    from questionforge.learning.quality import QUALITY_MODEL, local_quality_source

    try:
        local_quality_source()
    except RuntimeError:
        pytest.skip("The optional stronger editor is not downloaded.")
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / "app.py", default_timeout=600).run()
    app.sidebar.selectbox[1].select("Learning-focused hybrid 4B (experimental)")
    app.sidebar.toggle[0].set_value(False)
    app.run()
    app.text_area[0].input("A solar panel converts light into electrical energy. Shading the panel reduces its voltage.")
    app.text_area[1].input("electrical energy")
    app.text_area[2].input("Understand the panel's function.")
    app.button[1].click().run(timeout=600)
    assert not app.exception and not app.error
    row = app.session_state["questions"][0]
    assert row["model_id"] == QUALITY_MODEL.model_id
    assert row["draft_model_id"] == "lmqg/t5-small-squad-qg" and row["draft_question"].strip()
    assert row["answer"] == "electrical energy" and row["question"].strip()
