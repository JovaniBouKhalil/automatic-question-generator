"""Final app workflow checks use explicit model stubs; integration cases run real weights."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture(autouse=True)
def local_model_stubs(request, monkeypatch):
    if request.node.get_closest_marker("integration"):
        return
    from questionforge import verification
    from questionforge.learning import quality
    from questionforge.registry import GENERATORS
    from questionforge.text import evidence_for

    monkeypatch.setattr(quality, "local_quality_source", lambda: Path("models/test-stub"))

    def fake(context, language, count, **kwargs):
        answers = kwargs.get("manual_answers") or ["leaves"]
        rows = []
        for answer in answers[:count]:
            question = {"en": "Who planted the seeds?", "fr": "Qui a planté les graines ?",
                        "ar": "من زرع البذور؟"}[language]
            rows.append({"question": question, "answer": answer, "answer_start": context.index(answer),
                         "evidence": evidence_for(context, answer).text, "language": language,
                         "method": "learning_focused_hybrid", "model_id": quality.QUALITY_MODEL.model_id,
                         "draft_model_id": GENERATORS[language].model_id, "draft_question": question,
                         "learning_goal": "Understand the stated fact.", "lexical_relevance": .5, "answer_leak": False})
        return rows, {"returned": len(rows)}

    monkeypatch.setattr(quality, "create_quality_quiz", fake)
    monkeypatch.setattr(verification, "check_answer", lambda question, context, answer, language, verifier:
                        {"qa_answer": answer, "qa_score": .9, "answer_f1": 1., "answer_em": 1.})


def app_test():
    return AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()


def test_create_quiz_with_the_single_workflow_and_answer_checks():
    app = app_test()
    app.text_area[0].input("Lina planted nine seeds in a tray.")
    app.text_area[1].input("Lina")
    app.button[1].click().run()
    assert not app.exception and not app.error
    row = app.session_state["questions"][0]
    assert row["answer"] == "Lina" and row["method"] == "learning_focused_hybrid"
    assert row["verifier"] == "original" and row["answer_f1"] == 1.
    pick = next(box for box in app.checkbox if box.label == "Include in quiz exports")
    assert pick.value
    pick.uncheck().run()
    assert any("Select at least one question" in element.value for element in app.info)
    assert len(app.session_state["questions"]) == 1


def test_single_workflow_has_no_generation_or_adapter_switches():
    app = app_test()
    assert not app.exception
    assert [box.label for box in app.sidebar.selectbox] == ["Passage language / Langue / اللغة"]
    assert not list(app.sidebar.toggle) and not list(app.sidebar.select_slider)
    assert [slider.label for slider in app.sidebar.slider] == ["Questions to create"]


def test_missing_quality_model_blocks_generation_without_a_silent_fallback(monkeypatch):
    from questionforge.learning import quality

    def missing():
        raise RuntimeError("not installed")

    monkeypatch.setattr(quality, "local_quality_source", missing)
    app = app_test()
    assert not app.exception
    assert next(button for button in app.button if button.label == "Generate questions →").disabled
    assert any("not installed" in warning.value for warning in app.warning)
    assert not any(box.label == "Generation method" for box in app.selectbox)


def test_arabic_workflow_and_invalid_answer_feedback():
    app = app_test()
    app.sidebar.selectbox[0].select("ar").run()
    app.text_area[0].input("زرعت لينا تسع بذور في وعاء.")
    app.text_area[1].input("لينا")
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["questions"][0]["question"].endswith("؟")
    app.text_area[1].input("This answer is absent")
    app.button[1].click().run()
    assert app.error and app.session_state["questions"] == []


def test_multi_item_question_with_single_heading_answer_is_excluded(monkeypatch):
    from questionforge.learning import quality

    def fake(context, language, count, **kwargs):
        return [{"question": "Quels sont les 4 domaines du programme ?", "answer": "ENVIRONNEMENT",
                 "answer_start": context.index("ENVIRONNEMENT"), "evidence": "7–9 ENVIRONNEMENT",
                 "language": "fr", "method": "learning_focused_hybrid", "lexical_relevance": .5,
                 "answer_leak": False}], {"returned": 1}

    monkeypatch.setattr(quality, "create_quality_quiz", fake)
    app = app_test()
    app.sidebar.selectbox[0].select("fr").run()
    app.text_area[0].input("Les priorités organisent les actions du programme.\n7–9 ENVIRONNEMENT")
    app.text_area[1].input("ENVIRONNEMENT")
    app.button[1].click().run()
    assert not app.exception and not app.error
    assert not app.session_state["questions"]
    assert "4 items" in app.session_state["structural_rejections"][0]["reason"]


def test_outline_only_input_abstains_before_calling_the_generator(monkeypatch):
    from questionforge.learning import quality

    def unexpected(*args, **kwargs):
        raise AssertionError("Outline-only text must not invoke a question generator.")

    monkeypatch.setattr(quality, "create_quality_quiz", unexpected)
    app = app_test()
    app.text_area[0].input("CHAPTER TWO\n7–9 ENVIRONMENT\nSUMMARY")
    app.button[1].click().run()
    assert not app.exception and app.error
    assert "No explanatory facts" in app.error[0].value
    assert not app.session_state["questions"]


def test_final_workflow_passes_objective_and_fixed_inference_settings(monkeypatch):
    from questionforge.classroom.material import CLASSROOM_PLAN_SYSTEM, build_classroom_candidates
    from questionforge.learning import quality

    calls = []
    original = quality.create_quality_quiz

    def fake(context, language, count, **kwargs):
        calls.append(kwargs)
        return original(context, language, count, **kwargs)

    monkeypatch.setattr(quality, "create_quality_quiz", fake)
    app = app_test()
    app.text_area[0].input("A mesh catches leaves.")
    app.text_area[1].input("leaves")
    app.text_area[2].input("Understand how the filter works.")
    app.button[1].click().run()
    assert not app.exception and not app.error
    assert calls[0]["objective"] == "Understand how the filter works."
    assert calls[0]["role_focus"] is True and calls[0]["beams"] == 4 and calls[0]["checkpoint"] is None
    assert calls[0]["candidate_builder"] is build_classroom_candidates
    assert calls[0]["planning_system"] == CLASSROOM_PLAN_SYSTEM
    assert any(element.label == "Compare with the T5 draft" for element in app.expander)


def test_rerunning_the_final_workflow_keeps_the_users_passage_and_target():
    app = app_test()
    app.text_area[0].input("A mesh catches leaves.")
    app.text_area[1].input("leaves")
    app.sidebar.slider[0].set_value(2).run()
    assert not app.exception
    assert app.text_area[0].value == "A mesh catches leaves." and app.text_area[1].value == "leaves"


def test_demo_goal_updates_without_overwriting_a_users_goal():
    app = app_test()
    assert "training, validation and test" in app.text_area[2].value
    app.text_area[0].input("A mesh catches leaves.").run()
    assert app.text_area[2].value == ""
    app.text_area[2].input("Understand the filter's function.")
    app.text_area[0].input("A tank stores water.").run()
    assert app.text_area[2].value == "Understand the filter's function."


def test_qa_flagged_questions_start_excluded_and_can_be_included_after_review(monkeypatch):
    from questionforge import verification

    monkeypatch.setattr(verification, "check_answer", lambda *args, **kwargs:
                        {"qa_answer": "unrelated", "qa_score": .01, "answer_f1": 0., "answer_em": 0.})
    app = app_test()
    app.text_area[0].input("Lina planted nine seeds in a tray.")
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


def test_teacher_revision_preserves_generation_and_removes_stale_qa():
    import copy

    from app import revise_question

    original = {"question": "Who invented the panel?", "answer": "Lina", "language": "en",
                "evidence": "Lina connected the panel.", "method": "learning_focused_hybrid",
                "model_id": "Qwen/editor", "model_revision": "pinned", "draft_model_id": "T5/base",
                "edit_recipe": "answer-role-v1", "qa_answer": "Lina", "qa_score": .9,
                "answer_f1": 1., "answer_em": 1., "accepted": True, "review_reasons": [], "verifier": "original"}
    saved = copy.deepcopy(original)
    revised = revise_question(original, "Who connected the panel?")
    assert original == saved and revised["answer"] == original["answer"]
    assert revised["original_question"] == original["question"]
    assert revised["model_id"] == "teacher_revision" and revised["generation_model_id"] == "Qwen/editor"
    assert revised["generation_model_revision"] == "pinned" and revised["draft_model_id"] == "T5/base"
    assert revised["generation_edit_recipe"] == "answer-role-v1" and revised["teacher_edited"]
    assert not any(key in revised for key in ["qa_answer", "qa_score", "answer_f1", "answer_em", "accepted", "verifier"])
    assert revise_question(original, original["question"]) == original


def test_teacher_csv_keeps_revision_provenance_and_literal_formulas():
    import csv
    import io

    from app import classroom_csv, revise_question

    row = {"question": "=unsafe()", "answer": "Lina", "language": "en", "evidence": "Lina connected the panel.",
           "method": "learning_focused_hybrid", "model_id": "Qwen/editor", "learning_goal": "Understand the action."}
    data = classroom_csv([revise_question(row, "Who connected the panel?")])
    exported = next(csv.DictReader(io.StringIO(data.lstrip("\ufeff"), newline="")))
    assert data.startswith("\ufeff") and exported["original_question"] == "'=unsafe()"
    assert exported["question"] == "Who connected the panel?" and exported["teacher_edited"] == "True"
    assert exported["model_id"] == "teacher_revision" and exported["generation_model_id"] == "Qwen/editor"


def test_editing_wording_resets_export_selection_and_keeps_the_generated_record():
    import copy

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.text_area[0].input("Lina planted nine seeds in a tray.")
    app.text_area[1].input("Lina")
    app.button[1].click().run()
    original = copy.deepcopy(app.session_state["questions"])
    next(area for area in app.text_area if area.label == "Revised question").input("Who planted the seeds in the tray?").run()
    assert not app.exception and app.session_state["questions"] == original
    pick = next(box for box in app.checkbox if box.label == "Include in quiz exports")
    assert not pick.value
    pick.check().run()
    assert any("Selected 1 of 1" in text.value for text in app.markdown)
    next(area for area in app.text_area if area.label == "Revised question").input("Which student planted the seeds?").run()
    assert not next(box for box in app.checkbox if box.label == "Include in quiz exports").value
    next(area for area in app.text_area if area.label == "Revised question").input("").run()
    assert not app.exception and app.error
    assert next(box for box in app.checkbox if box.label == "Include in quiz exports").disabled
    assert app.session_state["questions"] == original


@pytest.mark.integration
def test_saved_english_research_adapter():
    import json

    root = Path(__file__).resolve().parents[1]
    expanded = root / "results/expanded_training/summary.json"
    if expanded.exists():
        study = json.loads(expanded.read_text(encoding="utf-8"))
        selected_path = study["languages"]["en"]["selected_app_adapter"]
    else:
        study = json.loads((root / "results/seed_study/summary.json").read_text(encoding="utf-8"))
        selected_path = study["optional_app_adapter"]
    from questionforge.models import generate_question

    row = generate_question("Lina planted nine seeds in a tray.", "Lina", "en", beams=4,
                            checkpoint=str(root / selected_path))
    assert row.question.strip() and Path(row.model_id) == root / selected_path



@pytest.mark.integration
@pytest.mark.parametrize("language,answer", [("fr", "Lina"), ("ar", "لينا")])
def test_saved_native_research_adapter(language, answer):
    import json

    root = Path(__file__).resolve().parents[1]
    path = root / "results/expanded_training/summary.json"
    if not path.exists():
        pytest.skip("Expanded training has not completed.")
    study = json.loads(path.read_text(encoding="utf-8"))
    selected = study["languages"][language]["selected_app_adapter"]
    if selected is None:
        pytest.skip("Development selection preferred the unadapted generator.")
    from questionforge.models import generate_question

    text = {"fr": "Lina a planté neuf graines dans un bac.", "ar": "زرعت لينا تسع بذور في وعاء."}[language]
    row = generate_question(text, answer, language, beams=4, checkpoint=str(root / selected))
    assert row.question.strip() and row.language == language and Path(row.model_id) == root / selected



@pytest.mark.integration
def test_saved_science_research_adapter():
    import json

    root = Path(__file__).resolve().parents[1]
    selection_path = root / "results/science_training/selection.json"
    if not selection_path.exists():
        pytest.skip("Science training is still running.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection["selected_checkpoint"]:
        pytest.skip("Validation selected the unadapted model.")
    from questionforge.models import generate_question

    text = "Chlorophyll captures the light energy that plants use to make food."
    row = generate_question(text, "Chlorophyll", "en", beams=4,
                            checkpoint=str(root / selection["selected_checkpoint"]))
    assert row.question.strip() and row.answer == "Chlorophyll"
    assert Path(row.model_id) == root / selection["selected_checkpoint"]



@pytest.mark.integration
@pytest.mark.parametrize("language,text,answer", [
    ("en", "A solar panel converts light into electrical energy.", "electrical energy"),
    ("fr", "Un panneau solaire convertit la lumière en énergie électrique.", "énergie électrique"),
    ("ar", "تحول اللوحة الشمسية الضوء إلى طاقة كهربائية.", "طاقة كهربائية"),
])
def test_final_workflow_through_the_application(language, text, answer):
    from questionforge.learning.quality import QUALITY_MODEL, local_quality_source

    try:
        local_quality_source()
    except RuntimeError:
        pytest.skip("The optional stronger editor is not downloaded.")
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(root / "app.py", default_timeout=600).run()
    app.run()
    app.sidebar.selectbox[0].select(language).run()
    app.text_area[0].input(text)
    app.text_area[1].input(answer)
    app.text_area[2].input("Understand the panel's function.")
    app.button[1].click().run(timeout=600)
    assert not app.exception and not app.error
    row = app.session_state["questions"][0]
    assert row["model_id"] == QUALITY_MODEL.model_id
    from questionforge.registry import GENERATORS

    assert row["draft_model_id"] == GENERATORS[language].model_id and row["draft_question"].strip()
    assert row["answer"] == answer and row["question"].strip()
    assert row["verifier"] == "original" and 0 <= row["answer_f1"] <= 1
    assert row["app_profile"] == "classroom-hybrid-4b-v1"


@pytest.mark.integration
def test_french_slide_outline_through_the_automatic_hybrid_application():
    from questionforge.classroom.material import VERSION, prepare_material
    from questionforge.learning.quality import QUALITY_MODEL, local_quality_source

    try:
        local_quality_source()
    except RuntimeError:
        pytest.skip("The stronger editor is not downloaded.")
    source = ("Projet Horizon\nLes quatre priorités\n1–2 QUALITÉ\n7–9 ENVIRONNEMENT\n\n"
              "Une politique interne guide les actions pour encourager des décisions\nresponsables.\n"
              "Le programme couvre quatre priorités : sécurité, qualité, transparence et environnement.\n"
              "Les contrôles réguliers permettent de repérer les erreurs avant la livraison.")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=600).run()
    app.sidebar.selectbox[0].select("fr")
    app.run()
    app.text_area[0].input(source)
    app.text_area[1].input("")
    app.text_area[2].input("")
    app.button[1].click().run(timeout=600)
    assert not app.exception and not app.error
    rows = app.session_state["questions"]
    assert rows and app.session_state["original_quiz_context"] == source
    assert app.session_state["quiz_context"] == prepare_material(source)
    assert app.session_state["learning_report"]["candidate_source"] == "build_classroom_candidates"
    for row in rows:
        assert row["model_id"] == QUALITY_MODEL.model_id and row["draft_model_id"] == "lmqg/mt5-small-frquad-qg"
        assert row["source_preparation"] == VERSION and row["answer"] not in {"QUALITÉ", "ENVIRONNEMENT"}
        start = row["answer_start"]
        assert app.session_state["quiz_context"][start:start + len(row["answer"])] == row["answer"]
