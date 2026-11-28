"""Local Streamlit demonstration. Run: python -m streamlit run app.py"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from questionforge.baseline import template_question
from questionforge.candidates import extract_candidates
from questionforge.classroom.material import (
    CLASSROOM_PLAN_SYSTEM,
    build_classroom_candidates,
    list_scope_issue,
    prepare_material,
    select_classroom_targets,
)
from questionforge.classroom.material import (
    VERSION as MATERIAL_VERSION,
)
from questionforge.export import as_csv, as_json, moodle_xml, worksheet_html
from questionforge.learning.exports import learning_csv
from questionforge.learning.pipeline import PLANNER, create_learning_quiz
from questionforge.learning.quality import QUALITY_MODEL, create_quality_quiz, local_quality_source
from questionforge.learning.rules import create_learning_t5_quiz
from questionforge.metrics import leaks_answer, lexical_relevance, quality_decision
from questionforge.models import generate_question
from questionforge.registry import GENERATORS
from questionforge.review import GUIDE, ratings_csv, summarize_ratings, validate_ratings
from questionforge.text import LANGUAGES, evidence_for
from questionforge.verification import check_answer

ROOT = Path(__file__).resolve().parent


def sample_passages(language, learning=False):
    if learning:
        source = json.loads((ROOT / "data/authoring/learning_passages.json").read_text(encoding="utf-8"))
        return {scenario["topic"].replace("_", " ").title(): scenario[language]["context"] for scenario in source}
    source = json.loads((ROOT / "data/authoring/scenarios.json").read_text(encoding="utf-8"))
    return {scenario["topic"].replace("_", " ").title(): scenario[language][0]
            for scenario in source if scenario["split"] == "dev"}


def read_upload(upload):
    if upload.size > 10 * 1024 * 1024:
        raise ValueError("Please choose a file smaller than 10 MB.")
    if upload.name.casefold().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(upload.getvalue()))
        if len(reader.pages) > 40:
            raise ValueError("Please select a PDF with at most 40 pages.")
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("This PDF contains no extractable text. Use a text PDF or paste OCR text.")
    else:
        text = upload.getvalue().decode("utf-8-sig")
    if len(text) > 20000:
        raise ValueError("Please select a passage of at most 20,000 characters.")
    return text


def marked_evidence(row):
    evidence, answer = row["evidence"], row["answer"]
    offset = evidence.find(answer)
    if offset < 0:
        return html.escape(evidence)
    return html.escape(evidence[:offset]) + "<mark>" + html.escape(answer) + "</mark>" + html.escape(evidence[offset + len(answer):])


def classroom_worksheet(rows, context, include_answers=False):
    page = worksheet_html(rows, include_answers)
    objective = next((row.get("learning_objective") for row in rows if row.get("learning_objective")), "")
    goal = f"<p><strong>Learning objective:</strong> {html.escape(objective)}</p>" if objective else ""
    passage = ("<section dir='auto' style='white-space:pre-wrap;line-height:1.8;border:1px solid #ccd5d9;padding:18px'>"
               "<strong>Reading passage / Texte / النص</strong><br>" + html.escape(context) + "</section>") if context else ""
    return page.replace("<ol>", goal + passage + "<h2>Questions</h2><ol>", 1)


def revise_question(row, wording):
    """Preserve the generated record; an edited question cannot inherit its QA result."""
    wording = wording.strip()
    revised = dict(row)
    if wording == row["question"].strip():
        return revised
    revised.update(question=wording, original_question=row["question"], teacher_edited=True,
                   generation_method=row["method"], method="teacher_revision",
                   generation_model_id=row.get("model_id", ""), model_id="teacher_revision",
                   generation_model_revision=row.get("model_revision", ""),
                   generation_edit_recipe=row.get("edit_recipe", ""), edit_recipe="teacher-wording-revision",
                   educational_quality_status="teacher_edited_unrated")
    for key in ["qa_answer", "qa_score", "answer_f1", "answer_em", "accepted", "review_reasons",
                "verifier", "model_revision"]:
        revised.pop(key, None)
    revised["lexical_relevance"] = lexical_relevance(wording, row["evidence"], row["language"])
    revised["answer_leak"] = leaks_answer(wording, row["answer"], row["language"])
    return revised


def classroom_csv(rows):
    base = learning_csv(rows) if any(row.get("learning_goal") for row in rows) else as_csv(rows)
    reader = csv.DictReader(io.StringIO(base.lstrip("\ufeff"), newline=""))
    extra = ["model_id", "teacher_edited", "original_question", "generation_method", "generation_model_id",
             "generation_model_revision", "generation_edit_recipe"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, [*reader.fieldnames, *[key for key in extra if key not in reader.fieldnames]])
    writer.writeheader()
    for exported, original in zip(reader, rows):
        for key in extra:
            value = original.get(key, False if key == "teacher_edited" else "")
            exported[key] = "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
        writer.writerow(exported)
    return "\ufeff" + stream.getvalue()


def main():
    st.set_page_config(page_title="QuestionForge · From passage to practice", page_icon="✦", layout="wide")
    st.markdown("""<style>
      .stApp {background:#f6f7f3;color:#173c3b}
      .block-container {max-width:1180px;padding-top:2.4rem}
      h1 {letter-spacing:-.045em!important;font-size:3.7rem!important;line-height:1.06!important}
      h2,h3 {letter-spacing:-.025em!important}
      [data-testid='stSidebar'] {background:#e9eee6}
      textarea {line-height:1.8!important}
      .eyebrow {font-size:12px;letter-spacing:.18em;font-weight:700;color:#69816a;margin-bottom:14px}
      .intro {font-size:18px;color:#60756c;max-width:660px;line-height:1.65}
      .question {font-size:23px;line-height:1.65;color:#173c3b;font-weight:600;padding:8px 0}
      .evidence {line-height:1.9;padding:16px 20px;background:#edf1e8;border-radius:8px;color:#425a50}
      mark {background:#d6e8aa;color:#173c3b;padding:2px 4px;border-radius:3px}
      .stButton>button[kind='primary'] {background:#205c4a;border-color:#205c4a;border-radius:8px}
      [data-testid='stMetricValue'] {color:#205c4a}
      .stAppDeployButton {display:none}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ✦ QuestionForge")
        st.caption("NLP FINAL PROJECT · 05")
        initial_language = st.query_params.get("lang", "en")
        initial_language = initial_language if initial_language in LANGUAGES else "en"
        language = st.selectbox("Passage language / Langue / اللغة", list(LANGUAGES),
                                index=list(LANGUAGES).index(initial_language), format_func=LANGUAGES.get)
        try:
            local_quality_source()
            default_method = 4
        except RuntimeError:
            default_method = 0
        method = st.selectbox("Generation method", ["Pretrained Transformer", "Template baseline",
                                                    "Learning-focused T5 (experimental)",
                                                    "Learning-focused hybrid (experimental)",
                                                    "Learning-focused hybrid 4B (experimental)"], index=default_method)
        learning_mode = method.startswith("Learning-focused")
        quality_mode = "4B" in method
        role_guidance = True
        if method == "Pretrained Transformer":
            st.caption("T5-only comparison mode. Qwen is not used here. For lesson concepts, try the 4B hybrid when its weights are installed.")
        if learning_mode:
            if "hybrid" in method:
                st.caption("Selects important facts, creates T5 drafts, then edits questions for understanding. The local editor takes longer.")
                if quality_mode:
                    st.caption("Uses the stronger 4B planner/editor. CPU generation can take several minutes; inspect wording and source support.")
            else:
                st.caption("Targets explicitly stated purposes, functions and properties. Uses the existing T5 model and can return fewer questions.")
        beams = st.select_slider("Decoding beams", options=[1, 4], value=4,
                                  help="1 = greedy decoding. 4 = deterministic beam search.")
        verify = st.toggle("Check answerability", value=True,
                           help="An independent QA model attempts to recover the answer. This is an imperfect proxy.")
        verifier = "original"
        if verify:
            verifier = st.selectbox("Answerability checker", ["original", "multilingual"],
                                    format_func=lambda value: {"original": "Original QA checker",
                                                              "multilingual": "Experimental multilingual QA"}[value],
                                    help="The experimental checker has multilingual QA training. Both can miss unsupported premises.")
        checkpoint = None
        study_path = ROOT / "results/seed_study/summary.json"
        expanded_path = ROOT / "results/expanded_training/summary.json"
        selected = None
        use_by_default = False
        if expanded_path.exists() and (method == "Pretrained Transformer" or learning_mode):
            expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
            selection = expanded["languages"][language]["selected_app_adapter"]
            if selection:
                selected = ROOT / selection
        elif language == "en" and (method == "Pretrained Transformer" or learning_mode) and study_path.exists():
            study = json.loads(study_path.read_text(encoding="utf-8"))
            selected = ROOT / study["optional_app_adapter"]
        if selected is not None and selected.is_dir():
            label = {"en": "Use adapted English generator", "fr": "Use adapted French generator", "ar": "Use adapted Arabic generator"}[language]
            if st.toggle(label, value=use_by_default, key=f"use_adapter_{language}",
                         help="Checkpoint selected using development loss. Disable to compare the original pretrained generator."):
                checkpoint = str(selected)
                science_path = ROOT / "results/science_training/selection.json"
                if language == "en" and science_path.exists():
                    science = json.loads(science_path.read_text(encoding="utf-8"))
                    if science.get("complete") and science.get("selected_checkpoint"):
                        adaptation = st.selectbox("English adaptation data", ["General reading", "Educational science"],
                                                  help="The science adapter was selected on SciQ validation loss. Compare its questions with the original model.")
                        if adaptation == "Educational science":
                            checkpoint = str(ROOT / science["selected_checkpoint"])
        if quality_mode:
            role_guidance = st.toggle("Guide the target answer's role", value=True,
                                      help="Adds cues for functions, purposes, limitations and comparisons. Disable to compare the original 4B editor; both styles need source review.")
        count = st.slider("Questions to create", 1, 8, 3)
        st.divider()
        st.markdown("**Works locally. No paid API.**")
        st.caption("The first use downloads model weights. Once cached, generation works offline. CPU inference is supported.")
        with st.expander("About the models"):
            st.write(GENERATORS[language].model_id)
            if learning_mode and "hybrid" in method:
                st.write("Local planner/editor: " + (QUALITY_MODEL.model_id if quality_mode else PLANNER.model_id))
            st.caption("Questions are drafts for teacher review. The original checker has English QA training; the experimental checker adds multilingual QA training. French remains zero-shot for both.")
    direction = "rtl" if language == "ar" else "ltr"
    alignment = "right" if language == "ar" else "left"
    font_size = "18px" if language == "ar" else "15px"
    st.markdown(f"<style>textarea{{direction:{direction};text-align:{alignment};font-family:Arial,sans-serif!important;font-size:{font_size}!important}}</style>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>READ. QUESTION. UNDERSTAND.</div>", unsafe_allow_html=True)
    st.title("Turn a passage into a learning moment.")
    st.markdown("<p class='intro'>Create factual questions in English, French, and Arabic. Keep the source evidence close—and see which questions need a second look.</p>", unsafe_allow_html=True)
    compose, results_tab, research, review_tab = st.tabs(["Create a quiz", "Question library", "Experiment results", "Review study"])
    with compose:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("01 / Your learning material")
            samples = sample_passages(language, learning_mode)
            sample = st.selectbox("Try a classroom passage", list(samples))
            passage_key = f"passage_{language}"
            family_key = f"sample_family_{language}"
            if passage_key not in st.session_state:
                st.session_state[passage_key] = samples[sample]
            elif st.session_state.get(family_key, learning_mode) != learning_mode:
                previous_samples = sample_passages(language, not learning_mode)
                # Replace only an untouched demo. Keep pasted/uploaded material and chosen answers.
                uploaded = st.session_state.get("upload_digest", "").endswith(language)
                if (st.session_state[passage_key] in previous_samples.values() and not uploaded
                        and not st.session_state.get(f"answers_{language}", "").strip()):
                    st.session_state[passage_key] = samples[sample]
            st.session_state[family_key] = learning_mode
            if st.button("Use this example"):
                st.session_state[passage_key] = samples[sample]
            upload = st.file_uploader("Or upload a text document", type=["txt", "pdf"],
                                     help="UTF-8 text or a PDF with selectable text. Maximum 10 MB / 40 pages.")
            if upload is not None:
                digest = hashlib.sha256(upload.getvalue()).hexdigest() + language
                if st.session_state.get("upload_digest") != digest:
                    try:
                        st.session_state[passage_key] = read_upload(upload)
                        st.session_state["upload_digest"] = digest
                    except Exception as exc:
                        st.error(str(exc))
            passage = st.text_area("Passage", key=passage_key, height=295, max_chars=20000)
            st.caption(f"{len(passage):,} characters · Questions stay in the selected language.")
            st.caption("Use explanatory sentences beneath the headings. Headings alone may produce no questions.")
        with right:
            st.subheader("02 / Choose your facts")
            st.write("Add short, exact answer phrases from the passage, one per line. Leave this blank to suggest key facts automatically.")
            manual = st.text_area("Target answers (optional)", key=f"answers_{language}", height=130,
                                   placeholder=({"en": "update a model's weights\nchoose hyperparameters",
                                                 "fr": "mettre à jour les poids d'un modèle\nchoisir les hyperparamètres",
                                                 "ar": "تحديث أوزان النموذج\nاختيار المعلمات الفائقة"} if learning_mode else
                                                {"en": "Lina\nNine", "fr": "Lina\nNeuf", "ar": "لينا\nتسع"})[language])
            objective = ""
            if learning_mode:
                learning_examples = json.loads((ROOT / "data/authoring/learning_passages.json").read_text(encoding="utf-8"))
                auto_goal = next((item[language]["objective"] for item in learning_examples
                                  if item[language]["context"] == passage), "")
                objective_key = f"learning_objective_{language}"
                last_auto_key = f"last_auto_goal_{language}"
                last_passage_key = f"objective_passage_{language}"
                if (objective_key not in st.session_state or
                        (st.session_state.get(last_passage_key) != passage and
                         st.session_state[objective_key] == st.session_state.get(last_auto_key, ""))):
                    st.session_state[objective_key] = auto_goal
                st.session_state[last_auto_key] = auto_goal
                st.session_state[last_passage_key] = passage
                objective = st.text_area("What should students understand? (optional)", max_chars=500,
                                         key=objective_key,
                                         help="For example: how the components work, the steps of a process, or a cause stated in the passage.")
                if "hybrid" not in method:
                    st.caption("In T5 mode this goal is saved for your review; automatic selection follows explicit relation markers. Use manual targets to control the facts.")
                st.caption("Focuses on concepts, functions and relationships stated in your passage. It may return fewer questions when there are too few useful facts.")
            else:
                st.caption("Automatic selection uses TF-IDF and simple span heuristics. Specific, short answers usually give clearer questions.")
            if st.button("Generate questions →", type="primary", use_container_width=True):
                # A failed new request must not leave an earlier quiz looking current.
                st.session_state["questions"] = []
                st.session_state["structural_rejections"] = []
                st.session_state["quiz_revision"] = st.session_state.get("quiz_revision", 0) + 1
                st.session_state.pop("learning_report", None)
                try:
                    if not passage.strip():
                        raise ValueError("Add a passage first.")
                    answers = list(dict.fromkeys(line.strip() for line in manual.splitlines() if line.strip()))
                    if len(answers) > 8:
                        raise ValueError("Use at most eight manual answers per quiz.")
                    original_passage = passage
                    prepared = prepare_material(passage)
                    # Keep the raw passage for review; all offsets refer to the prepared source.
                    for answer in answers:
                        evidence_for(prepared, answer)
                    passage = prepared
                    if learning_mode:
                        automatic = []
                        if not answers and not build_classroom_candidates(passage, language):
                            raise ValueError("No explanatory facts were found. Add complete statements beneath the headings.")
                    elif method == "Template baseline":
                        automatic = [(c.answer, c.start) for c in extract_candidates(passage, language, count)]
                    else:
                        automatic = [(c["answer"], c["answer_start"]) for c in select_classroom_targets(passage, language, count)]
                    candidates = [] if learning_mode else ([(answer, passage.find(answer)) for answer in answers[:count]] if answers else automatic)
                    if not candidates and not learning_mode:
                        raise ValueError("No explanatory facts were found. Add complete statements beneath the headings, or select exact answers from an explanatory paragraph.")
                    rows = []
                    started = time.perf_counter()
                    with st.status("Building your question set…", expanded=True) as status:
                        if learning_mode:
                            create = (create_quality_quiz if quality_mode else create_learning_quiz) if "hybrid" in method else create_learning_t5_quiz
                            rows, learning_report = create(passage, language, count,
                                        objective=objective, manual_answers=answers, beams=beams,
                                        checkpoint=checkpoint, progress=st.write,
                                        **({"candidate_builder": build_classroom_candidates,
                                            "planning_system": CLASSROOM_PLAN_SYSTEM} if "hybrid" in method else {}),
                                        **({"role_focus": role_guidance} if quality_mode else {}))
                            st.session_state["learning_report"] = learning_report
                            for row in rows:
                                row["learning_objective"] = objective
                                if verify:
                                    row.update(check_answer(row["question"], passage, row["answer"], language, verifier))
                                    row["verifier"] = verifier
                                    row["accepted"], row["review_reasons"] = quality_decision(row)
                        for index, (answer, start) in enumerate(candidates, 1):
                            st.write(f"Question {index}/{len(candidates)}")
                            if method == "Template baseline":
                                row = {"question": template_question(passage, answer, language, start),
                                       "answer": answer, "answer_start": start, "language": language,
                                       "evidence": evidence_for(passage, answer, start).text, "method": "baseline"}
                            else:
                                row = asdict(generate_question(passage, answer, language, answer_start=start,
                                                               beams=beams, checkpoint=checkpoint))
                            row["lexical_relevance"] = lexical_relevance(row["question"], row["evidence"], language)
                            row["answer_leak"] = leaks_answer(row["question"], answer, language)
                            if verify:
                                row.update(check_answer(row["question"], passage, answer, language, verifier))
                                row["verifier"] = verifier
                                row["accepted"], row["review_reasons"] = quality_decision(row)
                            rows.append(row)
                        visible, rejected = [], []
                        for row in rows:
                            issue = list_scope_issue(row["question"], row["answer"], language)
                            if issue:
                                rejected.append({"question": row["question"], "answer": row["answer"], "reason": issue})
                            else:
                                visible.append({**row, "source_preparation": MATERIAL_VERSION})
                        rows = visible
                        st.session_state["structural_rejections"] = rejected
                        status.update(label="Your question set is ready", state="complete", expanded=False)
                    st.session_state["questions"] = rows
                    st.session_state["quiz_method"] = method
                    st.session_state["quiz_context"] = passage
                    st.session_state["original_quiz_context"] = original_passage
                    st.session_state["elapsed"] = time.perf_counter() - started
                    st.session_state["verified"] = verify
                    if rows:
                        st.success(f"Created {len(rows)} questions. Open Question library to review and export.")
                        if learning_mode and len(rows) < count:
                            st.info("The planner found fewer targets, or some drafts did not pass the checks. Review the selected facts and their evidence.")
                    else:
                        st.info("No suitable questions passed the checks. Try a paragraph that explains a concept or process, or supply a specific target answer.")
                except Exception as exc:
                    st.error(f"Could not complete this quiz: {exc}")
    with results_tab:
        rows = st.session_state.get("questions", [])
        rejected = st.session_state.get("structural_rejections", [])
        if rejected:
            st.warning(f"Excluded {len(rejected)} question(s) whose requested list did not match the target answer.")
            with st.expander("Inspect excluded drafts"):
                st.write(rejected)
        if not rows:
            st.info("Your generated questions will appear here, with answers and source evidence.")
        else:
            quiz_method = st.session_state.get("quiz_method")
            if quiz_method is None and rows[0].get("method", "").startswith("transformer_"):
                quiz_method = "Pretrained Transformer"
            if quiz_method and quiz_method != method:
                st.info(f"This quiz was generated with {quiz_method}. Generate questions again to use {method}.")
            a, b, c = st.columns(3)
            a.metric("Questions", len(rows))
            b.metric("Generation + checks", f"{st.session_state.get('elapsed', 0):.1f} s")
            if st.session_state.get("verified"):
                c.metric("Original QA answer agreement (token F1)", f"{100 * sum(r.get('answer_f1', 0) for r in rows) / len(rows):.0f}%")
                st.caption("Scores describe the original generated questions. Edited wording is unchecked. QA recovery does not establish factual support or educational usefulness.")
            else:
                c.metric("Answerability checks", "Not run")
            show_answers = st.toggle("Show answers and evidence", value=True)
            export_rows = []
            context = st.session_state.get("quiz_context", "")
            if context != st.session_state.get("original_quiz_context", context):
                with st.expander("Passage used for generation"):
                    st.write(context)
                    st.caption("Wrapped lines were joined. Target offsets and worksheet evidence refer to this prepared passage; the original input is retained.")
            st.caption("Choose questions for the exported quiz. QA-flagged questions start unselected; inspect their evidence before including them.")
            for i, generated_row in enumerate(rows, 1):
                with st.container(border=True):
                    st.caption(f"QUESTION {i:02d} · {LANGUAGES[generated_row['language']]} · {generated_row['method']}")
                    with st.expander("Edit question wording"):
                        st.caption("Changing the wording clears its previous answerability scores. Inspect the source before including an edit.")
                        wording = st.text_area("Revised question", value=generated_row["question"], height=95,
                                               max_chars=400, key=f"wording_{st.session_state.get('quiz_revision', 0)}_{i}")
                    row = revise_question(generated_row, wording)
                    edited = row.get("teacher_edited", False)
                    if edited:
                        st.info("Wording edited · unchecked. Select it for export after reviewing the source.")
                        with st.expander("Original generated question"):
                            st.write(generated_row["question"])
                    if not row["question"]:
                        st.error("Question wording cannot be empty.")
                    scope_issue = list_scope_issue(row["question"], row["answer"], row["language"])
                    if scope_issue:
                        st.error(scope_issue)
                    st.markdown(f"<div class='question' dir='auto'>{html.escape(row['question'])}</div>", unsafe_allow_html=True)
                    if row.get("learning_goal"):
                        st.write("**Learning goal:** " + row["learning_goal"])
                        if row.get("draft_question"):
                            with st.expander("Compare with the T5 draft"):
                                st.write(row["draft_question"])
                                st.caption("The editor can change wording and focus. Inspect the passage for logical support; the displayed goal does not establish educational value.")
                    if show_answers:
                        st.write("**Answer:** " + row["answer"])
                        st.markdown(f"<div class='evidence' dir='auto'>{marked_evidence(row)}</div>", unsafe_allow_html=True)
                        if "answer_f1" in row:
                            if row.get("accepted"):
                                st.success("Passed QA consistency rules · review wording and source support")
                            else:
                                st.warning("Review suggested: " + "; ".join(row.get("review_reasons", [])))
                            with st.expander("Inspect the check"):
                                st.write({"QA model's answer": row["qa_answer"], "Answer token F1": round(row["answer_f1"], 3),
                                          "QA span score (uncalibrated)": round(row["qa_score"], 3),
                                          "Lexical relevance (cosine)": round(row["lexical_relevance"], 3)})
                                st.caption("Agreement supports review; it does not establish factual correctness. A low score can also reflect a weak QA verifier.")
                    wording_id = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:12]
                    if st.checkbox("Include in quiz exports", value=not edited and bool(row.get("accepted", True)),
                                   disabled=not row["question"] or bool(scope_issue),
                                   key=f"export_pick_{st.session_state.get('quiz_revision', 0)}_{i}_{wording_id}") and row["question"] and not scope_issue:
                        export_rows.append(row)
            st.subheader("Take it to class")
            st.write(f"Selected {len(export_rows)} of {len(rows)} questions for export.")
            if not export_rows:
                st.info("Select at least one question to enable quiz exports.")
            formats = st.columns(5)
            formats[0].download_button("JSON", as_json(export_rows), "questionforge.json", "application/json", disabled=not export_rows)
            csv_output = classroom_csv(export_rows)
            formats[1].download_button("CSV", csv_output, "questionforge.csv", "text/csv", disabled=not export_rows)
            context = st.session_state.get("quiz_context", "")
            formats[2].download_button("Worksheet", classroom_worksheet(export_rows, context), "worksheet.html", "text/html", disabled=not export_rows)
            formats[3].download_button("Answer key", classroom_worksheet(export_rows, context, True), "answer-key.html", "text/html", disabled=not export_rows)
            formats[4].download_button("Moodle XML", moodle_xml(export_rows), "questionforge.xml", "application/xml", disabled=not export_rows)
            st.caption("Print the HTML worksheet from your browser. Moodle imports one exact short-answer variant; review acceptable alternatives.")
    with research:
        science_folder = ROOT / "results/science_training"
        science_test = ROOT / "results/science_evaluation_test/summary.json"
        if science_test.exists() and (science_folder / "selection.json").exists():
            with st.expander("Educational science training: all eight settings", expanded=False):
                science_selection = json.loads((science_folder / "selection.json").read_text(encoding="utf-8"))
                science_results = json.loads(science_test.read_text(encoding="utf-8"))
                st.write("Eight equal-budget settings vary learning rate, LoRA rank and dropout. Checkpoint selection uses validation loss only.")
                st.write("Selected setting: " + science_selection["selected_trial"])
                trials = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(science_folder.glob("lr*.json"))]
                st.dataframe([{key: trial[key] for key in ["learning_rate", "rank", "dropout", "updates", "best_step", "best_dev_loss"]}
                              for trial in trials], hide_index=True)
                figure = science_folder / "validation_search.png"
                if figure.exists():
                    st.image(str(figure))
                st.write("Reserved science test: 200 passage/question/answer inputs, evaluated after selection.")
                st.dataframe([{"method": name, **{key: score[key] for key in ["bleu", "chrf_pp", "rouge_l", "answer_f1"]}}
                              for name, score in science_results["methods"].items() if name != "reference_control"], hide_index=True)
                st.caption("Reference-wording agreement improves; QA improvement is uncertain. Some outputs copy long cloze sentences. These scores do not measure educational usefulness.")
                comparison = ROOT / "results/science_evaluation_dev/comparison.html"
                if comparison.exists():
                    st.download_button("Inspect all science development questions", comparison.read_bytes(),
                                       "science-development-comparison.html", "text/html")
        learning_studies = {label: ROOT / folder for label, folder in [
            ("T5 with relation targets", "results/learning_t5_educational_v2"),
            ("Hybrid: educational passages", "results/learning_focus_educational_v3"),
            ("Hybrid: classroom passages", "results/learning_focus_dev_v3"),
            ("4B hybrid: educational passages", "results/learning_focus_educational_4b_v1"),
            ("4B hybrid: classroom passages", "results/learning_focus_dev_4b_v1"),
            ("4B additional synthetic topics", "results/learning_challenge_v1"),
            ("Answer-role edits: classroom passages", "results/learning_role_classroom_v1"),
            ("Answer-role edits: educational passages", "results/learning_role_educational_v1")]
            if (ROOT / folder / "summary.json").exists()}
        if learning_studies:
            with st.expander("Inspect learning-focused questions", expanded=True):
                comparison = st.selectbox("Learning-focused development comparison", list(learning_studies),
                                          index=len(learning_studies) - 1)
                comparison_folder = learning_studies[comparison]
                learning_summary = json.loads((comparison_folder / "summary.json").read_text(encoding="utf-8"))
                st.write(f"{learning_summary['passage_language_pairs']} passage/language pairs recorded.")
                if not learning_summary.get("study_complete", True):
                    st.info("This comparison is incomplete; inspect the cases recorded so far.")
                st.download_button("Download source-linked comparison", (comparison_folder / "comparison.html").read_bytes(),
                                   comparison_folder.name + ".html", "text/html")
                template = comparison_folder / "learning_review_template.csv"
                if template.exists():
                    st.download_button("Download learning-review template", template.read_bytes(),
                                       comparison_folder.name + "-review.csv", "text/csv")
                st.caption("Inspect source support, clarity and learning value. These are synthetic development examples; human ratings are pending. The answer-role comparisons hold target answers and T5 drafts fixed.")
        st.subheader("Measured results, reproducible comparisons")
        paths = sorted((ROOT / "results").glob("*/summary.csv"))
        if not paths:
            st.info("Run the evaluation commands in README.md to populate measured results.")
        else:
            import pandas as pd
            chosen = st.selectbox("Experiment", paths, format_func=lambda p: p.parent.name)
            frame = pd.read_csv(chosen)
            columns = [c for c in ["language", "method", "n", "chrf_pp", "rouge_l", "answer_f1", "accepted", "errors"] if c in frame]
            st.dataframe(frame[columns], use_container_width=True, hide_index=True)
            st.caption("chrF++, ROUGE-L and QA agreement are shown on 0–100 scales. Acceptance is rule-based coverage, not accuracy. Reference and unrelated questions diagnose the verifier.")
        st.markdown("**Fair comparison:** identical passages and target answers for the baseline, greedy decoding, and beam search. The main experiment supplies target answers; automatic answer selection is a separate demonstration feature.")
        st.markdown("**Known limits:** lexical overlap misses good paraphrases; automatic QA is an imperfect judge; checkpoint training exposure is partly unknown; multilingual checkpoints differ in size and training data.")
        if expanded_path.exists():
            with st.expander("Expanded training in English, French and Arabic", expanded=True):
                expanded = json.loads(expanded_path.read_text(encoding="utf-8"))
                st.dataframe([{"language": lang, "training_examples": info["training_examples"],
                               "development_examples": info["development_examples"], "updates": info["steps"],
                               "initial_dev_loss": info["initial_dev_loss"], "best_dev_loss": info["best_dev_loss"]}
                              for lang, info in expanded["languages"].items()], hide_index=True)
                st.image(str(expanded_path.parent / "comparison.png"))
                st.caption("Each language has its own trained adapter. App selection uses development loss, including the unadapted model and the previous English adapter as candidates. Test scores are not selection criteria.")
        if study_path.exists():
            with st.expander("Training stability across three seeds"):
                study = json.loads(study_path.read_text(encoding="utf-8"))
                st.dataframe(study["runs"], hide_index=True)
                figure = study_path.parent / "seed_stability.png"
                if figure.exists():
                    st.image(str(figure))
                st.caption("The same test inputs repeat across seeds; they are not additional independent examples. Checkpoints and the optional app adapter use development loss only.")
        verification_path = ROOT / "results/verification_study/summary.json"
        if verification_path.exists():
            with st.expander("Verifier ablation on constructed controls"):
                verification = json.loads(verification_path.read_text(encoding="utf-8"))
                st.dataframe([{k: v for k, v in r.items() if k != "conditions"}
                              for r in verification["models"] if r["partition"] == "test"], hide_index=True)
                st.caption("Reference questions and constructed negatives test intended-answer consistency. These are not human ratings of generated questions, and span scores are uncalibrated.")
    with review_tab:
        st.subheader("Independent review of question quality")
        st.write("Rate relevance, answerability and fluency with the passage in view. The offline booklet hides model names and automatic scores.")
        folder = ROOT / "results/human_evaluation"
        expanded_review = ROOT / "results/expanded_human_evaluation"
        if (expanded_review / "review_booklet.html").exists():
            study_label = st.selectbox("Review comparison", ["Expanded native adapters", "Original baseline and decoding"])
            if study_label == "Expanded native adapters":
                folder = expanded_review
        if not (folder / "items.json").exists():
            st.info("Prepare the review sample with python -m scripts.prepare_human_review.")
        else:
            items = json.loads((folder / "items.json").read_text(encoding="utf-8"))
            mapping = json.loads((folder / "researcher_mapping.json").read_text(encoding="utf-8"))
            st.download_button("Download offline review booklet", (folder / "review_booklet.html").read_bytes(),
                               f"questionforge-review-{folder.name}.html", "text/html")
            st.caption(f"{len(items)} questions prepared. Open the downloaded HTML file in a browser, rate languages you can judge, and export a CSV. No human study is claimed until ratings are supplied.")
            with st.expander("Scoring instructions"):
                for dimension, explanation in GUIDE.items():
                    st.write(f"**{dimension.title()}:** {explanation}")
            uploads = st.file_uploader("Import reviewer CSVs for analysis", type=["csv"], accept_multiple_files=True,
                                       key=f"review_csvs_{folder.name}")
            if uploads:
                try:
                    rows = []
                    for upload in uploads:
                        rows.extend(validate_ratings(upload.getvalue().decode("utf-8-sig"), items))
                    rows = validate_ratings(ratings_csv(rows), items)
                    summary = summarize_ratings(rows, items, mapping)
                    st.metric("Complete rating rows", summary["complete_rating_rows"])
                    st.metric("Complete paired reviewer/input blocks", summary["complete_paired_blocks"])
                    if summary["method_means_0_to_2"]:
                        st.dataframe(summary["method_means_0_to_2"], hide_index=True)
                    if summary["inter_rater_agreement"]:
                        st.dataframe(summary["inter_rater_agreement"], hide_index=True)
                    st.download_button("Download review analysis", json.dumps(summary, indent=2),
                                       "human-review-analysis.json", "application/json")
                    st.caption(summary["interpretation"])
                except (ValueError, UnicodeDecodeError) as exc:
                    st.error(f"Could not analyze ratings: {exc}")
            else:
                st.info("Human ratings pending. Prepared questions and automated checks are not completed human annotations.")


if __name__ == "__main__":
    main()
