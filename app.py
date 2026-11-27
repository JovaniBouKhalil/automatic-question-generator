"""Local Streamlit demonstration. Run: python -m streamlit run app.py"""

from __future__ import annotations

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
from questionforge.export import as_csv, as_json, moodle_xml, worksheet_html
from questionforge.learning.pipeline import create_learning_quiz
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
        method = st.selectbox("Generation method", ["Pretrained Transformer", "Template baseline",
                                                    "Learning-focused T5 (experimental)",
                                                    "Learning-focused hybrid (experimental)"])
        learning_mode = method.startswith("Learning-focused")
        if learning_mode:
            if "hybrid" in method:
                st.caption("Selects important facts, creates T5 drafts, then edits questions for understanding. The local editor takes longer.")
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
        count = st.slider("Questions to create", 1, 8, 3)
        st.divider()
        st.markdown("**Works locally. No paid API.**")
        st.caption("The first use downloads model weights. Once cached, generation works offline. CPU inference is supported.")
        with st.expander("About the models"):
            st.write(GENERATORS[language].model_id)
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
            passage_key = f"passage_{language}" + ("_learning" if learning_mode else "")
            if passage_key not in st.session_state:
                st.session_state[passage_key] = samples[sample]
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
        with right:
            st.subheader("02 / Choose your facts")
            st.write("Add exact answers from the passage, one per line. Leave this blank to suggest key facts automatically.")
            manual = st.text_area("Target answers (optional)", key=f"answers_{language}", height=130,
                                   placeholder={"en": "Lina\nNine", "fr": "Lina\nNeuf", "ar": "لينا\nتسع"}[language])
            objective = ""
            if learning_mode:
                objective = st.text_area("What should students understand? (optional)", max_chars=500,
                                         help="For example: how the components work, the steps of a process, or a cause stated in the passage.")
                if "hybrid" not in method:
                    st.caption("In T5 mode this goal is saved for your review; automatic selection follows explicit relation markers. Use manual targets to control the facts.")
                st.caption("Focuses on concepts, functions and relationships stated in your passage. It may return fewer questions when there are too few useful facts.")
            else:
                st.caption("Automatic selection uses TF-IDF and simple span heuristics. Specific, short answers usually give clearer questions.")
            if st.button("Generate questions →", type="primary", use_container_width=True):
                # A failed new request must not leave an earlier quiz looking current.
                st.session_state["questions"] = []
                st.session_state.pop("learning_report", None)
                try:
                    if not passage.strip():
                        raise ValueError("Add a passage first.")
                    answers = list(dict.fromkeys(line.strip() for line in manual.splitlines() if line.strip()))
                    if len(answers) > 8:
                        raise ValueError("Use at most eight manual answers per quiz.")
                    for answer in answers:
                        evidence_for(passage, answer)
                    candidates = [] if learning_mode else ([(answer, passage.find(answer)) for answer in answers[:count]] if answers else
                                  [(c.answer, c.start) for c in extract_candidates(passage, language, count)])
                    if not candidates and not learning_mode:
                        raise ValueError("No suitable facts were found. Enter an exact target answer from the passage.")
                    rows = []
                    started = time.perf_counter()
                    with st.status("Building your question set…", expanded=True) as status:
                        if learning_mode:
                            create = create_learning_quiz if "hybrid" in method else create_learning_t5_quiz
                            rows, learning_report = create(passage, language, count,
                                        objective=objective, manual_answers=answers, beams=beams,
                                        checkpoint=checkpoint, progress=st.write)
                            st.session_state["learning_report"] = learning_report
                            for row in rows:
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
                        status.update(label="Your question set is ready", state="complete", expanded=False)
                    st.session_state["questions"] = rows
                    st.session_state["quiz_context"] = passage
                    st.session_state["elapsed"] = time.perf_counter() - started
                    st.session_state["verified"] = verify
                    if rows:
                        st.success(f"Created {len(rows)} questions. Open Question library to review and export.")
                        if learning_mode and len(rows) < count:
                            st.info("Returned fewer questions because some proposed facts or questions did not pass source-span and format checks.")
                    else:
                        st.info("No suitable learning-focused questions passed the checks. Try a passage that explains a concept or process, or supply a specific target answer.")
                except Exception as exc:
                    st.error(f"Could not complete this quiz: {exc}")
    with results_tab:
        rows = st.session_state.get("questions", [])
        if not rows:
            st.info("Your generated questions will appear here, with answers and source evidence.")
        else:
            a, b, c = st.columns(3)
            a.metric("Questions", len(rows))
            b.metric("Generation + checks", f"{st.session_state.get('elapsed', 0):.1f} s")
            if st.session_state.get("verified"):
                c.metric("QA answer agreement (token F1)", f"{100 * sum(r.get('answer_f1', 0) for r in rows) / len(rows):.0f}%")
            else:
                c.metric("Answerability checks", "Not run")
            show_answers = st.toggle("Show answers and evidence", value=True)
            for i, row in enumerate(rows, 1):
                with st.container(border=True):
                    st.caption(f"QUESTION {i:02d} · {LANGUAGES[row['language']]} · {row['method']}")
                    st.markdown(f"<div class='question' dir='auto'>{html.escape(row['question'])}</div>", unsafe_allow_html=True)
                    if row.get("learning_goal"):
                        st.write("**Learning goal:** " + row["learning_goal"])
                        if row.get("draft_question"):
                            with st.expander("Compare with the T5 draft"):
                                st.write(row["draft_question"])
                                st.caption("The editor can change wording and focus. Inspect the passage for logical support; the learning goal is a model proposal.")
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
            st.subheader("Take it to class")
            formats = st.columns(5)
            formats[0].download_button("JSON", as_json(rows), "questionforge.json", "application/json")
            formats[1].download_button("CSV", as_csv(rows), "questionforge.csv", "text/csv")
            formats[2].download_button("Worksheet", worksheet_html(rows), "worksheet.html", "text/html")
            formats[3].download_button("Answer key", worksheet_html(rows, True), "answer-key.html", "text/html")
            formats[4].download_button("Moodle XML", moodle_xml(rows), "questionforge.xml", "application/xml")
            st.caption("Print the HTML worksheet from your browser. Moodle imports one exact short-answer variant; review acceptable alternatives.")
    with research:
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
