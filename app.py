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
from questionforge.metrics import leaks_answer, lexical_relevance, quality_decision
from questionforge.models import generate_question, verify_question
from questionforge.registry import GENERATORS
from questionforge.text import LANGUAGES, evidence_for

ROOT = Path(__file__).resolve().parent


def sample_passages(language):
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
        method = st.selectbox("Generation method", ["Pretrained Transformer", "Template baseline"])
        beams = st.select_slider("Decoding beams", options=[1, 4], value=4,
                                  help="1 = greedy decoding. 4 = deterministic beam search.")
        verify = st.toggle("Check answerability", value=True,
                           help="An independent QA model attempts to recover the answer. This is an imperfect proxy.")
        count = st.slider("Questions to create", 1, 8, 3)
        st.divider()
        st.markdown("**Works locally. No paid API.**")
        st.caption("The first use downloads model weights. Once cached, generation works offline. CPU inference is supported.")
        with st.expander("About the models"):
            st.write(GENERATORS[language].model_id)
            st.caption("Questions are drafts for teacher review. The verifier is fine-tuned on English; French and Arabic checks may be weaker.")
    direction = "rtl" if language == "ar" else "ltr"
    alignment = "right" if language == "ar" else "left"
    font_size = "18px" if language == "ar" else "15px"
    st.markdown(f"<style>textarea{{direction:{direction};text-align:{alignment};font-family:Arial,sans-serif!important;font-size:{font_size}!important}}</style>", unsafe_allow_html=True)
    st.markdown("<div class='eyebrow'>READ. QUESTION. UNDERSTAND.</div>", unsafe_allow_html=True)
    st.title("Turn a passage into a learning moment.")
    st.markdown("<p class='intro'>Create factual questions in English, French, and Arabic. Keep the source evidence close—and see which questions need a second look.</p>", unsafe_allow_html=True)
    compose, results_tab, research = st.tabs(["Create a quiz", "Question library", "Experiment results"])
    with compose:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("01 / Your learning material")
            samples = sample_passages(language)
            sample = st.selectbox("Try a classroom passage", list(samples))
            passage_key = f"passage_{language}"
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
            st.caption("Automatic selection uses TF-IDF and simple span heuristics. Specific, short answers usually give clearer questions.")
            if st.button("Generate questions →", type="primary", use_container_width=True):
                try:
                    if not passage.strip():
                        raise ValueError("Add a passage first.")
                    answers = list(dict.fromkeys(line.strip() for line in manual.splitlines() if line.strip()))
                    if len(answers) > 8:
                        raise ValueError("Use at most eight manual answers per quiz.")
                    for answer in answers:
                        evidence_for(passage, answer)
                    candidates = ([(answer, passage.find(answer)) for answer in answers[:count]] if answers else
                                  [(c.answer, c.start) for c in extract_candidates(passage, language, count)])
                    if not candidates:
                        raise ValueError("No suitable facts were found. Enter an exact target answer from the passage.")
                    rows = []
                    started = time.perf_counter()
                    with st.status("Building your question set…", expanded=True) as status:
                        for index, (answer, start) in enumerate(candidates, 1):
                            st.write(f"Question {index}/{len(candidates)}")
                            if method == "Template baseline":
                                row = {"question": template_question(passage, answer, language, start),
                                       "answer": answer, "answer_start": start, "language": language,
                                       "evidence": evidence_for(passage, answer, start).text, "method": "baseline"}
                            else:
                                row = asdict(generate_question(passage, answer, language, answer_start=start, beams=beams))
                            row["lexical_relevance"] = lexical_relevance(row["question"], row["evidence"], language)
                            row["answer_leak"] = leaks_answer(row["question"], answer, language)
                            if verify:
                                row.update(verify_question(row["question"], passage, answer, language))
                                row["accepted"], row["review_reasons"] = quality_decision(row)
                            rows.append(row)
                        status.update(label="Your question set is ready", state="complete", expanded=False)
                    st.session_state["questions"] = rows
                    st.session_state["quiz_context"] = passage
                    st.session_state["elapsed"] = time.perf_counter() - started
                    st.session_state["verified"] = verify
                    st.success(f"Created {len(rows)} questions. Open Question library to review and export.")
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


if __name__ == "__main__":
    main()
