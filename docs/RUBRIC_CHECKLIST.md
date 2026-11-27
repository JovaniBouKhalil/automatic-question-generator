# Rubric evidence checklist

This maps the supplied 100-point rubric to concrete artifacts. It is not a grade guarantee.

| Criterion | Points | Evidence |
|---|---:|---|
| Dataset suitability, sources, exploration, splits | 15 | Data manifest and length figures; sources/licenses; exact span checks; paragraph tests; quantified near-duplicate/title audit; English article-isolated training sensitivity |
| Baseline and advanced models, justification, comments | 25 | `baseline.py`; native T5/mT5/AraT5 generators in `models.py`; checkpoint-specific prompts; answer-preserving cropping; course-facing notebook |
| Configuration, variations, fair comparison, reproducibility | 20 | Baseline/greedy/beam on identical IDs; original LoRA plus three 160-update seeds; article sensitivity; fixed protocol, hashes/revisions/versions; development-only selection; resumable predictions |
| Suitable metrics, figures, failures, interpretation | 20 | chrF++, BLEU, Unicode ROUGE-L, QA F1/EM; figures; paired topic intervals and seed dispersion; source-linked failures; 54-item blinded booklet and paired/inter-rater analysis (human ratings pending) |
| Project-specific advanced component | 10 | Independent QA/relevance and review rules; two-verifier ablation with 720 model/control records, three negative constructions, per-language false acceptance and topic-bootstrap differences |
| Slides, working demonstration, technical Q&A | 10 | 12-slide PPTX/PDF with notes; app, optional checker/adapter and exports; 8 executed cells; defense guide; 46 unit/workflow and 8 real-model integration checks |

Before submission: annotate a representative sample as a human, personalize the slides, run the
demo in each language, and confirm you can explain every model/input/metric choice. The rubric
requires no standalone technical report. Multilingual support is an extension; experimental
rigor and your understanding remain essential.
