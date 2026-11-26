# Rubric evidence checklist

This maps the supplied 100-point rubric to concrete artifacts. It is not a grade guarantee.

| Criterion | Points | Evidence |
|---|---:|---|
| Dataset suitability, sources, exploration, splits | 15 | `questionforge/data.py`; data manifest, counts and length figure in `results/data/`; source/license table; exact span checks; paragraph-group partition tests |
| Baseline and advanced models, justification, comments | 25 | `baseline.py`; native T5/mT5/AraT5 generators in `models.py`; checkpoint-specific prompts; answer-preserving cropping; course-facing notebook |
| Configuration, variations, fair comparison, reproducibility | 20 | Baseline vs greedy vs beam-4 on identical IDs; English LoRA extension; seed, hashes, revisions and versions; development-only selection; resumable predictions |
| Suitable metrics, figures, failures, interpretation | 20 | chrF++, BLEU, Unicode ROUGE-L, QA F1/EM; comparison figures; paired bootstrap intervals; ranked review examples; limitations and blank human annotation sheet |
| Project-specific advanced component | 10 | Independent answerability assessment, lexical relevance, answer leakage and review coverage; reference/unrelated verifier controls; separate multilingual results |
| Slides, working demonstration, technical Q&A | 10 | 11-slide PPTX with notes; Streamlit application and classroom exports; runnable notebook; defense/demo guide |

Before submission: annotate a representative sample as a human, personalize the slides, run the
demo in each language, and confirm you can explain every model/input/metric choice. The rubric
requires no standalone technical report. Multilingual support is an extension; experimental
rigor and your understanding remain essential.
