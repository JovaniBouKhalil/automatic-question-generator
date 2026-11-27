# Rubric evidence checklist

This maps the supplied 100-point rubric to concrete artifacts. It is not a grade guarantee.

| Criterion | Points | Evidence |
|---|---:|---|
| Dataset suitability, sources, exploration, splits | 15 | Sources/licenses and manifests; exact span checks; near-duplicate/title audit; article-isolated sensitivity; expanded SQuAD/NewsQuAD-fr/TyDiQA+ARCD splits with full validation-source protection |
| Baseline and advanced models, justification, comments | 25 | Transparent baseline; native T5/mT5/AraT5; three larger-data LoRA adapters; native Arabic target-format audit; answer-preserving cropping; exact training-source snapshots; course-facing notebook |
| Configuration, variations, fair comparison, reproducibility | 20 | Baseline/greedy/beam; original LoRA, three seeds and article sensitivity; larger 600/256/512-update runs; 21 generation experiments; fixed protocols, shared-development selection, actual exposure counts, hashes and resumable state |
| Suitable metrics, figures, failures, interpretation | 20 | chrF++, BLEU, Unicode ROUGE-L, QA F1/EM; intervals/seed dispersion; mixed outcomes preserved; ten paired before/after cases; original 54-item and new 36-item blinded booklets with paired/inter-rater analysis (ratings pending) |
| Project-specific advanced component | 10 | Independent QA/relevance and review rules; two-verifier ablation with 720 model/control records, three negative constructions, per-language false acceptance and topic-bootstrap differences |
| Slides, working demonstration, technical Q&A | 10 | 12-slide PPTX/PDF; app with three native adapters, optional checker and exports; 9 executed cells including trained generation; defense guide; 53 unit/workflow and 10 real-model integration checks |

Before submission: annotate a representative sample as a human, personalize the slides, run the
demo in each language, and confirm you can explain every model/input/metric choice. The rubric
requires no standalone technical report. Multilingual support is an extension; experimental
rigor and your understanding remain essential.
