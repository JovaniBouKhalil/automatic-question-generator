# One final application workflow

The app has one generation path in English, French and Arabic:

**Prepared passage → Qwen 4B fact selection → native T5 drafts → Qwen refinement → QA review flags → review/export.**

Choose the language and maximum question count, then paste/upload your material.
Optional exact target answers constrain the selected facts; an optional learning objective
guides selection and refinement. Review source evidence and wording before exporting.

The generation-method, decoding, adapter, answerability-checker and editing-recipe
selectors are removed. The settings are fixed to the existing classroom workflow:

| Component | Fixed setting |
|---|---|
| Planner/editor | Pinned Qwen3-4B-Instruct-2507, FP32 CPU, greedy decoding |
| Draft generator | Native unadapted English T5, French mT5 or Arabic AraT5 |
| Draft decoding | Beam 4 |
| Target selection | Prepared source relation/list/statement candidates |
| Refinement | Existing answer-role guidance enabled |
| QA | Original independent extractive checker, always run |

These steps run sequentially. The checker flags drafts for review; it does not guarantee
correctness. Editing wording clears the old QA scores, preserves the generated wording
and starts the revision unselected for export. Rejected list/count mismatches remain
inspectable. Older quizzes in an open session are labeled for regeneration.

There is no alternate generation fallback when Qwen is missing. The app explains setup
and disables generation until its verified 4B weights are installed. Download the core
models and Qwen using the commands in README. The final app needs roughly 16 GB for
Qwen parameters alone, plus the native generator/checker and runtime memory. Allow
substantial additional RAM; CPU generation takes time.

The model weights, prompts, candidate rules and thresholds are unchanged by this UI
consolidation. The frozen unused-data evaluation still describes this same hybrid
configuration. Its failures and limited evidence remain applicable; consolidation does
not improve the measured question quality. No new training or test-based tuning occurs.

Baseline, smaller-model, adapter and decoder comparisons remain research artifacts and
reproduction scripts for the course rubric. They appear in Experiment results and the
notebook, rather than as alternative quiz-generation options. Git history preserves
earlier checkpoints; `main` and `submission/QuestionForge-submission.zip` contain the
single current application.

Long chapters are automatically sectioned with the actual Qwen tokenizer and processed
through this same workflow. Selected section coverage is displayed and source offsets
remain relative to the prepared full document. More than eight requested questions are
batched. See [the input support and remaining coverage limits](LONG_DOCUMENTS.md).
The model settings remain fixed; long-document orchestration is a separate development
extension and is not included in the earlier frozen small-passage quality claims.
