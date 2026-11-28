# Course alignment and question-quality audit

Checked on 27 November 2026 against the three newly supplied slide files, the earlier
chapter notebooks, the project rubric, the implementation and saved predictions.
Page references below use physical PDF pages or PowerPoint slide order; some slides'
printed footer numbers differ. Instructor files remain in Downloads and are not redistributed.

## Does the project follow the chapters?

It follows the main concepts and uses a suitable model family. It is not a reproduction
of the classroom code, and conceptual alignment does not establish strong question quality.

| Supplied material | What it teaches | What this project actually implements |
|---|---|---|
| `NLP_week_7_8_9 (2).pdf`, pp. 5–13 | Transfer learning, matching tokenizers, encoder-only / decoder-only / encoder–decoder families; T5 and BART explicitly appear on p. 11 | Pinned T5/mT5/AraT5 question-generation checkpoints with their own tokenizers and answer-aware input formats. Encoder–decoder generation directly matches the assigned seq2seq task. |
| Same PDF, pp. 36–37, 52–58 | Small learning rates, training with validation, freezing versus full fine-tuning; class example is full fine-tuning a small BERT classifier using KerasHub | Teacher-forced question-token cross-entropy, AdamW, validation loss, checkpoint selection and early stopping. PyTorch/PEFT updates low-rank attention adapters while the base weights stay frozen. This is parameter-efficient adaptation, rather than the full-backbone class example. |
| Same PDF, pp. 41–48; `GPT2_Coding_Exercise.ipynb` | Causal next-token generation and inference-time decoding choices | T5's decoder generates autoregressively while also attending to the encoder. Greedy and beam-4 are compared. The project does not train GPT-2 or compare top-k/top-p sampling. |
| `Week 10 (3).pptx`, slides 5–24 | Parameters versus hyperparameters, learning rate, batch size, regularization, sequence length, validation-only model selection and KerasTuner | The original experiments use fixed LoRA settings and multiple seeds. The follow-up SciQ study now completes an equal-budget eight-setting search over learning rate, rank and dropout, selecting checkpoints on validation loss only. It uses PyTorch rather than KerasTuner. Multiple seeds alone are not a hyperparameter search. |
| Same deck, slides 26–43 | Precision/recall/F1, thresholds, false positives/negatives, inspecting examples and fair comparisons | Reference-overlap metrics, answer-token F1/EM, positive/negative verifier controls, paired intervals and source-linked failure analysis. There are no representative human correctness labels from which to report generation accuracy or a natural-question correctness confusion matrix. |
| `NLP_Week12_13 (2).pptx`, slides 5, 8, 13 | Systems combining components; Hugging Face Transformers; prompting at inference | A local generation/checking/export workflow using Hugging Face models. These task-tuned seq2seq checkpoints use their documented training prompts; they are not assumed to obey arbitrary chat instructions. |
| Same deck, slides 14–20, 27–30 | RAG: overlapping document chunks, embeddings, semantic retrieval, evidence-based generation | The user supplies a passage; the model receives an answer-centered window. There is no vector database, embedding retriever or document-collection search. It should not be presented as the slides' RAG implementation. |

The Week 10 classroom exercise on slide 43 asks for at least three tuned neural-network
hyperparameters and classification metrics. It is a classroom exercise, not an additional
final-project requirement. The separately supplied project rubric accepts TensorFlow/Keras
**or PyTorch**, asks for at least two meaningful model/configuration comparisons, and requires
metrics appropriate to the assigned task. Full large-model fine-tuning is not mandatory.
Classification accuracy and class F1 cannot simply be substituted for generation evaluation.

LoRA learning rate `3e-4` applies to newly trained adapter parameters. The class's `2e-5`
example updates pretrained BERT weights. Their different parameterizations make a direct
learning-rate comparison misleading.

## Why the model can still feel weak

The following are specific observed weaknesses, rather than deductions from passing tests.

1. **Unsupported relations can pass QA checks.** For `novel:solar_lab:ar:0`, the passage says
   Rami connected a solar panel to a lamp. Both the original and expanded Arabic models ask
   `من الذي قام بإختراع المصباح؟` — “Who invented the lamp?” The verifier extracts Rami and
   assigns perfect answer-token F1. The question's invention claim is unsupported. The old
   qualitative note incorrectly called this a supported connection question; it has been
   corrected in the source annotations, exported examples and paired comparison.
2. **A different fact can replace the target fact.** In French public case `fr:2739`, the
   target Ankara is the party providing military aid. The expanded question instead asks
   who initiated the offensive. QA recovery of Ankara does not repair that changed relation.
   In the English David/Trajan case, both models combine unrelated sculpture facts into
   an unsupported premise, again with perfect QA answer agreement.
3. **Automatic answer selection has its own failures.** On the existing development reading-club
   passage, the French selector proposes `club lit` (“club reads”) as a target. In Arabic it
   proposes `عشرة` from `الثامنة عشرة` (“eighteenth”), and `تدير سلمى النقاش` (“Salma leads the
   discussion”) as a whole clause. These were reproduced with `extract_candidates(..., limit=3)`.
   They are weak inputs for answer-conditioned generation. The controlled benchmarks supply
   exact target answers and therefore do not measure this end-to-end problem.
4. **More training did not consistently solve quality.** Expanded English/French/Arabic
   development losses fell, but held-out generation results remain mixed. For example, French
   public QA F1 falls from 58.38 to 51.12 and Arabic classroom QA F1 from 66.20 to 64.81.
   English improves on QA recovery but reference ROUGE-L decreases. These are small repeated
   evaluation sets, and their paired intervals do not establish a consistent improvement.
   Full figures and caveats are in [the measured training findings](EXPANDED_TRAINING_RESULTS.md).
5. **The app's adapted-model toggle starts off disabled.** Selecting a language alone runs
   its original pretrained generator. The optional adapted generator was kept separate
   because the measured outcomes are mixed. Training completion does not mean the default
   interface now uses the new weights.

The eleven qualitative examples were re-read against their saved passages/questions during
this audit. They remain purposively selected observations, not human scores, a representative
error sample or an estimate of error frequency. Predictions and aggregate numerical results
were preserved; correcting an explanatory annotation does not improve model weights.

## What would constitute a meaningful improvement?

### Completed science follow-up

The English SciQ study now implements the proposed three-hyperparameter search on a new
educational source. It uses 2,043 usable training / 126 validation examples after token-budget
filtering and reserves 200 published test examples. All eight trials receive 256 updates and
the same example order. Validation selects learning rate 3e-4, rank 16 and dropout .05;
token loss changes from 1.1829 to .9964. Exact source and all adapter bundles are retained.

On the reserved science test, BLEU changes from 31.83 to 42.11 and ROUGE-L from 59.83 to
64.15. QA answer F1 changes from 72.63 to 74.11; its paired 95% interval for the difference
is [-2.34, 5.60] percentage points. This is better reference-wording agreement, without an
established directional QA improvement or a human usefulness result. Development inspection
finds clearer target focus in some questions and long copied/malformed cloze questions in
others. The educational domain and adaptation both change; this is not an isolated causal
test of data quantity. Source-level public pretraining exposure remains unknown.

See [the measured science report](SCIENCE_TRAINING_RESULTS.md). The learning-focused
candidate-selection and planner/editor comparisons are separate development experiments,
documented in [the learning workflow](LEARNING_FOCUS.md). Neither exact spans nor QA
agreement certifies logical relationships.

### Answer-role follow-up, 28 November

The stronger local 4B planner/editor is now downloaded and evaluated. The selected
writing recipe adds source-derived output/purpose/limitation/comparison cues. A targeted
12-response pilot and three-response Arabic grammatical probe precede the final recipe;
the extra Arabic instruction fails to fix conversion wording and is discarded.

A controlled re-edit keeps 50 targets and T5 drafts fixed across 18 synthetic
passage/language cases. Coverage stays at 24/25 educational and 25/25 classroom targets,
with zero execution errors. Conditional QA F1 rises from 71.83 to 73.52 on educational
cases and falls from 48.80 to 45.94 on classroom cases. Source inspection finds actual
purpose/comparison repairs and wording regressions; these are not human usefulness scores.
The additional-topic audit is reported separately in [the answer-role findings](ANSWER_ROLE_RESULTS.md).

Qwen is an inference-time decoder-only extension. The native T5 encoder–decoder and
measured LoRA training remain the course-aligned generative model work. The app also
allows wording corrections before export, clears stale QA results on edits and retains
the original generated question/model. Software checks and edit controls support a
review workflow; they do not certify the question's factual premise or learning value.

### Recommendations retained from the initial audit

The initial audit proposed the directions below. The SciQ grid and role-guided comparisons
address parts of that plan; the remaining items are future work:

- Measure automatic target selection separately and then evaluate the complete pipeline.
  Improve complete entity/number spans and reject pronouns, verb fragments and partial clauses.
- Extend the completed learning-rate/rank/dropout grid across budgets or seeds only with
  validation-based selection. Assess generated development questions as well as token loss,
  and preserve a fresh, independently authored final evaluation set for future configurations.
- Compare full-passage context with sentence/neighboring-sentence evidence windows. This tests
  the evidence-focus idea without claiming vector-retrieval RAG. Shorter context may reduce
  irrelevant relations but can also lose antecedents, so it needs measured comparison.
- Evaluate factual relations, target focus and fluency directly. QA agreement remains one
  diagnostic. A second automatic model is not a substitute for correctness labels, and adding
  an unsupported-word rule alone would not catch all semantic relation errors.

The current implementation is a defensible course project with meaningful experiments. It
has not demonstrated dependable multilingual question quality, and this audit does not assign
a grade or claim that software checks establish educational usefulness.

## Week 10 and the unused-data audit

The final app now presents one fixed Qwen 4B/native T5/original QA workflow. Older
model, adapter and decoder alternatives discussed in the historical audits remain
research configurations and saved comparisons. See [the final workflow](FINAL_WORKFLOW.md).

Physical slides 7/11 cover dropout/early stopping, slide 19 separates training,
validation and final test roles, slide 20 discusses configuration search, and slides
41/42 motivate error analysis and fair comparisons. The existing eight-setting grid
varies learning rate/rank/dropout and restores validation-selected weights. Its fixed
budget is not the same as patience-based early termination of every trial.

The classroom repair changes structural input/answer selection, not learned weights.
After that repair was frozen, another 12 human-authored multilingual paragraphs and
50 SciQ pairs were evaluated without test-based tuning. The selected adapter improves
BLEU but reduces QA recovery descriptively on the additional SciQ sample, with a
paired interval crossing zero. This is mixed evidence, not a diagnosis of overfitting
or proof that more training is better. See [all results](GENERALIZATION_RESULTS.md).
