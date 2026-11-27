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
| `Week 10 (3).pptx`, slides 5–24 | Parameters versus hyperparameters, learning rate, batch size, regularization, sequence length, validation-only model selection and KerasTuner | Fixed documented LoRA settings, development-selected checkpoints, multiple seeds and decoding comparisons. Learning rate, rank and dropout are configured, but have not been systematically searched. Multiple seeds are not a hyperparameter search. |
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

The next experiment should address these observed errors, rather than extend training without
a selection criterion. These are recommendations, not completed experiments:

- Measure automatic target selection separately and then evaluate the complete pipeline.
  Improve complete entity/number spans and reject pronouns, verb fragments and partial clauses.
- Compare a small, fixed search of learning rate, adapter rank and dropout on the same training
  and development splits. Assess generated development questions as well as token loss.
  Preserve a fresh, independently authored final evaluation set for the chosen configuration.
- Compare full-passage context with sentence/neighboring-sentence evidence windows. This tests
  the evidence-focus idea without claiming vector-retrieval RAG. Shorter context may reduce
  irrelevant relations but can also lose antecedents, so it needs measured comparison.
- Evaluate factual relations, target focus and fluency directly. QA agreement remains one
  diagnostic. A second automatic model is not a substitute for correctness labels, and adding
  an unsupported-word rule alone would not catch all semantic relation errors.

The current implementation is a defensible course project with meaningful experiments. It
has not demonstrated dependable multilingual question quality, and this audit does not assign
a grade or claim that software checks establish educational usefulness.
