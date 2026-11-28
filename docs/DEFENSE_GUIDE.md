# Demo and technical defense guide

## A five-minute demonstration

1. Launch `python -m streamlit run app.py`. Start in English with a classroom example.
2. Enter a specific target answer from the passage. Generate a question and show its evidence.
3. Explain the independent QA answer, token F1 and uncalibrated span score. Inspect a weak example.
4. Repeat in French and Arabic; point out Arabic text direction and native-language checkpoints.
5. Hide answers for a student worksheet, then export the teacher answer key or Moodle XML.
6. Open measured experiment results and compare baseline, greedy and beam decoding.

Have cached models and the executed notebook ready. If the network is unavailable, cached
inference works. The template baseline with checks disabled is an explicitly labeled alternative.

## Course connections

The supplied slides have now been checked directly. See [the course alignment audit](COURSE_ALIGNMENT.md)
for exact page references, implementation differences, and the generation weaknesses that remain.
The implementation follows the transfer-learning workflow, but does not copy the Keras labs:
it uses PyTorch, encoder-decoder generation and LoRA adaptation. Do not describe the current
application as a full RAG system. The later science study does implement an eight-setting
learning-rate/rank/dropout search; this is separate from the original short adaptation run.

| Course material | Connection |
|---|---|
| Weeks 1–3: vectorization, entities and exploratory work | Inspect lengths and source spans; automatic answer selection ranks TF-IDF phrases and simple name/number heuristics. These heuristics are not a trained NER system. |
| Week 4: BoW and TF-IDF | Explain term weighting, stopwords, lexical cosine and limits of word overlap. |
| Week 5: baseline and mistake inspection | A transparent baseline and per-example review precede claims about advanced models. |
| Week 6: RNN/LSTM and overfitting | Contrast recurrent sequential state with attention; development selection avoids test tuning. |
| Transformers notebook | Token representations, attention, residuals, normalization and feed-forward blocks appear in pretrained T5. |
| Weeks 7–9: pretrained model families and fine-tuning | The supplied PDF explicitly includes T5/BART encoder–decoder models (p. 11). We reuse matching pretrained tokenizers and task checkpoints. The class fine-tunes BERT in Keras; our generator adapts T5-family models with PyTorch/LoRA. |
| GPT-2 exercise | Autoregressive next-token decoding and greedy/beam choices; T5 additionally conditions through encoder cross-attention. |
| Week 10: hyperparameters and evaluation | Shared splits, validation-only selection and error inspection follow the slides. The later eight-setting science grid searches learning rate/rank/dropout and restores the best validation checkpoint within each fixed budget; this is distinct from patience-based early termination. The frozen unused-data audit retains mixed results. Classification F1 differs from our answer-token F1. |
| Weeks 12–13: modern NLP systems and RAG | Hugging Face inference and a generator/checker workflow follow the system-level discussion. Supplying a passage and cropping around an answer is not the slides' embedding/vector-retrieval RAG pipeline. |

## Questions you should be able to answer

**Can a correct question be educationally weak?** Yes. Recovering an incidental name can
score perfectly without testing the passage's main idea. The learning-focused extension
targets concepts, functions and explicit relationships, with a separate proposed learning
goal. Its usefulness needs direct review; it cannot be established by QA F1.

**Which model writes the new questions?** In learning-focused T5 mode, the native T5 model
writes the question after a transparent relation-span selector chooses a target. In hybrid
mode, Qwen plans targets, T5 writes drafts, and Qwen edits them against the passage and goal.
Final hybrid questions therefore include a decoder-only model's contribution. The app
shows the T5 draft so this contribution is inspectable. Qwen is used with frozen pretrained
weights, not described as newly trained; CPU quantization is a runtime choice.

**Does the learning-objective field change both modes?** In hybrid mode it conditions the
planner and editor. In T5-only mode it is recorded for review; automatic selection follows
explicit relation markers. Manual exact targets control either mode.

**Why T5 instead of GPT-2?** The task maps a given passage/answer to a question. Encoder–decoder
models directly support conditional sequence generation, and compact task-tuned checkpoints exist.
GPT-2 is decoder-only and would require a different prompting/training design.

**How does attention work here?** Encoder self-attention uses the full input. Decoder self-attention
is causal, and cross-attention attends to encoder states. Tokens are generated one at a time.

**Why three checkpoints?** They have documented task-specific input formats and language training.
This provides native language support without assuming an English model understands Arabic. Their
sizes/data differ, so cross-language performance is not an architecture-controlled experiment.

**What is the baseline?** Replace the supplied answer in its evidence sentence with a WH phrase.
It uses the same input information as the Transformer but lacks learned syntax and paraphrasing.
It is an honest cheap comparison, although its grammar is limited.

**What is TF-IDF used for?** Ranking candidate answer phrases in the template/original
selection workflow and as a fallback in native automatic mode. The classroom hybrid
instead chooses source-supported relation/list/statement candidates with Qwen. TF-IDF
does not learn the generator. Controlled benchmark scores use provided answers; the
separate 12-case automatic audit tests selection and writing without a supplied answer.
Neither QA recovery nor overlap scores are fully end-to-end correctness percentages.

**Why not remove punctuation, lowercase everything or stem Arabic?** Pretrained tokenizers expect
natural text; aggressive preprocessing loses information and corrupts source character offsets.
Separate light normalization is used only for duplicate grouping or answer metrics.

**How do you avoid leakage?** Reserve evaluation paragraphs, drop matching training paragraphs,
split remaining train/dev by normalized paragraph group, sample deterministically, and assert
disjointness. Related articles/near duplicates remain a limitation. Public checkpoint exposure
is partly unknown; the small original supplement addresses only some of that concern.

**Why chrF++ and ROUGE-L?** chrF++ includes character n-grams and word bigrams, useful across
writing systems and inflections. ROUGE-L measures ordered token overlap. Both can undervalue
correct paraphrases and cannot establish factuality. Our Unicode tokenizer keeps Arabic text.

**What does QA F1 measure?** Token overlap between the independent QA model's extracted answer
and the intended answer. It is a proxy for asking about the intended fact. It can be high for
bad questions or low when the verifier fails, hence reference and unrelated controls.

**Is the QA score confidence that the question is correct?** No. It is the model's span score,
not a calibrated correctness probability. Rule-based acceptance is coverage, not measured accuracy.

**How independent is the verifier?** It has separate parameters and a different architecture
from the question generator. That does not imply statistically independent errors: their public
training corpora can overlap. The expected target is used only after QA inference to score the
extracted answer; it is not supplied to the verifier's question/passage input.

**Why the controls?** If the verifier cannot answer a good reference question in Arabic, its
judgment of generated Arabic questions is weak evidence. If it gives the target answer to an
unrelated question, that exposes false agreement. Interpret generated scores with both controls.

**What is LoRA?** For frozen projection W, learn low-rank BA so the effective weight is W + BA,
with rank far below the matrix dimensions. We update Q/V attention projections. This reduces
trainable parameters; inference can merge the update. It does not guarantee improved generation.

**What is the training loss?** Teacher-forced token cross-entropy against the reference question,
with padding masked to -100. Development loss selects the adapter. Generated test questions are
evaluated separately, because token loss and generation quality can move differently.

**What does a bootstrap interval mean?** Resample paragraph/topic groups to show sampling variation
on this small dataset. It does not include annotation uncertainty or upstream exposure uncertainty.
An interval crossing zero does not establish a clear directional paired improvement.

**What would you improve next?** Human annotations with two reviewers, stronger multilingual QA
controls, semantic relevance/entailment assessment, article-level and near-duplicate isolation,
larger independently written educational test passages, and domain-specific training data.

## Follow-up questions

**Why is the second verifier not the new default?** Its reference-answer recovery improves,
but false acceptance of constructed negatives also rises. It lacks impossible-question
training, and its uncalibrated span score is not a probability that a question is correct.
Architecture and adaptation data both change. No universal improvement is established.

**Why show all seeds?** They expose training variation. Seed 73 worsens the classroom score.
The seed-mean topic intervals include zero. Reporting only the most favorable seed would
overstate the evidence. The optional app checkpoint is chosen using development loss only.

**Why add an article-isolated run?** Paragraph-disjoint data can still share facts across
articles. The audit found English development titles in training. The sensitivity excludes
23 training paragraphs sharing held-out titles. It also reduces training coverage, so the
result does not isolate the causal effect of isolation. Arabic title/near-duplicate risks
are reported separately. That sensitivity trains English only; the larger native-language study is separate.

**Is the human evaluation complete?** No. The original 54-item decoding booklet and new 36-item
adapter comparison, CSV validation and paired/inter-rater analysis are prepared. Only actual reviewer CSVs can
turn that preparation into human evidence. Constructed controls and explanatory source
inspections do not replace those ratings.

## Larger-training questions

**Why does Arabic train on an English `question:` prefix?** That control prefix is part of
the pinned Arabic checkpoint's decoder target format. It does not translate the question.
The author removes it from decoded output. A development-only audit exposed the mismatch;
the final run trains with the native format and the existing generator strips it for display.

**Did the saved English model see all 5,000 prepared training examples?** No. The whole run
saw 4,800 before stopping at update 600, but its best checkpoint is step 350 and saw 2,800
distinct examples. Later updates are discarded. The metadata/report distinguish preparation,
actual optimization and selected-checkpoint exposure.

**Why exclude source validation examples whose offsets are unusable?** Their passages and
article facts are still held out. Filtering unusable answers does not release those sources
for training. The full-pool audit removes ten additional Arabic training passages by title.

**How is the app checkpoint chosen?** Compare development token loss on identical rows,
including the unadapted model and the previous English adapter. Test generation is evaluated
after selection. Lower loss does not guarantee better human-rated question quality.

## Learning-focused questions and educational training

**Why can a correct question still be unhelpful?** It may ask about an incidental name,
color or time while missing the concept the student should learn. Judge source support,
clear wording and learning value separately. Question length and QA answer recovery do
not measure importance. A simple definition can be useful when it is the lesson's subject.

**Why not just ask more “why” questions?** A why-question needs an explanation in the
passage. Otherwise it introduces an unsupported premise or requires outside knowledge.
Functions, processes, limitations and explicit comparisons are other useful targets.

**What does the new planner change?** It chooses IDs from source-derived answer/evidence
candidates. Code resolves exact source quotes and offsets. T5 writes drafts for those
targets; a local instruction model edits them against the passage and objective. This
separates target selection from wording. Original TF-IDF targets may differ; planned T5
and edited questions share targets, making that comparison easier to interpret.

**Is the editor also an encoder–decoder model?** No. Qwen is decoder-only. The native
T5/mT5/AraT5 generator remains the project's encoder–decoder system and controlled
research baseline. The hybrid adds an inference-time planner/editor. Its final exports
identify the editor and preserve the T5 draft model separately. Do not describe the final
hybrid question as solely written by T5 or claim that Qwen was trained in this project.

**What are the three tuned hyperparameters?** The science grid varies learning rate
{1e-4, 3e-4}, LoRA rank {8, 16} and dropout {.05, .15}: eight combinations. Data order,
seed, 256-update budget and validation protocol are held fixed. All results are retained.
The unadapted step-zero checkpoint can win. One seed does not establish global optimality.

**Why SciQ?** Its human-authored science questions provide an educational training domain.
The preparation uses support passages and uniquely aligned answers, keeps published
split roles, removes duplicate questions and protects all raw held-out support/question
strings before sampling. This is an English-only curriculum study; it is not new Arabic
or French training, and public pretraining exposure remains unknown.

**Does the new training prove that questions are more useful?** No. Validation token loss
selects weights; reserved test generation measures overlap and QA recovery after selection.
Those are different from educational usefulness. Read generated development questions
against their passages and obtain actual human usefulness ratings for stronger evidence.

**Why keep the unsuccessful smaller-editor runs?** They expose real failure modes:
unrelated output after one generic int8 conversion, fragile quote copying, wrong-language
questions and reversed factual roles. Exact spans or valid JSON did not solve semantics.
The stronger-model comparison changes capacity and retains the same planning/editing task.

**How do I inspect the work locally?** Run `python -m scripts.learning_status` for progress.
Use the source-linked comparison pages in `results/learning*` and the science development
comparison. Human review templates remain empty until a person supplies ratings. Avoid
starting a second training process while the first is active.

**Why distinguish an answer's role from its words?** A source can say that a panel
converts light into electrical energy. If the target is electrical energy, asking which
device performs the conversion requests the wrong endpoint of the relation. Exact
source spans and fluent grammar alone do not ensure question–answer alignment. The
answer-role development protocol holds targets and drafts fixed while changing the
editing instructions; all original responses remain available for comparison.

**Does the checker decide what goes into the worksheet?** Questions it flags start
unselected in the app. The teacher can include or exclude each question before export.
This changes the classroom quiz, while stored experiment outputs and reported metrics
remain intact. Read the source even when the checker accepts an item: acceptance is
an imperfect automated decision, not a correctness guarantee.

**What happens when I edit a question?** The app keeps the original generated record,
clears old QA results from the revision and starts its export checkbox unselected.
Exports identify the revision and retain the original wording/model. A score for an
earlier question cannot establish that edited wording has the same support. Empty
revisions cannot be exported. Editing a demo never changes stored benchmark results.

## Human review procedure

Annotate relevance, answerability and fluency independently on a 0–2 scale: 0 = poor, 1 = partial,
2 = good. Use the passage and intended answer, not just the reference wording. Label errors such
as wrong relation, answer leakage, malformed grammar, wrong language, ambiguity or verifier failure.
Use a stratified sample across languages/methods; if two people review, report agreement and
disagreements. Blank templates are pending annotation, never a completed human evaluation.
