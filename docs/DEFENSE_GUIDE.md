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
experiments as a three-hyperparameter search or the application as a full RAG system.

| Course material | Connection |
|---|---|
| Weeks 1–3: vectorization, entities and exploratory work | Inspect lengths and source spans; automatic answer selection ranks TF-IDF phrases and simple name/number heuristics. These heuristics are not a trained NER system. |
| Week 4: BoW and TF-IDF | Explain term weighting, stopwords, lexical cosine and limits of word overlap. |
| Week 5: baseline and mistake inspection | A transparent baseline and per-example review precede claims about advanced models. |
| Week 6: RNN/LSTM and overfitting | Contrast recurrent sequential state with attention; development selection avoids test tuning. |
| Transformers notebook | Token representations, attention, residuals, normalization and feed-forward blocks appear in pretrained T5. |
| Weeks 7–9: pretrained model families and fine-tuning | The supplied PDF explicitly includes T5/BART encoder–decoder models (p. 11). We reuse matching pretrained tokenizers and task checkpoints. The class fine-tunes BERT in Keras; our generator adapts T5-family models with PyTorch/LoRA. |
| GPT-2 exercise | Autoregressive next-token decoding and greedy/beam choices; T5 additionally conditions through encoder cross-attention. |
| Week 10: hyperparameters and evaluation | Shared splits, development selection, early stopping and error inspection follow the slides. Greedy/beam and adapter comparisons are implemented; systematic learning-rate/rank/dropout search is not. Classification F1 differs from our answer-token F1. |
| Weeks 12–13: modern NLP systems and RAG | Hugging Face inference and a generator/checker workflow follow the system-level discussion. Supplying a passage and cropping around an answer is not the slides' embedding/vector-retrieval RAG pipeline. |

## Questions you should be able to answer

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

**What is TF-IDF used for?** Suggesting candidate answer phrases in the app. It does not learn the
question generator. Controlled benchmark scores use provided answers, so they do not measure
automatic answer-selection quality. Do not describe those scores as fully end-to-end accuracy.

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

## Human review procedure

Annotate relevance, answerability and fluency independently on a 0–2 scale: 0 = poor, 1 = partial,
2 = good. Use the passage and intended answer, not just the reference wording. Label errors such
as wrong relation, answer leakage, malformed grammar, wrong language, ambiguity or verifier failure.
Use a stratified sample across languages/methods; if two people review, report agreement and
disagreements. Blank templates are pending annotation, never a completed human evaluation.
