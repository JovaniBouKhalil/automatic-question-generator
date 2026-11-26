# Data, models and experimental protocol

## Sources and permissions

| Dataset | Language/domain | Target | Hub license metadata |
|---|---|---|---|
| [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) | English / Wikipedia | Human question + extractive answer span | CC-BY-SA-4.0 |
| [NewsQuAD-fr](https://huggingface.co/datasets/lincoln/newsquadfr) | French / news | Human question + extractive answer span | CC-BY-NC-SA-4.0; noncommercial academic use |
| [ARCD](https://huggingface.co/datasets/hsseinmz/arcd) | Arabic / Wikipedia | Human question + extractive answer span | MIT in Hub metadata; underlying source attribution also applies |

Pinned revisions and file paths are in `questionforge/data.py`. No dataset script or remote
repository code is executed. QA datasets are repurposed for question generation: the model
receives **context + target answer**, and predicts the reference question. Full source counts,
rejections and selected counts are measured in `data/processed/manifest.json`, copied into
`results/data/manifest.json` for submission. Source URLs above document provenance and permissions.

## Partitions and preprocessing

1. Validate nonempty questions and an exact answer at its supplied character offset.
2. Exclude invalid spans and duplicate `(normalized paragraph, question, answer)` records.
3. Reserve the published evaluation pool for our test partition. For NewsQuAD-fr this is
   `valid.json`; development data comes from source training, not this evaluation pool.
4. Remove training rows whose normalized paragraph appears in the evaluation pool.
5. Split remaining paragraph groups into train/development with seed 42 and a 15% development fraction.
6. Deterministically sample one random question per paragraph: up to 600 training, 15 development
   and 40 test paragraphs per language. Actual counts may be smaller and are recorded.
7. Assert that no paragraph group crosses partitions.

Duplicate grouping normalizes Unicode compatibility forms, case and whitespace. Model input
retains original text, accents, Arabic letters and character offsets. No vocabulary, index or
adapter is fitted on development/test data. Paragraph grouping does not guarantee article-level
or near-duplicate isolation: related paragraphs may still share facts.

`data/authoring/scenarios.json` adds 15 original parallel classroom scenarios, two questions per
language. Three topics form development (18 questions), twelve form test (72 questions). All
translations of one scenario share a group/split. Source JSON and generated JSONL are committed
before inference. These AI-assisted scenarios describe fictional classroom events; support means
support within the passage. They need independent human linguistic review and are too small and
simple to replace public human-annotated benchmarks. They are not used to train the adapter.

## Pretrained models

| Purpose | Checkpoint | Expected input | License |
|---|---|---|---|
| English QG | [lmqg/t5-small-squad-qg](https://huggingface.co/lmqg/t5-small-squad-qg) | `generate question: ... <hl> answer <hl> ...` | CC-BY-4.0 |
| French QG | [lmqg/mt5-small-frquad-qg](https://huggingface.co/lmqg/mt5-small-frquad-qg) | `... <hl> answer <hl> ...` | CC-BY-4.0 |
| Arabic QG | [Mihakram/AraT5-base-question-generation](https://huggingface.co/Mihakram/AraT5-base-question-generation) | `context: passage answer: answer </s>` | Not clearly specified |
| Independent QA | [deepset/xlm-roberta-base-squad2-distilled](https://huggingface.co/deepset/xlm-roberta-base-squad2-distilled) | Question/passage pair | MIT |

Revisions are pinned in `questionforge/registry.py`. English T5-small has about 60M parameters;
mT5-small has about 300M because of its large multilingual vocabulary. AraT5 differs in size,
vocabulary and training data. This is a comparison of language-specific systems, not a controlled
architecture comparison across languages. Method references:
[T5](https://arxiv.org/abs/1910.10683), [mT5](https://arxiv.org/abs/2010.11934),
[AraT5](https://arxiv.org/abs/2109.12068), [lmqg](https://arxiv.org/abs/2305.17416).

Public checkpoint cards disclose benchmark training, but complete example-level provenance is
unavailable, particularly for Arabic. Benchmark results carry **upstream exposure risk**. Our
audits prevent new leakage; they do not prove that all upstream training data was disjoint.

## Fixed experiments

All configurations receive the same passages and exact target answers. The untrained baseline
replaces the target in its evidence sentence with a language-specific WH phrase. It is cheap
and grounded but does not reliably form natural question word order.

Neural configurations use **greedy (1 beam)** and **4-beam search**, sampling off, maximum input
384 tokens, maximum new output 64 tokens, repeated-trigram constraint 3, seed 42. These settings
are fixed before final test runs. Beam search is not assumed to be better. Actual prompted token
counts determine answer-centered cropping. The answer is preserved; oversized targets raise an
error. Cropping can cut neighboring words/sentences and lose context; crop frequency is reported.
CPU float32 is the default. Cold model-load time is separated from generation time.

The **English LoRA** extension follows [Hu et al., LoRA](https://arxiv.org/abs/2106.09685),
freezing T5 and adapting attention query/value projections: rank 8,
alpha 16, dropout 0.05, AdamW, learning rate 0.0003, weight decay 0.01, batch 2, four microbatches
per update (effective batch 8), 40 optimizer updates, gradient clipping 1.0 and linear scheduling.
Training input/target budgets are 256/64 tokens. Checkpoint selection minimizes development
target-token cross-entropy, including step 0. The adapter, loss history, parameter counts, hashes,
runtime and environment are saved. Lower loss need not improve generation; test generation is
evaluated separately. Small adapter weights may be submitted; backbone weights are excluded.

## Metrics and controls

- **chrF++ / corpus BLEU / ROUGE-L** measure agreement with one reference, not factuality.
  BLEU uses `intl` tokenization; signatures/versions are saved. ROUGE uses Unicode token LCS
  without English-only stemming or ASCII filtering.
- **Round-trip QA token F1 / exact match** compare an independent extracted answer with the
  target. English articles, Arabic vowel marks and elongation are removed for answer matching;
  French accents are retained. This is documented custom normalization, not a claim of exact
  compatibility with every official benchmark evaluator.
- **Lexical relevance** is content-token cosine similarity between question and evidence;
  it is not semantic entailment. **Answer leakage** detects token-bounded target text in a question.
- **Acceptance/coverage** uses fixed F1 ≥ 0.5, nonempty QA span score ≥ 0.1, a nonempty question,
  and no detected answer leakage. It is a review heuristic, not accuracy or calibrated confidence.
- **Reference controls** ask reference questions; **unrelated controls** ask about a red submarine's
  serial number. These expose judge weakness and false agreement.
- `scripts/analyze_controls.py` measures reference/unrelated answer rates and span-score AUROC
  on these constructed controls. This diagnoses control discrimination, not human-rated question
  accuracy or confidence calibration; synthetic unrelated negatives are relatively easy.
- **95% group bootstrap intervals** resample whole paragraphs or scenario topics 1,000 times
  (seed 42). Paired differences use matching record IDs. Small samples limit generalization.
- Errors stay in denominators as explicit failure rows with zero scores. Generation, verification
  and cold-load times are recorded separately.

The verifier has multilingual pretraining but **English SQuAD2 QA fine-tuning**. A low French or
Arabic score can be a judge failure. Neither high overlap nor QA agreement proves fluency,
educational value or factual correctness. `human_review.csv` is a blank annotation template;
human ratings are not invented. Different datasets/domains and checkpoints prevent causal
claims that one language is intrinsically easier.

## Redistribution

Full downloaded datasets and backbone weights are ignored locally. Prediction artifacts contain
small source excerpts for auditability; the named dataset licenses and source URLs apply to them.
French excerpts are noncommercial/share-alike. No blanket permissive license is assigned to
third-party data/weights. Arabic checkpoint weights are not redistributed. Course notebooks and
the instructor's PDF were reviewed locally and are not republished.
