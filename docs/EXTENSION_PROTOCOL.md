# Extension protocol, fixed before the new runs

Starting snapshot: `a768944`. The existing public and classroom test outputs have already
been inspected. These extensions are follow-up robustness analyses, not a newly untouched
test benchmark. Every planned seed, model, threshold and comparison is specified here
before collecting the new outputs. Negative results remain in the submission.

## Training stability

- Preserve the original 40-update adapter and all original predictions.
- Train English T5 LoRA for 160 optimizer updates with seeds **13, 42, 73**. Keep rank 8,
  learning rate 0.0003, effective batch 8 and the original training/development partitions.
- Within each run, select the lowest development token loss, including step zero.
- Compare every selected checkpoint against the frozen unadapted beam-4 generator on
  identical public and classroom test IDs. Report all three seeds, not only the best.
- Select an optional application adapter using development loss alone. Test scores do not
  select seeds, learning rates or checkpoints. Report seed dispersion separately from
  paragraph/topic bootstrap uncertainty; three runs do not establish broad robustness.

## Verifier ablation

- Compare the existing English-QA-finetuned XLM-R verifier with
  `mrm8488/bert-multi-cased-finetuned-xquadv1`, revision
  `610161fac355eac3e06ac622d376e178a948a5d0`.
- The alternative has multilingual QA adaptation, including Arabic; French is zero-shot.
  Its card describes augmented XQuAD data without complete example provenance. XQuAD is
  derived from SQuAD, so public benchmark overlap is a serious limitation. Use the original
  synthetic classroom development/test partitions for this controlled verifier study.
- XQuAD-v1 has no unanswerable examples. Do not pretend this model has a trained abstention
  head: disable its impossible-answer heuristic and measure false target agreement directly.
- Fixed acceptance rules for both: target token F1 >= 0.5, span score >= 0.1, no answer leak,
  and a nonempty question. Span scores are not calibrated and are not comparable confidence
  probabilities between model families.
- Conditions: correct reference; unrelated question; unsupported location qualifier; and
  an otherwise valid reference paired with the other answer in the same classroom topic.
  The last condition tests the intended-answer relation, not question unanswerability.
- Report reference F1/EM, positive acceptance, false acceptance per negative condition,
  balanced accuracy on these constructed controls, and paired topic bootstrap intervals.
  There is no threshold search. These controls are not human labels of generated quality.
- Preserve the original default verifier. Expose the alternative explicitly as an
  experimental option; a test score must not silently change application behavior.

## Blinded human evaluation

- Sample three public paragraphs and three classroom topics per language, seed 42. Include
  baseline, greedy and beam-4 for the same selected inputs: 54 generated questions total.
- Shuffle items and hide method, checkpoint, reference question, automatic scores and
  acceptance decisions. Show source passage, generated question and intended answer.
- Independently rate relevance, answerability and fluency 0/1/2 using anchored instructions.
  Preserve pseudonymous reviewer IDs, detect duplicates and incomplete/invalid scores.
- Report actual completion counts; report method means only on complete paired blocks.
  If two reviewers participate, compute ordinal weighted Cohen kappa on shared items.
- Keep ratings empty until a person supplies them. A prepared annotation interface is not
  a completed human study. The final presentation must distinguish observed results from
  pending annotations.

## Article-isolation sensitivity, added before its training run

The completed overlap audit found that 11/15 English development paragraphs share an
article title with the training subset. No English test title overlaps the training
subset. Paragraph-disjoint development can still be optimistic when related facts recur.
For one additional **seed-42, 160-update** run, remove every English training paragraph
whose nonempty normalized title appears in development or test. Keep the exact same
development/test examples and generation settings. Save excluded IDs and data hashes.
This is a follow-up sensitivity run with less training data, not a controlled claim that
article isolation itself improves model quality. Report its result even if it is worse;
do not use its test scores to select the application adapter.
