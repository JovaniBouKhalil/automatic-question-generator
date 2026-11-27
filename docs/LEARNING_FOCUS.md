# Questions that help students understand

The experimental **Learning-focused hybrid** adds a local fact planner/editor around
the existing native-language T5 generators. It targets explicitly stated concepts,
functions, processes, causes/effects and comparisons. Each question displays the user's
objective or a generic goal assigned from its relation type. Users can also specify exact
target answers. These labels do not establish that the question meets the objective.

**Learning-focused T5** is a second, faster option using explicit relation markers and
an evidence window. It needs only the existing T5 checkpoints. An entered objective is
saved for review in this option; it does not instruct T5 or alter automatic rule selection.
Use manual targets to control the selected facts. The first two development versions
still contain logical errors, especially in French/Arabic, and remain experimental.

The original recall and template modes remain available. Remembering a definition,
important date or central entity can be useful; the aim is to avoid incidental trivia,
not to force every question to start with “why.” Explanations must be in the passage.

## Run it

```powershell
python -m scripts.download_models planner
python -m streamlit run app.py
```

Choose **Learning-focused hybrid (experimental)** in the generation-method menu.
Try the new passages on data splitting, text representation or light and voltage.
Optionally enter what students should understand. Leave target answers blank to let
the planner select facts; manual answers constrain the quiz to your chosen facts.
Inspect the source evidence and expand **Compare with the T5 draft** for each question.

The extra checkpoint is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct),
revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0). The download is about
3.1 GB and stays in ignored `models/`. The editor uses full-precision FP32 on CPU. A
generic int8 conversion failed paired instruction-following controls and was discarded.
The app still runs locally without a paid API. This
mode takes longer than T5 alone and performs inference, not additional fine-tuning.

**Learning-focused hybrid 4B (experimental)** uses the stronger
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Its 8.04 GB of weights are optional and
fully downloaded on the development computer, with all three shard checksums verified.
Prepare it with `python -m scripts.download_quality_model`. The downloader preserves partial
files and checks every complete shard. If the Hugging Face transfer stalls, the alternate
`--source modelscope` uses a pinned Qwen repository whose three published shard checksums
must match the pinned Hugging Face metadata exactly. Runtime parameters alone need roughly
16 GB in FP32, plus inference memory. The original mode remains the default.

The paired precision controls and both fixed nine-passage/language development comparisons
are complete. The main comparisons use FP32; the three control responses alone do not prove
that reduced precision preserves all questions. The queue changes neither application defaults nor human scores.
Completed reports appear in the app's Experiment results tab. The [science adapter
training study](SCIENCE_TRAINING_RESULTS.md) is a separate English-only experiment.

## Completed larger-model development comparison

| Synthetic development set | Cases | Planned T5 drafts | Edited questions | Execution errors |
|---|---:|---:|---:|---:|
| Educational concepts | 9 | 25 | 24 | 0 |
| Classroom scenarios | 9 | 25 | 25 | 0 |

Read the [educational comparison](../results/learning_focus_educational_4b_v1/comparison.html)
and [classroom comparison](../results/learning_focus_dev_4b_v1/comparison.html).
All nine cases in each set have source-linked development inspection notes under
`results/learning_focus/inspection*_4b_v1.json`. English/French educational edits repair
several reversed subjects and incomplete questions. Arabic is more consistent in language
than the smaller model, but retains awkward grammar and conversion-role errors. Whole
statement targets can also be broader than the shortest sufficient answer, especially
in simple classroom scenarios. A supported recall question need not test a deep concept.

These are inspected synthetic development cases, not a fresh held-out test. Conditional
QA averages cannot establish educational value, and coverage differs between versions.
The separate [answer-role refinement protocol](ANSWER_ROLE_PROTOCOL.md) holds these
facts and T5 drafts fixed when testing revised writing instructions. All baseline outputs
remain intact, including rejected questions.

The planner chooses IDs from source-derived relation spans and complete statements.
Code resolves the exact answers and evidence; generated answer paraphrases are not used.
T5 generates drafts for those targets;
the editor revises the questions using the passage and goals. Invalid spans, malformed
responses, duplicate targets/questions, answer leakage and some language/format errors
are rejected. Fewer questions may be returned rather than filling the quiz with weaker
items. These checks do not prove that a question's relations or learning goal are valid.

## Evaluate usefulness separately from answer recovery

The study compares original TF-IDF + T5, planned-target T5, and edited questions. The
last two share answer targets; the original selector may choose different facts.
The protocol and exact source fingerprints are saved with each run. See
[the recorded protocol](LEARNING_FOCUS_PROTOCOL.md).

```powershell
python -m scripts.run_learning_focus_study --output outputs/learning_focus_classroom_reproduction
python -m scripts.run_learning_focus_study --source educational --output outputs/learning_focus_educational_reproduction
python -m scripts.run_learning_t5_study --output outputs/learning_t5_reproduction
python -m scripts.validate_learning_focus
```

Each completed study includes raw model responses, T5 drafts, edited questions, rejection
reasons, runtimes, QA diagnostics, `comparison.html` and an empty review CSV template.
Existing compatible cases are reused on resume. Different source/configuration requires
a new output directory; do not overwrite prior runs or silently exclude failed cases.
The template is not blinded because its purpose is development inspection. It is not
the original blinded human evaluation. No human score is inferred from QA diagnostics.

Suggested human review of **each question with its passage**:

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Factual support | Unsupported premise, wrong relation or answer | Partly supported or ambiguous | Passage clearly supports the question and answer |
| Clarity | Unclear or malformed | Understandable with wording problems | Clear, specific and self-contained |
| Learning value | Incidental trivia, redundant copying, or no clear purpose | Relevant fact but limited understanding | Tests an important concept, function, process or relationship for the stated objective |

Review the quiz as a set too: repeated wording or several questions about the same fact
does not give broad coverage. A sophisticated-sounding question is not automatically
useful. Report these ratings only when a person actually supplies them; the generated
learning goals and question-type labels are proposals, not measured educational outcomes.

The new educational examples are original synthetic development/demo passages. They
are not independently authored held-out tests, and no human validation is claimed.
The light/voltage example expands a previously inspected scenario.
