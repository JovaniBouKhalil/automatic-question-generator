# Answer-role editing: results and remaining failures

The optional 4B learning mode retains source-ID planning and native T5 drafts, then
guides the editor with the target answer's role: output, purpose, limitation, criterion
or explicit comparison. Qwen is used for inference; the separate English SciQ adapter
is the model-training experiment. Original and smaller-model results remain available.

## Recipe selection

Six purposive development cases compare role-hinted editing and draft-free writing
(12 responses). Both repair the inspected Arabic validation-purpose and English
weight-contrast mismatches. Keeping T5 drafts preserves a more specific English
voltage-effect question; draft removal does not repair Arabic conversion grammar.

An extra Arabic grammatical-frame probe on three cases still asks for the wrong
conversion role. Its additional instruction is discarded. The recorded selection
uses source/target inspection, not QA-score maximization. All failed responses remain
in the artifacts. See [the protocol](ANSWER_ROLE_PROTOCOL.md) and
[selection record](../results/learning_focus/refinement_selection.json).

## Fixed-target re-edit

Every answer, offset, passage, objective and T5 draft stays fixed. The model is the
same pinned Qwen3-4B-Instruct-2507 checkpoint, greedy FP32 CPU. Coverage counts below
are not question accuracy, usefulness ratings or independent samples of people.

| Synthetic development set | Cases | Fixed targets | Original 4B returned | Role-guided returned | Execution errors |
|---|---:|---:|---:|---:|---:|
| Educational | 9 | 25 | 24 | 24 | 0 |
| Classroom | 9 | 25 | 25 | 25 | 0 |

| Development set | Original QA answer F1 | Role-guided QA answer F1 |
|---|---:|---:|
| Educational | 71.83 | 73.52 |
| Classroom | 48.80 | 45.94 |

These 0–100 diagnostic means condition on returned questions. They rise in the
educational set and fall in the classroom set. Neither direction establishes a
change in correctness or learning value; source inspection also finds wording regressions.

Read every saved question in the [educational comparison](../results/learning_role_educational_v1/comparison.html)
and [classroom comparison](../results/learning_role_classroom_v1/comparison.html).
Conditional QA recovery remains a diagnostic; it does not select the recipe or
establish that a returned question is supported.

## Source-linked observations

Source-linked development observations, not human ratings, a representative error estimate or proof of educational benefit.

- [learning:light_and_voltage:ar](../results/learning_role_educational_v1/learning_light_and_voltage_ar.json): The screen-purpose question is natural and supported. The conversion still asks for the input and leaks the output answer, so that item is rejected. No successful conversion repair is claimed.
- [learning:light_and_voltage:en](../results/learning_role_educational_v1/learning_light_and_voltage_en.json): The three questions retain supported screen purpose, conversion output and measured voltage effect. They are unchanged from the saved 4B baseline; retaining a correct question is appropriate.
- [learning:light_and_voltage:fr](../results/learning_role_educational_v1/learning_light_and_voltage_fr.json): The two questions ask the stated screen purpose and conversion output. The new passive conversion wording remains supported. Planning supplies only two targets, so no third item is invented.
- [learning:text_representation:ar](../results/learning_role_educational_v1/learning_text_representation_ar.json): The criterion, weighting comparison and benefit questions match the source. The benefit question repairs the earlier awkward gender agreement. Planning still omits the bag-of-words word-order limitation named in the objective, and the contrast target is broader than a minimal answer.
- [learning:text_representation:en](../results/learning_role_educational_v1/learning_text_representation_en.json): The weight item now asks how the two word groups differ, matching the contrast target rather than asking why. The bag-of-words limitation and document-frequency adjustment remain supported. The objective's explanation for lowering common-word weight is not directly tested by the replacement comparison.
- [learning:text_representation:fr](../results/learning_role_educational_v1/learning_text_representation_fr.json): The missing-order, TF-IDF purpose and weighting-comparison questions remain supported. The contrast target is long, but the question requests that contrast rather than an unstated explanation.
- [learning:training_and_evaluation:ar](../results/learning_role_educational_v1/learning_training_and_evaluation_ar.json): The test-isolation and article-grouping questions preserve the stated prevention reasons. The validation item now naturally asks for the purpose of using validation data, repairing the earlier data-as-agent wording.
- [learning:training_and_evaluation:en](../results/learning_role_educational_v1/learning_training_and_evaluation_en.json): The training-use, validation-use and test-isolation questions remain supported and unchanged from the baseline. Their good T5 drafts are retained rather than unnecessarily rewritten.
- [learning:training_and_evaluation:fr](../results/learning_role_educational_v1/learning_training_and_evaluation_fr.json): Training and validation purposes and test isolation remain supported. The concise use questions still omit the initial accent in A quoi; no complete grammatical polish is claimed.
- [novel:book_club:ar](../results/learning_role_classroom_v1/novel_book_club_ar.json): Place and reading material remain supported. The leader question mixes masculine and feminine verb agreement, a wording regression rather than a repair. Whole-statement targets remain broader than the minimal answers. The simple passage offers limited conceptual material.
- [novel:book_club:en](../results/learning_role_classroom_v1/novel_book_club_en.json): Meeting place, reading material and recorded suggestions are supported. The record question now names the discussion; the other targets remain broader-than-minimal statements. These are relevant recall items rather than evidence of deeper conceptual understanding.
- [novel:book_club:fr](../results/learning_role_classroom_v1/novel_book_club_fr.json): Meeting place, the discussion leader's role and reading material remain supported and readable. Targets are complete statements, broader than minimal answers; role guidance supplies no clear additional learning-value evidence here.
- [novel:school_bus:ar](../results/learning_role_classroom_v1/novel_school_bus_ar.json): The cause item still substitutes the unrelated word for astonishment where the journey taking time is intended. This remains malformed despite a correct source reason. The driver question is supported with a broad statement target.
- [novel:school_bus:en](../results/learning_role_classroom_v1/novel_school_bus_en.json): The explicit road-repair why-question is supported. The destination item changes the clearer baseline Where question to the awkward What did Omar drive the bus to wording. This is a wording regression, not a universal benefit of role guidance.
- [novel:school_bus:fr](../results/learning_role_classroom_v1/novel_school_bus_fr.json): Departure time and driver remain supported. The duration-and-cause question is readable but asks two things because its target is a two-fact statement. It does not isolate a single relation as cleanly as a focused target would.
- [novel:seedlings:ar](../results/learning_role_classroom_v1/novel_seedlings_ar.json): Count and watering duration remain supported. The planting item replaces the earlier repeated wording with a grammatical but broad request for planting details. Whole-statement targets remain longer than minimal answers.
- [novel:seedlings:en](../results/learning_role_classroom_v1/novel_seedlings_en.json): The count and planting-action questions are supported. The watering item now asks frequency and duration, matching more of its statement target but becoming a two-part question. The planting target remains broad.
- [novel:seedlings:fr](../results/learning_role_classroom_v1/novel_seedlings_fr.json): Count and watering schedule are supported; the schedule question now asks both timing and duration. The planting item still asks for the relatively incidental classroom situation and copies much of its broad statement target. Role guidance does not solve target granularity.

## Additional topic audit

After freezing the recipe, the full automatic path runs on three additional topics
in English, French and Arabic: physical separation, search limitations and greenhouse
conditions. The synthetic passages were authored for this audit; they have no human
reference questions or ratings. This is development transfer inspection, not an
independently authored final benchmark. No prompt is retuned on these outputs.

Recorded 9 passage/language pairs; 0 execution-error cases. The planner supplies 25 targets/drafts; the editor returns 25 questions.

[Inspect the complete additional-topic comparison](../results/learning_challenge_v1/comparison.html).

- [learning:greenhouse_conditions:ar](../results/learning_challenge_v1/learning_greenhouse_conditions_ar.json): The three Arabic questions are readable and supported: both temperature thresholds, unchanged window position between them, and manual mode stopping automatic movement while the sensor continues recording. The first/manual targets are broad statements. The shading tradeoff named in the objective is not selected, so full objective coverage is not established.
- [learning:physical_separation:ar](../results/learning_challenge_v1/learning_physical_separation_ar.json): The first item identifies salt remaining in water and the evaporation item uses the stated heating procedure, though both have awkward wording. The second item asks what remained with it, leaving an ambiguous filter-versus-water referent. Two selected statements repeat the filtration limitation while the screening step is omitted. Unique evidence does not ensure conceptual diversity.
- [learning:word_search_limitations:ar](../results/learning_challenge_v1/learning_word_search_limitations_ar.json): The stemming-function and over-aggressive-merging risk questions are supported. The librarian question adds a correction purpose that the source does not explicitly state; the source only says she reviews errors. Planning omits the inverted-index function and exact-match synonym limitation, concentrating all three items on stemming and its review.
- [learning:greenhouse_conditions:en](../results/learning_challenge_v1/learning_greenhouse_conditions_en.json): The questions preserve the unchanged state between thresholds, the heating-versus-light tradeoff, and manual mode stopping window movement while sensor recording continues. These are supported relationships rather than incidental facts. The opening/closing behavior itself is not directly tested; threshold values are given in the first question.
- [learning:physical_separation:en](../results/learning_challenge_v1/learning_physical_separation_en.json): The filter-limitation and heating-method questions are source-supported, although verbose. The sand item adds a wrong premise that sand passed through the paper filter, whereas the source says the paper retained it. Two items overlap on filtration and omit the screening step. Valid spans and QA recovery do not catch this relation reliably.
- [learning:word_search_limitations:en](../results/learning_challenge_v1/learning_word_search_limitations_en.json): All three questions are clear and supported: the synonym limitation of exact matching, stemming's function and the risk of merging different meanings. These target useful search concepts. Whole-statement answers include extra context, and the inverted-index function named in the objective is not selected.
- [learning:greenhouse_conditions:fr](../results/learning_challenge_v1/learning_greenhouse_conditions_fr.json): The opening/closing thresholds, maintained window position and manual-mode behavior are supported and readable. The first question asks both threshold actions, matching its broad statement target. Shading's tradeoff is omitted from the selected quiz.
- [learning:physical_separation:fr](../results/learning_challenge_v1/learning_physical_separation_fr.json): The single returned question correctly asks what the filter did not remove. The planner deliberately selects only one candidate, with no invalid-ID rejections, despite several usable source statements. Screening and evaporation are omitted, so requested count and objective coverage are substantially incomplete.
- [learning:word_search_limitations:fr](../results/learning_challenge_v1/learning_word_search_limitations_fr.json): All three questions are supported: exact matching's benefit and synonym limitation, the stemming operation and the risk of merging different meanings. The first is a two-part question matching its broad statement target. The inverted-index function in the objective is omitted.

## Practical use and limits

Choose **Learning-focused hybrid 4B (experimental)** in the app. The default native
T5 mode and optional science adapter remain available. The stronger editor can take
several minutes per quiz on CPU. Inspect source support and wording, especially Arabic.

QA-flagged questions start unselected for export. The teacher can include or exclude
individual questions and revise wording before export. A revision clears the original
QA scores, starts unselected, and retains the generated question and model provenance.
Exports include the passage and optional objective. Review choices do not remove
benchmark records or change reported scores. Exact-span, leakage and
format checks cannot prove a question's premise. Human learning-value ratings remain
pending; no percentage of correct or useful questions is claimed.

Reproduction commands and the three-criterion review rubric are in
[the learning-mode guide](LEARNING_FOCUS.md). `scripts.validate_learning_focus` checks
saved source snapshots, fixed target/draft identity and counts;
`scripts.validate_role_refinement` checks the pilot, selection and replay inputs.
