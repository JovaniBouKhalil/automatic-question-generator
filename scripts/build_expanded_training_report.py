"""Write the training findings from measured artifacts; never fill pending human scores."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "results/expanded_training/summary.json").read_text(encoding="utf-8"))
    data = json.loads((root / "results/expanded_training/data_manifest.json").read_text(encoding="utf-8"))
    lines = ["# Expanded-data training findings", "",
             "This follow-up trains separate native-language LoRA adapters from the original pinned, already",
             "task-tuned checkpoints. It preserves all original experiments and the three-seed study.",
             "Selection uses development token loss; held-out generation is evaluated after selection.", "",
             "## Actual data use and checkpoint selection", "",
             "Prepared rows, rows seen during the whole run, and rows seen by the saved checkpoint differ.",
             "Early stopping can discard later optimizer updates; they must not be attributed to saved weights.", "",
             "| Language | Prepared train / dev | Run updates | Saved step | Distinct rows: full run / saved checkpoint | Dev loss: initial → best |",
             "|---|---:|---:|---:|---:|---:|"]
    for lang, info in report["languages"].items():
        lines.append(f"| {lang.upper()} | {info['prepared_training_examples']:,} / {info['development_examples']} | "
                     f"{info['steps']} | {info['best_step']} | {info['unique_training_ids_seen']:,} / {info['saved_checkpoint_unique_examples_seen']:,} | "
                     f"{info['initial_dev_loss']:.4f} → {info['best_dev_loss']:.4f} |")
    lines += ["", "English is also compared with the previous seed-42 adapter on the same 200 development rows.",
              f"That adapter's loss is {report['languages']['en']['previous_english_adapter_dev_loss']:.4f}. "
              "The app uses the lowest development-loss candidate, including the unadapted model.",
              "Losses use each checkpoint's native tokenizer/target format and are not comparable across languages.", "",
              "## Held-out generation", "",
              "Differences below are percentage points against the same unadapted beam-4 generator.",
              "These repeat the previously inspected original test sets; they are not newly untouched tests.",
              "QA F1 is target-answer recovery by an imperfect independent verifier, not question accuracy.", "",
              "| Language | Test subset | Inputs | QA F1: base → expanded | Difference [95% topic interval] | ROUGE-L: base → expanded |",
              "|---|---|---:|---:|---:|---:|"]
    for row in report["evaluations"]:
        low, high = row["paired_topic_ci95_pp"]
        lines.append(f"| {row['language'].upper()} | {row['dataset']} | {row['n_unique_inputs']} | "
                     f"{row['base_qa_f1']:.2f} → {row['adapted_qa_f1']:.2f} | "
                     f"{row['qa_f1_delta_pp']:+.2f} [{low:.2f}, {high:.2f}] | "
                     f"{row['base_rouge_l']:.2f} → {row['adapted_rouge_l']:.2f} |")
    lines += ["", "All intervals resample whole paragraph/topic groups. One training seed per language cannot",
              "establish broad stability. Data volume, development splits and training duration change together.",
              "Neither lower loss nor higher overlap/QA agreement establishes fluency or factual support.", "",
              "English comparison with the previous selected adapter:", ""]
    for row in report["evaluations"]:
        if row["language"] == "en":
            low, high = row["expanded_minus_previous_ci95_pp"]
            lines.append(f"- {row['dataset']}: previous QA F1 {row['previous_adapter_qa_f1']:.2f}; "
                         f"expanded-minus-previous {row['expanded_minus_previous_qa_f1_pp']:+.2f} [{low:.2f}, {high:.2f}].")
    lines += ["", "These test comparisons do not override development-only app selection.", "",
              "## Data provenance and the Arabic corrections", "",
              "English and French use the original pinned SQuAD and NewsQuAD-fr snapshots.",
              "Arabic adds [Google Research TyDiQA-GoldP](https://huggingface.co/datasets/google-research-datasets/tydiqa),",
              f"revision `{data['additional_source']['revision']}`, to the original ARCD source.",
              "TyDiQA uses native human questions rather than translations; its metadata gives Apache-2.0,",
              "and underlying Wikipedia attribution still applies. Other source permissions are in DATA_AND_MODELS.md.", ""]
    files = data["additional_source"]["files"]
    for split, label in [("train", "training"), ("test", "published validation")]:
        stats = files[split]
        rejected = stats["excluded"].get("invalid_record_or_span", 0)
        lines.append(f"- TyDiQA {label}: {stats['valid_arabic_records']:,} Arabic records pass exact answer-span checks; "
                     f"{rejected:,} fail. Offsets are not guessed or silently realigned.")
    lines += ["", "All published validation source paragraphs remain reserved, including those with unusable answer",
              "annotations. A final full-pool title audit excluded 10 additional Arabic training passages, leaving",
              "1,990 prepared training rows. That preliminary attempt was preserved outside the submission.",
              "The clean final run uses the corrected inputs from step zero.", "",
              "The Arabic decoder expects `question: ` in its target format. A development-only format audit",
              "detected the mismatch, and the final Arabic run restarts with the native prefix. The existing",
              "generation code strips it for display. See ARABIC_TARGET_FORMAT.md for the measured diagnostic;",
              "the abandoned plain-target attempt is not reported as a completed experiment.", "",
              "Article grouping uses known titles; French has no reliable titles, so it uses paragraph groups.",
              "These safeguards do not prove absence of semantic near-duplicates or upstream checkpoint exposure.", "",
              "## Reproduction and review", "", "```powershell",
              "python -m scripts.run_expanded_study", "python -m scripts.expanded_training_status", "```", "",
              "Install PEFT (now a runtime dependency), prepare the original data/models, and use Python 3.11.",
              "The runner preserves completed compatible adapters and resumes unfinished optimizer state.",
              "Each submitted adapter contains its exact training source and metadata; full weights, full datasets,",
              "optimizer states and aborted attempts remain excluded. Prepared JSONL bytes are deterministic across platforms.", "",
              "The app offers development-selected native adapters through the Use adapted generator toggle. Original defaults are retained; held-out gains are mixed.",
              "The new 36-item offline booklet in `results/expanded_human_evaluation` hides original/expanded model identity.",
              "Its relevance, answerability and fluency ratings remain pending until people supply CSVs. The original",
              "54-item baseline/decoding booklet is retained separately. No human score has been fabricated.", "",
              "`results/expanded_training/before_after_cases.json` revisits all ten prior beam-search qualitative",
              "cases with the new outputs and source passages. Original error labels describe the old questions",
              "only; changed wording or higher QA agreement is not automatically labeled a factual repair.", ""]
    (root / "docs/EXPANDED_TRAINING_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote measured training findings and provenance: docs/EXPANDED_TRAINING_RESULTS.md")


if __name__ == "__main__":
    main()
