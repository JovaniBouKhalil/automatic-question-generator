"""Equal-budget, three-factor educational T5 LoRA search; test data stays closed."""

from __future__ import annotations

import gc
import itertools
import json
import random
import time
from pathlib import Path

from questionforge.data import load_jsonl
from questionforge.experiment import environment_info, fingerprint_file
from questionforge.models import answer_window, configure_torch, format_prompt, model_source
from questionforge.registry import GENERATORS


def main():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        get_linear_schedule_with_warmup,
    )

    root = Path("results/science_training")
    root.mkdir(parents=True, exist_ok=True)
    data = Path("data/processed/science")
    sources = [Path(__file__), Path("docs/SCIENCE_TRAINING_PROTOCOL.md"), Path("scripts/prepare_science_data.py")]
    config = {"seed": 42, "updates": 256, "batch_size": 2, "gradient_accumulation": 4,
              "grid": {"learning_rate": [1e-4, 3e-4], "rank": [8, 16], "dropout": [.05, .15]},
              "base_model": GENERATORS["en"].model_id, "base_revision": GENERATORS["en"].revision,
              "data_sha256": {s: fingerprint_file(data / f"{s}.jsonl") for s in ["train", "dev", "test"]},
              "sources_sha256": {p.as_posix(): fingerprint_file(p) for p in sources},
              "test_opened_for_training_or_selection": False, "precision": "float32", "device": "cpu"}
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Science grid changed; preserve this study and use a new version.")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    for path in sources:
        target = root / "source" / path.relative_to(Path.cwd()) if path.is_absolute() else root / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
    # Only training/development rows are read here. Test bytes are fingerprinted above.
    partitions = {s: load_jsonl(data / f"{s}.jsonl") for s in ["train", "dev"]}
    spec = GENERATORS["en"]
    source, options = model_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **options)
    encoded, ids, excluded = {}, {}, {}
    for split, rows in partitions.items():
        encoded[split], ids[split], excluded[split] = [], [], []
        for row in rows:
            labels = tokenizer(text_target=row["question"], truncation=False)["input_ids"]
            if len(labels) > 64:
                excluded[split].append(row["id"])
                continue
            window, offset, _ = answer_window(row["context"], row["answer"], row["answer_start"], tokenizer,
                                             spec.prompt_style, 256)
            features = tokenizer(format_prompt(window, row["answer"], offset, spec.prompt_style))
            features["labels"] = labels
            encoded[split].append(features)
            ids[split].append(row["id"])
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)
    trials = []
    for lr, rank, dropout in itertools.product([1e-4, 3e-4], [8, 16], [.05, .15]):
        name = f"lr{lr:.0e}_r{rank}_d{dropout:.2f}"
        output = Path("artifacts/science_lora") / name
        receipt = root / f"{name}.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text(encoding="utf-8"))
            if not previous.get("complete"):
                raise ValueError(f"Unfinished trial {name}; preserve and audit its state before retry.")
            trials.append(previous)
            continue
        if output.exists():
            raise ValueError(f"Unfinished adapter {output}; do not overwrite it silently.")
        torch, device = configure_torch(42, "cpu")
        random.seed(42)
        backbone = AutoModelForSeq2SeqLM.from_pretrained(source, trust_remote_code=False, **options)
        model = get_peft_model(backbone, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=rank,
                              lora_alpha=rank * 2, lora_dropout=dropout, target_modules=["q", "v"], bias="none"))
        model.peft_config["default"].base_model_name_or_path = spec.model_id
        model.peft_config["default"].revision = spec.revision
        model.to(device)

        def dev_loss():
            model.eval()
            total, count = 0., 0
            with torch.inference_mode():
                for start in range(0, len(encoded["dev"]), 2):
                    batch = collator([dict(r) for r in encoded["dev"][start:start + 2]])
                    tokens = int((batch["labels"] != -100).sum())
                    total += float(model(**batch, use_cache=False).loss) * tokens
                    count += tokens
            return total / count

        started = time.perf_counter()
        initial = best = dev_loss()
        best_step, history = 0, [{"step": 0, "dev_loss": initial}]
        output.mkdir(parents=True)
        model.save_pretrained(output)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, 16, 256)
        order = list(range(len(encoded["train"])))
        random.Random(42).shuffle(order)
        cursor, epoch, seen = 0, 0, set()
        print(json.dumps({"trial": name, "train": len(encoded["train"]), "dev": len(encoded["dev"]),
                          "initial_dev_loss": initial}), flush=True)
        model.train()
        for step in range(1, 257):
            optimizer.zero_grad()
            train_loss = 0.
            for _ in range(4):
                if cursor >= len(order):
                    epoch, cursor = epoch + 1, 0
                    order = list(range(len(encoded["train"])))
                    random.Random(42 + epoch).shuffle(order)
                indices = order[cursor:cursor + 2]
                cursor += len(indices)
                seen.update(ids["train"][i] for i in indices)
                batch = collator([dict(encoded["train"][i]) for i in indices])
                loss = model(**batch, use_cache=False).loss / 4
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in {name}, step {step}")
                train_loss += float(loss.detach())
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            event = {"step": step, "train_loss": train_loss}
            if step % 128 == 0:
                event["dev_loss"] = dev_loss()
                if event["dev_loss"] < best:
                    best, best_step = event["dev_loss"], step
                    model.save_pretrained(output)
                model.train()
            event["seconds"] = time.perf_counter() - started
            history.append(event)
            if step % 32 == 0:
                print(json.dumps({"trial": name, **event}), flush=True)
        result = {"trial": name, "complete": True, "learning_rate": lr, "rank": rank, "dropout": dropout,
                  "updates": 256, "initial_dev_loss": initial, "best_dev_loss": best, "best_step": best_step,
                  "adapter": output.as_posix(), "history": history, "seconds": time.perf_counter() - started,
                  "unique_training_ids_seen": len(seen), "counts": {s: len(r) for s, r in encoded.items()},
                  "excluded_overlong_targets": excluded, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "adapter_sha256": fingerprint_file(output / "adapter_model.safetensors"),
                  "test_used_for_training_or_selection": False}
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (output / "training_manifest.json").write_text(json.dumps({**config, **{k: v for k, v in result.items() if k != "history"}}, indent=2), encoding="utf-8")
        (output / "README.md").write_text(f"# SciQ educational T5 LoRA trial\n\nBase: {spec.model_id}, pinned revision {spec.revision}. "
                                         f"Academic SciQ adaptation, CC BY-NC 3.0 data. Best step: {best_step}. "
                                         "Selected only by validation loss. See docs/SCIENCE_TRAINING_PROTOCOL.md.\n", encoding="utf-8")
        trials.append(result)
        del model, backbone, optimizer, scheduler
        gc.collect()
    selected = min(trials, key=lambda r: (r["best_dev_loss"], r["best_step"], r["rank"]))
    selection = {"complete": len(trials) == 8, "trials": len(trials), "selected_trial": selected["trial"],
                 "selected_checkpoint": selected["adapter"] if selected["best_step"] else None,
                 "selected_step": selected["best_step"], "selected_dev_loss": selected["best_dev_loss"],
                 "baseline_dev_loss": selected["initial_dev_loss"],
                 "selection_metric": "minimum validation token cross-entropy, including unadapted step zero",
                 "test_used_for_selection": False, "human_usefulness_ratings": "pending", "environment": environment_info()}
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
