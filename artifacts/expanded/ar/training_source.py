"""Larger native-language LoRA runs with exact resume state and development-only selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

from questionforge.data import assert_disjoint, load_jsonl
from questionforge.experiment import environment_info, fingerprint_file
from questionforge.models import answer_window, configure_torch, format_prompt, load_generator, model_source
from questionforge.registry import GENERATORS

BUDGETS = {"en": (800, 50), "fr": (256, 32), "ar": (512, 50)}


def training_order(size, seed, epoch):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=list(BUDGETS))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        get_linear_schedule_with_warmup,
    )

    language, seed = args.language, 42
    maximum, evaluate_every = BUDGETS[language]
    data_dir = Path("data/processed/expanded") / language
    output = Path("artifacts/expanded") / language
    state_dir = Path("outputs/expanded") / language
    state_path = state_dir / "trainer_state.pt"
    manifest_path = output / "training_manifest.json"
    if manifest_path.exists():
        completed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if completed.get("training_complete"):
            expected = {split: fingerprint_file(data_dir / f"{split}.jsonl") for split in ["train", "dev", "test"]}
            if expected != completed["data_sha256"]:
                raise ValueError("Completed adapter has different data; preserve it and choose another run.")
            print(f"{language}: training already complete with matching data; preserved.", flush=True)
            return
    if output.exists() and (output / "adapter_config.json").exists() and not args.resume:
        raise ValueError("Unfinished adapter exists. Use --resume; never overwrite a checkpoint silently.")
    torch, device = configure_torch(seed, args.device)
    random.seed(seed)
    partitions = {split: load_jsonl(data_dir / f"{split}.jsonl") for split in ["train", "dev", "test"]}
    assert_disjoint(partitions)
    if not partitions["train"] or not partitions["dev"]:
        raise ValueError("Nonempty training/development partitions are required.")
    if any(r["language"] != language for rows in partitions.values() for r in rows):
        raise ValueError("Native-language adapters must not silently mix languages.")
    spec = GENERATORS[language]
    source, options = model_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **options)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(source, trust_remote_code=False, **options)
    model = get_peft_model(backbone, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
                           lora_dropout=.05, target_modules=["q", "v"], bias="none"))
    model.peft_config["default"].base_model_name_or_path = spec.model_id
    model.peft_config["default"].revision = spec.revision
    model.to(device)
    target_prefix = "question: " if language == "ar" else ""

    def encode(rows):
        encoded, retained, rejected = [], [], []
        for row in rows:
            labels = tokenizer(text_target=target_prefix + row["question"], truncation=False)["input_ids"]
            if len(labels) > 64:
                rejected.append(row["id"])
                continue
            window, offset, _ = answer_window(row["context"], row["answer"], row["answer_start"],
                                              tokenizer, spec.prompt_style, 256)
            features = tokenizer(format_prompt(window, row["answer"], offset, spec.prompt_style))
            features["labels"] = labels
            encoded.append(features)
            retained.append(row["id"])
        return encoded, retained, rejected

    train, train_ids, rejected_train = encode(partitions["train"])
    development, _, rejected_dev = encode(partitions["dev"])
    if not train or not development:
        raise ValueError("All training/development targets exceeded the token budget.")
    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100)

    def dev_loss(target_model):
        target_model.eval()
        total, count = 0., 0
        with torch.inference_mode():
            for offset in range(0, len(development), 2):
                batch = collator([dict(r) for r in development[offset:offset + 2]])
                batch = {k: v.to(device) for k, v in batch.items()}
                tokens = int((batch["labels"] != -100).sum())
                total += float(target_model(**batch, use_cache=False).loss) * tokens
                count += tokens
        return total / count

    config = {"language": language, "seed": seed, "model": spec.model_id, "revision": spec.revision,
              "maximum_updates": maximum, "evaluate_every": evaluate_every, "early_stopping_patience": 5,
              "batch_size": 2, "gradient_accumulation": 4, "nominal_effective_batch": 8,
              "learning_rate": .0003, "weight_decay": .01, "input_tokens": 256, "target_tokens": 64,
              "lora_rank": 8, "lora_alpha": 16, "lora_dropout": .05, "target_modules": ["q", "v"],
              "device": device, "precision": "float32", "target_prefix": target_prefix,
              "target_format_addendum_sha256": fingerprint_file(Path("docs/ARABIC_TARGET_FORMAT.md")) if language == "ar" else None,
              "data_sha256": {split: fingerprint_file(data_dir / f"{split}.jsonl") for split in partitions},
              "protocol_sha256": fingerprint_file(Path("docs/EXPANDED_TRAINING_PROTOCOL.md")),
              "trainer_sha256": fingerprint_file(Path(__file__))}
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Run configuration changed. Do not resume incompatible optimizer state.")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0003, weight_decay=.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=maximum)
    history, seen = [], set()
    epoch, cursor, start_step, best_step, stale = 0, 0, 0, 0, 0
    elapsed_before = 0.
    previous_adapter_loss = None
    if args.resume and state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=True)
        if saved["config"] != config:
            raise ValueError("Saved optimizer configuration differs.")
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        epoch, cursor, start_step = saved["epoch"], saved["cursor"], saved["step"]
        history, seen = saved["history"], set(saved["seen"])
        initial_loss, best_loss, best_step = saved["initial_loss"], saved["best_loss"], saved["best_step"]
        stale, elapsed_before = saved["stale"], saved["elapsed_seconds"]
        previous_adapter_loss = saved["previous_adapter_loss"]
        print(f"Resumed {language} at update {start_step}; {len(seen)} unique examples seen.", flush=True)
    else:
        if args.resume and (output / "adapter_config.json").exists():
            raise ValueError("An unfinished adapter has no optimizer checkpoint; preserve it for inspection.")
        initial_loss = best_loss = dev_loss(model)
        output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output)
        if language == "en":
            _, previous, _, _ = load_generator("en", device, "artifacts/seed_study/seed42")
            previous_adapter_loss = dev_loss(previous)
            load_generator.cache_clear()
            del previous
    started = time.perf_counter()
    order = training_order(len(train), seed, epoch)
    steps, examples_seen = start_step, sum(h.get("examples", 0) for h in history)
    print(json.dumps({"language": language, "training_examples": len(train), "development_examples": len(development),
                      "maximum_updates": maximum, "initial_dev_loss": initial_loss,
                      "previous_adapter_dev_loss": previous_adapter_loss, "device": device}), flush=True)

    def save_state(step, elapsed):
        saved = {"config": config, "adapter": get_peft_model_state_dict(model),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                 "epoch": epoch, "cursor": cursor, "step": step, "history": history, "seen": sorted(seen),
                 "initial_loss": initial_loss, "best_loss": best_loss, "best_step": best_step,
                 "stale": stale, "elapsed_seconds": elapsed, "previous_adapter_loss": previous_adapter_loss}
        temporary = state_path.with_suffix(".tmp")
        torch.save(saved, temporary)
        temporary.replace(state_path)

    if start_step == 0:
        save_state(0, elapsed_before)
    model.train()
    for step in range(start_step + 1, (maximum if stale < 5 else start_step) + 1):
        optimizer.zero_grad()
        loss_sum, batch_examples = 0., 0
        for _ in range(4):
            if cursor >= len(order):
                epoch, cursor = epoch + 1, 0
                order = training_order(len(train), seed, epoch)
            indices = order[cursor:cursor + 2]
            cursor += len(indices)
            batch_examples += len(indices)
            seen.update(train_ids[i] for i in indices)
            batch = collator([dict(train[i]) for i in indices])
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch, use_cache=False).loss / 4
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite; preserve the last optimizer checkpoint.")
            loss_sum += float(loss.detach())
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        steps, examples_seen = step, examples_seen + batch_examples
        event = {"step": step, "train_loss": loss_sum, "examples": batch_examples,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started}
        if step % evaluate_every == 0 or step == maximum:
            current = dev_loss(model)
            event["dev_loss"] = current
            if current < best_loss:
                best_loss, best_step, stale = current, step, 0
                model.save_pretrained(output)
            else:
                stale += 1
            model.train()
            event["elapsed_seconds"] = elapsed_before + time.perf_counter() - started
        history.append(event)
        print(json.dumps({"language": language, **event}), flush=True)
        if "dev_loss" in event:
            save_state(step, event["elapsed_seconds"])
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stale >= 5:
            print(f"Early stopping {language}: five development checks did not improve.", flush=True)
            break
    elapsed = elapsed_before + time.perf_counter() - started
    # The bundle retains the best validation checkpoint; optimizer state holds the last update.
    manifest = {**config, "training_complete": True, "steps": steps, "best_step": best_step,
                "initial_dev_loss": initial_loss, "best_dev_loss": best_loss,
                "previous_adapter_dev_loss": previous_adapter_loss, "elapsed_seconds": elapsed,
                "prepared_counts": {split: len(rows) for split, rows in partitions.items()},
                "counts": {"train": len(train), "dev": len(development), "test": len(partitions["test"])},
                "examples_seen": examples_seen, "unique_training_ids_seen": len(seen),
                "seen_training_ids_sha256": hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest(),
                "rejected_overlong_train_ids": rejected_train, "rejected_overlong_dev_ids": rejected_dev,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "selection_metric": "minimum development target-token cross-entropy, including step zero",
                "test_used_for_training_or_selection": False,
                "runtime_scope": "Optimizer loop and periodic validation; excludes setup and initial development scoring.",
                "environment": environment_info()}
    (output / "training_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "README.md").write_text(
        f"# {language.upper()} expanded-data LoRA adapter\n\n"
        f"Base: https://huggingface.co/{spec.model_id}, revision `{spec.revision}`. Base license: {spec.license}.\n\n"
        f"{len(train)} training examples; {steps} optimizer updates; {len(seen)} distinct training examples seen. "
        f"Saved step {best_step} selected by development token loss ({initial_loss:.4f} → {best_loss:.4f}). "
        "No test rows entered training or selection. Full backbone and optimizer state are excluded.\n\n"
        "See docs/EXPANDED_TRAINING_PROTOCOL.md and results/expanded_training for measured comparisons and source permissions.\n",
        encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "environment"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
