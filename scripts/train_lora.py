"""A bounded English LoRA adaptation experiment; no test questions enter training.

LoRA learns low-rank updates to attention Q/V projections. The frozen 60M-parameter
backbone is reused. Checkpoint selection uses only development token loss.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from questionforge.data import assert_disjoint, load_jsonl
from questionforge.experiment import environment_info, fingerprint_file
from questionforge.models import answer_window, configure_torch, format_prompt, model_source
from questionforge.registry import GENERATORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/english_lora"))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.learning_rate <= 0:
        parser.error("Steps and learning rate must be positive.")
    if (args.output / "adapter_config.json").exists():
        parser.error("Output already contains an adapter; choose a new directory to preserve it.")
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        get_linear_schedule_with_warmup,
    )

    torch, device = configure_torch(args.seed, args.device)
    random.seed(args.seed)
    partitions = {split: [r for r in load_jsonl(args.data_dir / f"{split}.jsonl") if r["language"] == "en"]
                  for split in ["train", "dev", "test"]}
    assert_disjoint(partitions)
    if not partitions["train"] or not partitions["dev"]:
        raise ValueError("Training and development data are required.")
    spec = GENERATORS["en"]
    source, kwargs = model_source(spec)
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
    backbone = AutoModelForSeq2SeqLM.from_pretrained(source, trust_remote_code=False, **kwargs)
    model = get_peft_model(backbone, LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
                           lora_dropout=0.05, target_modules=["q", "v"], bias="none"))
    model.peft_config["default"].base_model_name_or_path = spec.model_id
    model.peft_config["default"].revision = spec.revision
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)", flush=True)

    def encode(row):
        window, start, _ = answer_window(row["context"], row["answer"], row["answer_start"], tokenizer,
                                         spec.prompt_style, 256)
        encoded = tokenizer(format_prompt(window, row["answer"], start, spec.prompt_style))
        encoded["labels"] = tokenizer(text_target=row["question"], max_length=64, truncation=True)["input_ids"]
        return encoded

    train = [encode(row) for row in partitions["train"]]
    dev = [encode(row) for row in partitions["dev"]]
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    loader = DataLoader(train, batch_size=2, shuffle=True, collate_fn=collator,
                        generator=torch.Generator().manual_seed(args.seed))
    dev_loader = DataLoader(dev, batch_size=2, collate_fn=collator)

    def validation_loss():
        model.eval()
        loss_sum, token_count = 0.0, 0
        with torch.inference_mode():
            for batch in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                count = int((batch["labels"] != -100).sum())
                loss_sum += float(model(**batch).loss) * count
                token_count += count
        model.train()
        return loss_sum / token_count

    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    best_loss = validation_loss()
    initial_loss, best_step = best_loss, 0
    model.save_pretrained(args.output)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=min(4, args.steps // 10), num_training_steps=args.steps)
    iterator = iter(loader)
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        train_loss = 0.0
        for _ in range(4):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            train_loss += float(loss.detach()) / 4
            (loss / 4).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        event = {"step": step, "train_loss": train_loss, "elapsed_seconds": time.perf_counter() - started}
        if step % 10 == 0 or step == args.steps:
            dev_loss = validation_loss()
            event["dev_loss"] = dev_loss
            if dev_loss < best_loss:
                best_loss, best_step = dev_loss, step
                model.save_pretrained(args.output)
        history.append(event)
        print(json.dumps(event), flush=True)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    manifest = {"model": spec.model_id, "revision": spec.revision, "seed": args.seed,
                "steps": args.steps, "learning_rate": args.learning_rate, "batch_size": 2,
                "gradient_accumulation": 4, "effective_batch_size": 8, "input_tokens": 256, "target_tokens": 64,
                "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05, "target_modules": ["q", "v"],
                "trainable_parameters": trainable, "total_parameters": total, "device": device,
                "selection_metric": "development target-token cross-entropy (lower is better)",
                "initial_dev_loss": initial_loss, "best_dev_loss": best_loss, "best_step": best_step,
                "elapsed_seconds": time.perf_counter() - started,
                "counts": {k: len(v) for k, v in partitions.items()},
                "data_sha256": {k: fingerprint_file(args.data_dir / f"{k}.jsonl") for k in partitions},
                "environment": environment_info(), "test_used_for_training_or_selection": False}
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
