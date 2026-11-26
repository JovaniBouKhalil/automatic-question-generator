"""Describe an existing measured adapter without modifying its weights or historical metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / "artifacts/english_lora"
info = json.loads((directory / "training_manifest.json").read_text(encoding="utf-8"))
card = f"""# English question-generation LoRA adapter

Base: [{info['model']}](https://huggingface.co/{info['model']}), revision `{info['revision']}`.
The base checkpoint declares CC-BY-4.0. Credit the lmqg authors and original T5 authors.

Rank-8 Q/V attention updates: **{info['trainable_parameters']:,} trainable parameters** out of
{info['total_parameters']:,}. CPU training: {info['steps']} optimizer updates in
{info['elapsed_seconds']:.1f} seconds. Development token loss:
{info['initial_dev_loss']:.4f} → {info['best_dev_loss']:.4f}; selected step {info['best_step']}.

The adapter uses the paragraph-disjoint English training subset. No test example entered
training or checkpoint selection. A small decrease in token loss is not proof of improved
question quality. See `results/lora_test/` for the separate generation comparison.

`training_manifest.json` and `history.json` preserve measured losses, hyperparameters, data
hashes and environment. The recorded Git commit is the base working-tree commit during
development; the implementation checkpoint is `f071982`. Tokenizer warning suppression and
source-hash logging were improved subsequently without changing these adapter weights.
The frozen backbone is downloaded separately and is not redistributed.
"""
(directory / "README.md").write_text(card, encoding="utf-8")
