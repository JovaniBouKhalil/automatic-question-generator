# English question-generation LoRA adapter

Base: [lmqg/t5-small-squad-qg](https://huggingface.co/lmqg/t5-small-squad-qg), revision `57d0de8c285d3f89e01b5c607e22a24269fc5cc7`.
The base checkpoint declares CC-BY-4.0. Credit the lmqg authors and original T5 authors.

Rank-8 Q/V attention updates: **294,912 trainable parameters** out of
60,787,712. CPU training: 40 optimizer updates in
64.2 seconds. Development token loss:
1.5944 → 1.5834; selected step 40.

The adapter uses the paragraph-disjoint English training subset. No test example entered
training or checkpoint selection. A small decrease in token loss is not proof of improved
question quality. See `results/lora_test/` for the separate generation comparison.

`training_manifest.json` and `history.json` preserve measured losses, hyperparameters, data
hashes and environment. The recorded Git commit is the base working-tree commit during
development; the implementation checkpoint is `f071982`. Tokenizer warning suppression and
source-hash logging were improved subsequently without changing these adapter weights.
The frozen backbone is downloaded separately and is not redistributed.
