# English question-generation LoRA adapter

Base: [lmqg/t5-small-squad-qg](https://huggingface.co/lmqg/t5-small-squad-qg), pinned revision `57d0de8c285d3f89e01b5c607e22a24269fc5cc7`. Base checkpoint license: CC-BY-4.0. Credit the lmqg authors and original T5 authors.

Rank 8 Q/V attention updates: 294,912 trainable parameters out of 60,787,712. Trained for 160 optimizer updates on the paragraph-disjoint English training subset. Checkpoint selected at step 160 using development token loss (1.5944 → 1.5291). No test examples entered training or selection.

This is a small adaptation experiment, not a claim of state-of-the-art quality. See training_manifest.json and history.json for data hashes, configuration and measured losses; results/lora_test for generation comparisons. The frozen backbone is not redistributed.
