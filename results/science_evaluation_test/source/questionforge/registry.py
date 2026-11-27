"""Pinned public checkpoints. Remote Python code is never executed."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str
    prompt_style: str
    license: str


GENERATORS = {
    "en": ModelSpec("lmqg/t5-small-squad-qg", "57d0de8c285d3f89e01b5c607e22a24269fc5cc7", "en_highlight", "CC-BY-4.0"),
    "fr": ModelSpec("lmqg/mt5-small-frquad-qg", "1b4229a905dc759b3ebeeb22b1520398d1a2e818", "fr_highlight", "CC-BY-4.0"),
    "ar": ModelSpec("Mihakram/AraT5-base-question-generation", "1dc5a02543bcd9ce1f24cc404770ce11ddd39d74", "context_answer", "Unspecified; research demonstration only"),
}
VERIFIER = ModelSpec("deepset/xlm-roberta-base-squad2-distilled", "c1bbfe57bc3335c37960a48c5628ba26d7e9e3b7", "extractive_qa", "MIT")
