"""Small deterministic token counter for workflow tests; real tokenizers are checked separately."""

import math

import pytest


class ChatTokenizer:
    def encode(self, text, **kwargs):
        return list(range(math.ceil(len(text) / 4)))

    def apply_chat_template(self, messages, **kwargs):
        return self.encode("\n".join(m["content"] for m in messages))


@pytest.fixture
def chat_tokenizer():
    return ChatTokenizer()
