from __future__ import annotations

import pytest

from daph_learning.routing.logit_router import (
    score_route_batch_from_logits,
    score_route_batch_from_sequences,
)


torch = pytest.importorskip("torch")
from torch import nn


class MultiTokenTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        if text in {" SYMBOLIC", "SYMBOLIC"}:
            return [10, 11]
        if text in {" LLM", "LLM"}:
            return [20]
        if text.endswith(" SYMBOLIC"):
            return [1, 2, 3, 10, 11]
        if text.endswith(" LLM"):
            return [1, 2, 3, 20]
        return [1, 2, 3]

    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            raise NotImplementedError
        if isinstance(text, list):
            encoded = [self.encode(value, add_special_tokens=True) for value in text]
            width = max(map(len, encoded))
            padded = [[0] * (width - len(ids)) + ids for ids in encoded]
            return {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(
                    [[0] * (width - len(ids)) + [1] * len(ids) for ids in encoded]
                ),
            }
        ids = self.encode(text, add_special_tokens=True)
        if kwargs.get("return_tensors") == "pt":
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            }
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class SequencePreferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 4)
        inner = nn.Module()
        inner.layers = nn.ModuleList([nn.Identity()])
        self.model = inner
        self.config = type(
            "Config",
            (),
            {"hidden_size": 4, "_name_or_path": "sequence-test"},
        )()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = torch.zeros(
            *input_ids.shape,
            64,
            dtype=hidden.dtype,
            device=hidden.device,
        )
        # First continuation token prefers SYMBOLIC by one logit.
        prompt_positions = input_ids == 3
        logits[..., 10] = torch.where(
            prompt_positions,
            torch.tensor(5.0, device=hidden.device),
            logits[..., 10],
        )
        logits[..., 20] = torch.where(
            prompt_positions,
            torch.tensor(4.0, device=hidden.device),
            logits[..., 20],
        )
        # Completing SYMBOLIC is extremely unlikely, so the complete label
        # should lose to LLM despite its attractive first sub-token.
        sym_prefix_positions = input_ids == 10
        logits[..., 11] = torch.where(
            sym_prefix_positions,
            torch.tensor(-10.0, device=hidden.device),
            logits[..., 11],
        )
        return type("Output", (), {"logits": logits})()


def test_full_sequence_scoring_reverses_misleading_first_token() -> None:
    model = SequencePreferenceModel()
    tokenizer = MultiTokenTokenizer()
    prompt = "Choose a backend.\nACTION:"

    first_token = score_route_batch_from_logits(
        [prompt],
        model,
        tokenizer,
        anchor="ACTION:",
        token_resolver="contextual",
        allow_first_token_fallback=True,
    )
    full_sequence = score_route_batch_from_sequences(
        [prompt],
        model,
        tokenizer,
        anchor="ACTION:",
        normalization="mean",
    )
    assert first_token[0][0] == "symbolic"
    assert full_sequence[0][0] == "llm"


def test_sequence_scoring_fails_when_boundary_retokenizes() -> None:
    class UnstableTokenizer(MultiTokenTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.endswith(" SYMBOLIC"):
                return [99, 10, 11]
            return super().encode(text, add_special_tokens=add_special_tokens)

    from daph_learning.routing.errors import ContextBoundaryError

    with pytest.raises(ContextBoundaryError):
        score_route_batch_from_sequences(
            ["Choose a backend.\nACTION:"],
            SequencePreferenceModel(),
            UnstableTokenizer(),
        )
