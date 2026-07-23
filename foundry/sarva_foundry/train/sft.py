"""sarva_foundry.train.sft — supervised fine-tuning data prep (spec
§3.6e: "SFT -> DPO/RLHF -> agentic RL... this, not pretraining, is what
turns a base model into a Fable/K3-class agent"). This is F2's first
real piece: everything that turns a `(prompt, response)` pair into a
`Trainer.train_step`-ready batch.

The property that distinguishes SFT from plain next-token pretraining
is the loss mask: the model must learn to predict the *response* given
the prompt, never to predict the prompt itself. A masking bug that
included prompt tokens in the loss would silently train the model to
reproduce/memorize prompts instead of learning to respond to them — a
failure mode that would still look completely normal in a loss curve
(loss still decreases; it's just decreasing on the wrong objective),
which is exactly why `tests/foundry/test_sft.py` verifies the mask's
effect on `Trainer.train_step`'s actual loss value directly, not just
that `build_sft_batch` returns tensors of the right shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR
from sarva_foundry.tokenizer import ByteLevelBPETokenizer


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    response: str


def encode_sft_example(
    example: SFTExample,
    tokenizer: ByteLevelBPETokenizer,
    end_of_turn: str = DOCUMENT_SEPARATOR,
) -> tuple[list[int], list[int]]:
    """Returns `(token_ids, loss_mask)`: prompt tokens, then response
    tokens, then `end_of_turn` (so the model learns when to stop
    generating — the same boundary-marking role `end_of_turn` plays
    between documents in plain pretraining, reused rather than inventing
    a second special token for the same purpose). `loss_mask[i] == 1`
    iff `token_ids[i]` is part of the response or its terminating
    `end_of_turn` token; `0` for every prompt token."""
    if end_of_turn not in tokenizer.special_tokens:
        raise ValueError(f"tokenizer was not trained with {end_of_turn!r} as a special token")
    prompt_ids = tokenizer.encode(example.prompt)
    response_ids = tokenizer.encode(example.response) + tokenizer.encode(end_of_turn)
    token_ids = prompt_ids + response_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return token_ids, loss_mask


def build_sft_batch(
    examples: list[SFTExample],
    tokenizer: ByteLevelBPETokenizer,
    end_of_turn: str = DOCUMENT_SEPARATOR,
    pad_token_id: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pads a batch of SFT examples to the longest sequence and shifts
    for next-token prediction, returning `(input_ids, target_ids,
    loss_mask)` ready for `Trainer.train_step(x, y, loss_mask=...)`.

    Right-padding is safe under causal attention, by construction rather
    than by convention: a padded position can never influence an
    earlier, real position's output (causal masking already guarantees
    this — see `test_model.py`'s causal-masking test), and a padded
    position's own output is excluded from the loss via the mask.
    """
    encoded = [encode_sft_example(e, tokenizer, end_of_turn) for e in examples]
    max_len = max(len(ids) for ids, _ in encoded)
    if max_len < 2:
        raise ValueError("every example needs at least 2 tokens total to form one training pair")

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for token_ids, mask in encoded:
        pad_len = max_len - len(token_ids)
        ids = token_ids + [pad_token_id] * pad_len
        m = mask + [0] * pad_len
        input_rows.append(ids[:-1])
        target_rows.append(ids[1:])
        mask_rows.append(m[1:])

    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(target_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.float),
    )
