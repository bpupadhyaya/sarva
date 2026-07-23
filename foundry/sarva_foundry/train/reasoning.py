"""sarva_foundry.train.reasoning — reasoning/thinking-token training
(spec §3.6a: "reasoning/thinking-token support... o1/R1-class"; the
design doc's own capability table names "long CoT RL, verifiable
rewards" as the specific recipe, citing DeepSeek-R1-class open recipes
directly). This was the one item on §3.6a's frontier-architecture list
with zero code anywhere in `foundry/` — confirmed by grep before
starting.

Deliberately NOT a new architecture or a new training algorithm: SFT
(`sarva_foundry.train.sft`) and GRPO (`sarva_foundry.train.rl`) already
exist and are reused completely unchanged. What's missing, and what
this module adds, is the *reward function* that turns GRPO into
reasoning-token training specifically — one that checks a completion
actually used the `<think>...</think>` format AND got the right answer,
mirroring DeepSeek-R1's own published reward design (a format reward
plus an accuracy reward, summed).

The two-stage shape this module's own example script follows — a small
"cold start" SFT stage on hand-written `<think>...</think>` traces,
THEN GRPO refinement against `reasoning_reward` — is not an arbitrary
design choice; it's the R1 paper's own finding, taken directly: pure RL
from a base model ("R1-Zero" in the paper) learned to reason but
produced real format/readability problems, which is exactly why the
paper's final R1 recipe adds a cold-start SFT stage before RL. This
project's own GRPO work found something structurally similar (tiny,
freshly-initialized transformers have extremely peaked initial sampling
distributions, leaving no exploration for RL to work with) — the
cold-start stage here serves the same real purpose: giving RL something
better than random noise to refine, not starting from nothing.
"""

from __future__ import annotations

import re

THINK_START = "<think>"
THINK_END = "</think>"


def format_reward(completion_text: str) -> float:
    """1.0 iff `completion_text` is a single well-formed `<think>...
    </think>` block wrapping non-empty reasoning, followed by a
    non-empty answer segment; 0.0 otherwise. Checked structurally, not
    just "do the tag strings appear somewhere" — a `<think>` buried in
    the middle of unrelated text, or an empty `<think></think>`, does
    not count as genuinely following the format.

    Requires EXACTLY one `<think>`/`</think>` pair, not just a match
    against the first occurrence — caught empirically while building
    this module's own example script, not a hypothetical concern: an
    early GRPO-trained model discovered that repeating `</think>` many
    times after the real one padded the "answer segment"
    `answer_reward` scans with extra copies of the target digit,
    inflating reward from a real reward-hacking exploit of a looser,
    first-occurrence-only version of this check, not genuine
    correctness. Rejecting any completion with more than one closing
    tag closes that specific loophole."""
    if completion_text.count(THINK_START) != 1 or completion_text.count(THINK_END) != 1:
        return 0.0
    match = re.match(
        rf"^{re.escape(THINK_START)}(.+?){re.escape(THINK_END)}(.*)$",
        completion_text,
        re.DOTALL,
    )
    if match is None:
        return 0.0
    reasoning, answer = match.group(1), match.group(2)
    if not reasoning.strip() or not answer.strip():
        return 0.0
    return 1.0


def answer_reward(completion_text: str, expected_answer: str) -> float:
    """1.0 iff `expected_answer` appears anywhere in whatever text
    follows `</think>` — the same `contains_match` philosophy
    `sarva.eval`'s default grader uses, since real models rarely answer
    with *only* the expected string. Returns 0.0 (not a crash or an
    exception) when there's no `</think>` at all, since an unformatted
    completion has no defined "answer segment" to check in the first
    place.

    Requires EXACTLY one `</think>`, same as `format_reward` — a second
    real reward-hacking exploit caught empirically, not hypothetical:
    with only `format_reward` tightened to reject multiple closing
    tags, GRPO training still found that repeating `</think>` many
    times (abandoning format reward entirely, worth only 0.3 of
    `reasoning_reward`'s weight) padded the "answer segment" this
    function scans with extra copies of the target digit, inflating the
    larger 0.7-weighted answer reward without genuinely answering
    correctly. A completion with more than one `</think>` is treated as
    malformed by both reward functions consistently, not just one."""
    if completion_text.count(THINK_END) != 1:
        return 0.0
    answer_segment = completion_text.split(THINK_END, 1)[1]
    return 1.0 if expected_answer in answer_segment else 0.0


def reasoning_reward(
    completion_text: str,
    expected_answer: str,
    format_weight: float = 0.3,
    answer_weight: float = 0.7,
) -> float:
    """The combined verifiable reward GRPO trains against — a weighted
    sum of format compliance and answer correctness, the same two-signal
    shape DeepSeek-R1's own reward design uses. Weighted toward
    correctness (0.7 vs 0.3 by default) since a perfectly-formatted
    wrong answer is worth less than a correct one, but format still
    contributes real, separate signal — a model that abandons the
    format entirely to chase easier reward would score strictly worse
    than one that keeps it and is also correct."""
    return format_weight * format_reward(completion_text) + answer_weight * answer_reward(
        completion_text, expected_answer
    )
