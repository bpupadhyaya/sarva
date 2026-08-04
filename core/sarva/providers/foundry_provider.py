"""sarva.providers.foundry_provider — plugging a checkpoint trained by
this project's own from-scratch training code (`sarva_foundry`) into the
same `Provider` registry every frontier backend uses.

This is the piece the design doc's repo-structure diagram already names
(`providers/foundry.py`) and the eval-harness entry left explicitly
unbuilt: "the moment §3.1's planned foundry adapter exists... it becomes
gradable by this same harness with zero changes." It also closes the
loop the other direction from `sarva.distill` (core -> foundry SFT data)
and `sarva_foundry.train` (SFT/DPO/GRPO) — a checkpoint trained that way
can now come back into core as a real, routable model.

`sarva_foundry` (and its `torch` dependency) is deliberately NOT a hard
dependency of `sarva` core — it's the optional `foundry` extra (`pip
install sarva[foundry]`), for the same reason `core`/`sarva_foundry` have
been kept dependency-disjoint since the distillation glue script: most
Sarva installs never train or run a local model and shouldn't need to
pull in torch. Every function in this module that actually touches torch
or `sarva_foundry` imports them lazily and raises a clear, actionable
`ImportError` if the extra isn't installed — importing this MODULE always
succeeds either way, so `sarva.runtime` can probe for checkpoints without
crashing on a plain-core install.

A checkpoint "bundle" is a directory with three files this module writes
and reads together: `model.pt` (a `sarva_foundry.train.Trainer`
checkpoint), `tokenizer.json` (a `ByteLevelBPETokenizer`), and
`config.json` (the `TransformerConfig` fields needed to reconstruct the
model before loading weights into it, now including a real MoE and/or
long-context RoPE-scaling config when the checkpoint was trained with
either — both flat, JSON-safe dataclasses, serialized as nested
`null`-or-object fields; a bundle saved before this existed simply has
no such keys, and loads exactly as it always did). MoE/RoPE-scaling
support here used to be a real, named, deferred gap (`save_checkpoint_
bundle` refused a checkpoint trained with either rather than silently
reloading it as a plain dense/unscaled model) — closed once the two
config dataclasses turned out to need nothing more than a `dict` of
their own already-JSON-safe fields.

No chat template is applied when building a prompt: the prompt is just
the concatenated text of the system prompt (if any) and every message's
text, in order — exactly matching how `examples/10_sft_toy_assistant.py`
and `sarva_foundry.train.sft`'s tests train on raw prompt text with no
role tags. A checkpoint actually trained with some other chat-formatting
convention would need this to match; that's a real, named limitation of
this first adapter, not silently assumed universal.

Also honestly scoped: generation runs synchronously (`asyncio.to_thread`,
so the event loop keeps yielding to other work, but there is no wire-level
streaming protocol to translate the way there is for a real network API) —
the whole completion is decoded once and streamed as a single
`TextDeltaEvent`, not true incremental per-token streaming. Generation
itself now uses `sarva_foundry.inference.generate_with_cache` (real
KV-cache reuse across steps — see `sarva_foundry.model.kv_cache` — rather
than the naive full-recompute-per-token `sample_completion` this adapter
used before), but there is still no batching across concurrent requests —
one sequence at a time, the remaining named gap a real foundry inference
server (deferred, separate scope) would close.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sarva.atomic_write import atomic_write_text
from sarva.multimodal.content import Message, Modality, TextBlock
from sarva.providers.base import (
    DoneEvent,
    GenerateRequest,
    ModelCapabilities,
    ModelCost,
    ModelInfo,
    ModelNotFoundError,
    ProviderEvent,
    StopReason,
    TextDeltaEvent,
    Usage,
)

_CONFIG_FIELDS = (
    "vocab_size",
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "max_seq_len",
    "rope_theta",
    "norm_eps",
    "hidden_dim",
)


def _lazy_imports() -> Any:
    """Every real torch/sarva_foundry symbol this module needs, imported
    on demand. Returns a tiny namespace object rather than a tuple so call
    sites read as `mods.torch`, `mods.sample_completion`, etc. instead of
    an unlabeled positional unpack."""
    try:
        import torch
        from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR
        from sarva_foundry.inference import generate_with_cache
        from sarva_foundry.model import (
            DecoderOnlyTransformer,
            MoEConfig,
            RopeScalingConfig,
            TransformerConfig,
        )
        from sarva_foundry.tokenizer import ByteLevelBPETokenizer
    except ImportError as exc:
        raise ImportError(
            "FoundryProvider needs the optional 'foundry' extra (torch + "
            "sarva_foundry). Install with `pip install sarva[foundry]`, or "
            "inside this repo's uv workspace: `uv sync --all-packages`."
        ) from exc

    class _Mods:
        pass

    mods = _Mods()
    mods.torch = torch
    mods.DOCUMENT_SEPARATOR = DOCUMENT_SEPARATOR
    mods.DecoderOnlyTransformer = DecoderOnlyTransformer
    mods.TransformerConfig = TransformerConfig
    mods.MoEConfig = MoEConfig
    mods.RopeScalingConfig = RopeScalingConfig
    mods.ByteLevelBPETokenizer = ByteLevelBPETokenizer
    mods.generate_with_cache = generate_with_cache
    return mods


def _serialize_moe(moe: Any) -> dict[str, Any] | None:
    if moe is None:
        return None
    return {
        "n_experts": moe.n_experts,
        "n_experts_per_tok": moe.n_experts_per_tok,
        "n_shared_experts": moe.n_shared_experts,
    }


def _serialize_rope_scaling(rope_scaling: Any) -> dict[str, Any] | None:
    if rope_scaling is None:
        return None
    return {"method": rope_scaling.method, "factor": rope_scaling.factor}


def save_checkpoint_bundle(directory: Path, trainer: Any, tokenizer: Any, config: Any) -> None:
    """Writes `model.pt` + `tokenizer.json` + `config.json` to `directory`.
    `trainer`/`tokenizer`/`config` are a real `sarva_foundry.train.Trainer`,
    `ByteLevelBPETokenizer`, and `TransformerConfig` — this function itself
    doesn't need `_lazy_imports()` since it only calls methods on objects
    the caller already constructed (and imported torch to build).

    `moe`/`rope_scaling` serialize to plain nested dicts (both are flat
    dataclasses with only JSON-safe fields) -- `null` when unset, so a
    dense/unscaled checkpoint's `config.json` looks exactly as it did
    before either was wired in here."""
    directory.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(directory / "model.pt")
    tokenizer.save(directory / "tokenizer.json")
    config_data = {field: getattr(config, field) for field in _CONFIG_FIELDS}
    config_data["moe"] = _serialize_moe(config.moe)
    config_data["rope_scaling"] = _serialize_rope_scaling(config.rope_scaling)
    # Atomic write: this bundle's config.json is what load_checkpoint_bundle
    # reads back alongside the (already atomically-saved) model.pt/
    # tokenizer.json -- a crash mid-write here would leave a real trained
    # checkpoint's model/tokenizer files intact but the config needed to
    # actually reconstruct the model unreadable. See sarva.atomic_write.
    atomic_write_text(directory / "config.json", json.dumps(config_data, indent=2))


# A real bug found by actually running build_providers() in a loop the
# way sarva.server.app's /chat and /ws/chat handlers do it (a fresh
# AgentLoop -- and therefore a fresh FoundryProvider -- per request, so
# a saved config.json change takes effect without a server restart):
# every request re-did a full torch.load() of every configured
# checkpoint's weights, even requests that ended up routed to a
# completely different provider (Claude/GPT/Ollama), and even though
# nothing about the checkpoint had changed between requests. Confirmed
# live: 5 simulated requests against a real toy checkpoint fired
# load_checkpoint_bundle 5/5 times with zero reuse. Cached here, keyed
# by the resolved directory path AND model.pt's own mtime (not path
# alone) -- the same "mtime distinguishes touched from untouched"
# technique already used to fix the cross-process lock's Windows bug --
# so a checkpoint retrained and re-saved in place is picked up on its
# very next load instead of silently serving stale weights forever.
# Safe to share the returned model instance across concurrent requests:
# generate_with_cache (sarva_foundry.inference.generate_with_cache)
# allocates a fresh KVCache per call and never mutates persistent model
# state, and this adapter never trains through the loaded model.
_bundle_cache: dict[tuple[str, float], tuple[Any, Any, Any]] = {}


def load_checkpoint_bundle(directory: Path) -> tuple[Any, Any, Any]:
    """Returns `(model, tokenizer, config)`, the model in `.eval()` mode
    with real trained weights loaded — not just a freshly-initialized
    model of the right shape.

    Backward compatible with bundles saved before `moe`/`rope_scaling`
    were wired into the save format: `.get(...)` defaults both to `None`
    (a bundle from before this change simply never had those keys at
    all), reconstructing exactly the dense/unscaled config it would have
    loaded as previously.

    Cached across calls -- see `_bundle_cache`'s own comment above."""
    mods = _lazy_imports()
    model_pt_path = directory / "model.pt"
    cache_key = (str(directory.resolve()), model_pt_path.stat().st_mtime)
    cached = _bundle_cache.get(cache_key)
    if cached is not None:
        return cached

    config_data = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    moe_data = config_data.pop("moe", None)
    rope_scaling_data = config_data.pop("rope_scaling", None)
    config_data["moe"] = mods.MoEConfig(**moe_data) if moe_data is not None else None
    config_data["rope_scaling"] = (
        mods.RopeScalingConfig(**rope_scaling_data) if rope_scaling_data is not None else None
    )
    config = mods.TransformerConfig(**config_data)
    tokenizer = mods.ByteLevelBPETokenizer.load(directory / "tokenizer.json")
    model = mods.DecoderOnlyTransformer(config)
    # weights_only=True: same posture as Trainer.load_checkpoint -- refuse
    # to unpickle anything beyond documented safe types.
    state = mods.torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()
    result = (model, tokenizer, config)
    # A real bug found by giving this cache its own fresh-eyes sweep,
    # applying the project's own "does yesterday's fix have a bug"
    # lens to the fix that added this cache in the first place: keyed
    # by (directory, mtime) so a retrained-and-resaved checkpoint is
    # picked up fresh, but nothing ever evicted the OLD entry once a
    # new one landed for the same directory. Confirmed live: 20
    # retrain/re-save/reload cycles against one checkpoint directory
    # (the exact "no server restart needed" workflow this cache exists
    # to support) left 20 full model copies resident in memory
    # simultaneously -- 20x a single model's real footprint -- even
    # though only the newest entry is ever reachable again through
    # ordinary use, an unbounded leak over a long-running `sarva
    # serve` process's uptime. A directory only ever has one current
    # mtime at a time, so any entry under a different mtime for the
    # same resolved path is permanently unreachable the moment a new
    # one lands -- evicted here rather than left to accumulate.
    for stale_key in [k for k in _bundle_cache if k[0] == cache_key[0] and k != cache_key]:
        del _bundle_cache[stale_key]
    _bundle_cache[cache_key] = result
    return result


def discover_checkpoint_bundles(checkpoints_dir: Path) -> dict[str, Path]:
    """Every subdirectory of `checkpoints_dir` that has all three bundle
    files, keyed by directory name (which becomes the `foundry/<name>`
    model id). Torch-free -- safe to call even without the foundry extra
    installed, which is exactly what lets `sarva.runtime` probe for
    checkpoints before deciding whether to attempt the real (torch-needing)
    load."""
    if not checkpoints_dir.is_dir():
        return {}
    return {
        p.name: p
        for p in sorted(checkpoints_dir.iterdir())
        if p.is_dir()
        and (p / "config.json").is_file()
        and (p / "tokenizer.json").is_file()
        and (p / "model.pt").is_file()
    }


def model_info_for_bundle(name: str, path: Path) -> ModelInfo:
    """Builds a registry `ModelInfo` from just `config.json` -- no torch
    needed, so the registry can list a foundry checkpoint as a known model
    even on a plain-core install where it isn't actually loadable/runnable
    yet (the CLI's `models` command already marks unavailable models with
    an empty checkbox for exactly this kind of case)."""
    config_data = json.loads((path / "config.json").read_text(encoding="utf-8"))
    return ModelInfo(
        id=f"foundry/{name}",
        provider="foundry",
        display_name=f"Foundry checkpoint: {name} (local, from-scratch)",
        local=True,
        capabilities=ModelCapabilities(
            modalities_in={Modality.TEXT},
            modalities_out={Modality.TEXT},
            tool_use=False,
            thinking=False,
            context_window=config_data["max_seq_len"],
            max_output=config_data["max_seq_len"],
        ),
        cost=ModelCost(input_per_mtok=0.0, output_per_mtok=0.0),
    )


def _flatten_prompt(request: GenerateRequest) -> str:
    """`Message.text()` silently drops every non-`TextBlock` -- exactly
    right for a caller that only wants "just the words" from a message
    it already knows is text-only, but wrong here: a foundry checkpoint
    is genuinely text-only end to end (`model_info_for_bundle` declares
    `modalities_in={Modality.TEXT}`, `tool_use=False`), so any other
    block type reaching this function means a caller sent content (an
    image, a tool call, ...) this adapter has no wire-format mapping
    for at all. Silently dropping it would answer as if that content
    were never sent -- a materially misleading response, not a cosmetic
    gap, the same reasoning the Anthropic/OpenAI/Google adapters already
    apply to their own untranslatable block types. Reachable today only
    via an explicit model override (the router's own `needs` modality
    check would otherwise never route such a request here), same as
    those adapters' own reachability note."""
    parts: list[str] = []
    if request.system:
        parts.append(request.system)
    for m in request.messages:
        for b in m.content:
            if not isinstance(b, TextBlock):
                raise ValueError(
                    f"FoundryProvider cannot translate a {type(b).__name__!r} content "
                    "block (foundry checkpoints are text-only end to end)"
                )
        parts.append(m.text())
    return "".join(parts)


class FoundryProvider:
    """One instance can serve multiple checkpoints -- every bundle found
    under `checkpoints_dir` at construction time, each addressable as
    `foundry/<bundle-directory-name>` (the same `<provider>/<local-id>`
    namespacing `ollama/qwen3:8b` already established)."""

    name = "foundry"

    def __init__(self, checkpoints_dir: Path):
        bundles = discover_checkpoint_bundles(checkpoints_dir)
        if not bundles:
            raise ValueError(f"no valid foundry checkpoint bundles found under {checkpoints_dir}")
        # A real bug found by actually corrupting a real bundle's
        # model.pt (truncating it, simulating an interrupted save or
        # disk corruption) and calling the real build_providers():
        # torch.load() raised an uncaught OSError, and since this used
        # to be a plain dict comprehension with no error handling at
        # all, one bad bundle crashed every caller of build_providers()
        # -- every CLI command, /chat, /ws/chat, server startup -- not
        # just a request that tried to use that specific checkpoint.
        # discover_checkpoint_bundles only checks that the three bundle
        # files *exist*, never that they're actually valid, so this was
        # reachable any time a bundle got corrupted after being
        # discovered. Same "one bad case shouldn't take down everything"
        # posture already applied in sarva.eval.harness.run_benchmark:
        # a broken bundle is recorded (name -> error message, publicly
        # readable via self.broken_bundles) and skipped, not silently
        # dropped and not fatal to every OTHER valid bundle.
        self._loaded: dict[str, tuple[Any, Any, Any]] = {}
        self.broken_bundles: dict[str, str] = {}
        for bundle_name, bundle_path in bundles.items():
            try:
                self._loaded[bundle_name] = load_checkpoint_bundle(bundle_path)
            except Exception as e:
                self.broken_bundles[bundle_name] = f"{type(e).__name__}: {e}"
        if not self._loaded:
            detail = "; ".join(f"{name}: {reason}" for name, reason in self.broken_bundles.items())
            raise ValueError(
                f"found {len(bundles)} checkpoint bundle(s) under {checkpoints_dir}, but "
                f"none could actually be loaded: {detail}"
            )

    def _resolve(self, model_id: str) -> tuple[Any, Any, Any]:
        bundle_name = model_id.split("/", 1)[1] if "/" in model_id else model_id
        if bundle_name not in self._loaded:
            raise ModelNotFoundError(
                f"foundry checkpoint {bundle_name!r} not loaded (have: {sorted(self._loaded)})"
            )
        return self._loaded[bundle_name]

    async def generate(self, request: GenerateRequest) -> AsyncIterator[ProviderEvent]:
        import asyncio

        mods = _lazy_imports()
        model, tokenizer, config = self._resolve(request.model)

        prompt_ids = tokenizer.encode(_flatten_prompt(request))
        stop_token_id: int | None = None
        if mods.DOCUMENT_SEPARATOR in tokenizer.special_tokens:
            stop_token_id = tokenizer.special_tokens[mods.DOCUMENT_SEPARATOR]

        # `config.max_seq_len` bounds every position RoPE was precomputed
        # for -- generating past it would index outside that table, so the
        # available budget is capped here rather than left to fail deep
        # inside the model on a long prompt.
        #
        # A real bug found by actually sending an empty message (`sarva
        # chat "" --model foundry/<bundle>`, or an empty `--system`/
        # history with no other content, reachable with zero validation
        # anywhere upstream: `chat()`'s `message` CLI argument and
        # `ChatRequest.message` both accept ""): `not prompt_ids` used to
        # fall through this guard (budget is still positive with zero
        # prompt tokens) straight into `generate_with_cache`, where
        # `torch.tensor([[]])` -- no elements to infer a dtype from --
        # defaults to float32 instead of int64, crashing the token
        # embedding lookup with a raw, implementation-leaking
        # RuntimeError ("Expected tensor for argument #1 'indices' to
        # have one of the following scalar types: Long, Int; but got
        # torch.FloatTensor instead"). Confirmed live before this fix.
        # Even fixing just the dtype wouldn't actually make an empty
        # prompt generatable -- prefilling zero token positions leaves
        # no "last" logit for `next_logits = logits[0, -1]` to sample
        # from either, an IndexError one line later. Genuinely nothing
        # to prefill with, the same "no useful work to do" shape the
        # budget<=0 case right below already handles gracefully, so it
        # gets the identical clean-empty-response treatment rather than
        # its own special case.
        budget = config.max_seq_len - len(prompt_ids) - 1
        if budget <= 0 or not prompt_ids:
            yield DoneEvent(
                stop_reason=StopReason.MAX_TOKENS,
                message=Message(role="assistant", content=[TextBlock(text="")]),
                usage=Usage(input_tokens=len(prompt_ids)),
            )
            return

        max_new = min(request.config.max_tokens, budget)
        completion_ids = await asyncio.to_thread(
            mods.generate_with_cache, model, prompt_ids, max_new, 0.0, stop_token_id
        )

        hit_stop = bool(completion_ids) and completion_ids[-1] == stop_token_id
        text_ids = completion_ids[:-1] if hit_stop else completion_ids
        text = tokenizer.decode(text_ids)

        if text:
            yield TextDeltaEvent(text=text)

        yield DoneEvent(
            stop_reason=StopReason.END_TURN if hit_stop else StopReason.MAX_TOKENS,
            message=Message(role="assistant", content=[TextBlock(text=text)]),
            usage=Usage(input_tokens=len(prompt_ids), output_tokens=len(completion_ids)),
        )

    async def close(self) -> None:
        return None
