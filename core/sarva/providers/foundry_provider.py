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
    (directory / "config.json").write_text(json.dumps(config_data, indent=2))


def load_checkpoint_bundle(directory: Path) -> tuple[Any, Any, Any]:
    """Returns `(model, tokenizer, config)`, the model in `.eval()` mode
    with real trained weights loaded — not just a freshly-initialized
    model of the right shape.

    Backward compatible with bundles saved before `moe`/`rope_scaling`
    were wired into the save format: `.get(...)` defaults both to `None`
    (a bundle from before this change simply never had those keys at
    all), reconstructing exactly the dense/unscaled config it would have
    loaded as previously."""
    mods = _lazy_imports()
    config_data = json.loads((directory / "config.json").read_text())
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
    return model, tokenizer, config


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
    config_data = json.loads((path / "config.json").read_text())
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
    parts: list[str] = []
    if request.system:
        parts.append(request.system)
    parts.extend(m.text() for m in request.messages)
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
        self._loaded: dict[str, tuple[Any, Any, Any]] = {
            bundle_name: load_checkpoint_bundle(bundle_path)
            for bundle_name, bundle_path in bundles.items()
        }

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
        budget = config.max_seq_len - len(prompt_ids) - 1
        if budget <= 0:
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
