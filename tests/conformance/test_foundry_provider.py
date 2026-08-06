"""Conformance tests for sarva.providers.foundry_provider -- the adapter
that plugs a `sarva_foundry`-trained checkpoint into the same `Provider`
registry every frontier backend uses. Runs a real checkpoint through the
real adapter end to end (train tiny -> save bundle -> discover -> load ->
generate), not a mocked stand-in, matching this project's "verify it
actually works, don't assume the shapes line up" discipline throughout."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
import torch
from sarva.multimodal.content import Message, Modality, TextBlock
from sarva.providers.base import GenerateConfig, GenerateRequest, ModelNotFoundError, StopReason
from sarva.providers.foundry_provider import (
    FoundryProvider,
    discover_checkpoint_bundles,
    load_checkpoint_bundle,
    model_info_for_bundle,
    save_checkpoint_bundle,
)
from sarva.providers.registry import Registry
from sarva_foundry.data.dataset import DOCUMENT_SEPARATOR
from sarva_foundry.model import (
    DecoderOnlyTransformer,
    MoEConfig,
    RopeScalingConfig,
    TransformerConfig,
)
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train import Trainer

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the sky is blue and the grass is green",
]


def _tiny_tokenizer() -> ByteLevelBPETokenizer:
    tok = ByteLevelBPETokenizer()
    tok.train(CORPUS, vocab_size=300, special_tokens=[DOCUMENT_SEPARATOR])
    return tok


def _tiny_config(tokenizer: ByteLevelBPETokenizer) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=tokenizer.vocab_size, dim=16, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=32
    )


def _make_bundle(directory: Path) -> None:
    torch.manual_seed(0)
    tokenizer = _tiny_tokenizer()
    config = _tiny_config(tokenizer)
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)
    save_checkpoint_bundle(directory, trainer, tokenizer, config)


def test_save_and_load_checkpoint_bundle_round_trips_real_weights(tmp_path: Path):
    torch.manual_seed(0)
    tokenizer = _tiny_tokenizer()
    config = _tiny_config(tokenizer)
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)
    bundle_dir = tmp_path / "toy"
    save_checkpoint_bundle(bundle_dir, trainer, tokenizer, config)

    loaded_model, loaded_tokenizer, loaded_config = load_checkpoint_bundle(bundle_dir)

    assert loaded_config.vocab_size == config.vocab_size
    assert loaded_config.dim == config.dim
    assert loaded_tokenizer.encode("the sky is blue") == tokenizer.encode("the sky is blue")
    for key, original in model.state_dict().items():
        assert torch.equal(original, loaded_model.state_dict()[key]), f"weights diverged at {key}"


def test_save_checkpoint_bundle_config_json_does_not_destroy_the_previous_bundle_if_interrupted(
    tmp_path: Path, monkeypatch
):
    # A real bug found by actually simulating an interrupted write: the
    # bundle's config.json used to be written with a direct
    # path.write_text(), which truncates the file to 0 bytes the instant
    # it's opened -- before a single byte of new content is written. A
    # crash mid-write left model.pt/tokenizer.json (a real, potentially
    # expensive trained checkpoint) intact but the config.json needed to
    # actually reconstruct the model unreadable. Simulated here by
    # making os.replace() raise partway through re-saving the bundle.
    bundle_dir = tmp_path / "toy"
    _make_bundle(bundle_dir)
    good_config_bytes = (bundle_dir / "config.json").read_bytes()

    real_replace = os.replace

    def flaky_replace(src, dst):
        if str(dst).endswith("config.json"):
            raise OSError("simulated crash during os.replace")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    torch.manual_seed(1)
    tokenizer = _tiny_tokenizer()
    config = _tiny_config(tokenizer)
    trainer = Trainer(DecoderOnlyTransformer(config))
    with pytest.raises(OSError):
        save_checkpoint_bundle(bundle_dir, trainer, tokenizer, config)

    assert (bundle_dir / "config.json").read_bytes() == good_config_bytes
    _model, _tokenizer, loaded_config = load_checkpoint_bundle(bundle_dir)  # must not raise
    assert loaded_config.vocab_size == config.vocab_size


def test_save_and_load_checkpoint_bundle_round_trips_a_real_moe_config(tmp_path: Path):
    torch.manual_seed(0)
    tokenizer = _tiny_tokenizer()
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=32,
        moe=MoEConfig(n_experts=4, n_experts_per_tok=2, n_shared_experts=1),
    )
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)
    bundle_dir = tmp_path / "moe"
    save_checkpoint_bundle(bundle_dir, trainer, tokenizer, config)

    loaded_model, _tokenizer, loaded_config = load_checkpoint_bundle(bundle_dir)

    assert loaded_config.moe == config.moe
    for key, original in model.state_dict().items():
        assert torch.equal(original, loaded_model.state_dict()[key]), f"weights diverged at {key}"


def test_save_and_load_checkpoint_bundle_round_trips_a_real_rope_scaling_config(tmp_path: Path):
    torch.manual_seed(0)
    tokenizer = _tiny_tokenizer()
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=32,
        rope_scaling=RopeScalingConfig(method="ntk", factor=4.0),
    )
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)
    bundle_dir = tmp_path / "rope"
    save_checkpoint_bundle(bundle_dir, trainer, tokenizer, config)

    _loaded_model, _tokenizer, loaded_config = load_checkpoint_bundle(bundle_dir)

    assert loaded_config.rope_scaling == config.rope_scaling


def test_loading_a_bundle_saved_before_moe_rope_scaling_existed_still_works(tmp_path: Path):
    # A bundle written by an OLDER version of save_checkpoint_bundle has
    # no "moe"/"rope_scaling" keys in config.json at all -- real
    # backward compatibility, not just "the new fields default to None
    # in the dataclass," proven by hand-writing a config.json that
    # matches exactly what the pre-this-change code would have written.
    torch.manual_seed(0)
    tokenizer = _tiny_tokenizer()
    config = _tiny_config(tokenizer)
    model = DecoderOnlyTransformer(config)
    trainer = Trainer(model)
    bundle_dir = tmp_path / "legacy"
    bundle_dir.mkdir()
    trainer.save_checkpoint(bundle_dir / "model.pt")
    tokenizer.save(bundle_dir / "tokenizer.json")
    legacy_config_data = {
        "vocab_size": config.vocab_size,
        "dim": config.dim,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "max_seq_len": config.max_seq_len,
        "rope_theta": config.rope_theta,
        "norm_eps": config.norm_eps,
        "hidden_dim": config.hidden_dim,
    }
    (bundle_dir / "config.json").write_text(json.dumps(legacy_config_data))

    _model, _tokenizer, loaded_config = load_checkpoint_bundle(bundle_dir)

    assert loaded_config.moe is None
    assert loaded_config.rope_scaling is None


def test_load_checkpoint_bundle_is_cached_across_calls(tmp_path: Path, monkeypatch):
    # A real bug found by actually running FoundryProvider() in a loop
    # the way sarva.server.app's /chat and /ws/chat handlers do it (a
    # fresh FoundryProvider per request, so a saved config.json change
    # takes effect without a server restart): every request re-did a
    # full torch.load() of every configured checkpoint, forever, even
    # requests that ended up routed to a completely different provider.
    bundle_dir = tmp_path / "toy"
    _make_bundle(bundle_dir)

    calls = {"n": 0}
    real_torch_load = torch.load

    def counting_load(*args, **kwargs):
        calls["n"] += 1
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_load)

    for _ in range(5):
        FoundryProvider(tmp_path)

    assert calls["n"] == 1


def test_load_checkpoint_bundle_cache_invalidates_on_a_retrained_checkpoint(tmp_path: Path):
    # The cache is keyed on model.pt's own mtime, not just the directory
    # path -- a checkpoint retrained and re-saved in place must be
    # picked up on its very next load, not silently serve stale weights
    # forever just because a FoundryProvider once loaded that path.
    bundle_dir = tmp_path / "toy"
    _make_bundle(bundle_dir)
    first_model, _tokenizer, _config = load_checkpoint_bundle(bundle_dir)
    first_weight = next(iter(first_model.state_dict().values())).clone()

    model_pt = bundle_dir / "model.pt"
    os.utime(model_pt, (time.time() + 5, time.time() + 5))  # force a distinguishable mtime
    torch.manual_seed(999)
    retrained = DecoderOnlyTransformer(_config)
    retrained_trainer = Trainer(retrained)
    retrained_trainer.save_checkpoint(model_pt)

    second_model, _tokenizer, _config = load_checkpoint_bundle(bundle_dir)
    second_weight = next(iter(second_model.state_dict().values())).clone()

    assert not torch.equal(first_weight, second_weight)


def test_load_checkpoint_bundle_cache_evicts_the_superseded_entry(tmp_path: Path):
    # A real bug found by giving this cache its own fresh-eyes sweep,
    # applying the project's own "does yesterday's fix have a bug" lens
    # to the fix that added it: keyed on (directory, mtime) so a
    # retrained checkpoint is picked up fresh (see the test above), but
    # nothing ever evicted the OLD entry once a new one landed for the
    # same directory. Confirmed live: repeated retrain/re-save/reload
    # cycles against one checkpoint directory -- the exact "no server
    # restart needed" workflow this cache exists to support -- left
    # every superseded model copy resident in memory simultaneously,
    # even though only the newest is ever reachable again through
    # ordinary use. A directory only ever has one current mtime at a
    # time, so an entry under a different mtime for the same resolved
    # path is permanently unreachable the moment a new one lands.
    from sarva.providers.foundry_provider import _bundle_cache

    bundle_dir = tmp_path / "toy"
    _make_bundle(bundle_dir)
    tokenizer = _tiny_tokenizer()
    config = _tiny_config(tokenizer)
    model_pt = bundle_dir / "model.pt"

    for i in range(5):
        os.utime(model_pt, (time.time() + i + 1, time.time() + i + 1))
        Trainer(DecoderOnlyTransformer(config)).save_checkpoint(model_pt)
        load_checkpoint_bundle(bundle_dir)

    entries_for_this_dir = [k for k in _bundle_cache if k[0] == str(bundle_dir.resolve())]
    assert len(entries_for_this_dir) == 1, (
        f"expected exactly 1 cache entry for this directory after 5 "
        f"retrain/reload cycles, found {len(entries_for_this_dir)} -- the old ones "
        "were never evicted"
    )


def test_concurrent_loads_of_the_same_cold_bundle_never_redo_the_expensive_load(tmp_path: Path):
    # A real bug found by a fresh-eyes sweep, applying the exact lens
    # that already caught the identical shape one module over in
    # runtime.py's _probe_ollama (round 128): the cache's read-check and
    # write-back are two separate, unsynchronized steps with the real,
    # expensive torch.load() in between -- a classic unguarded
    # check-then-act race. build_providers() runs load_checkpoint_bundle
    # on a real OS worker thread via asyncio.to_thread for every
    # /chat, /ws/chat, and diagnostics-driving endpoint, so genuine
    # concurrent requests against a cold cache -- not a contrived
    # scenario -- used to all redo the full weight load instead of the
    # 1 the cache exists to guarantee. Proven here with real OS threads
    # racing load_checkpoint_bundle() against the SAME bundle, a
    # threading.Barrier forcing them through the read-check at the same
    # instant, and a slowed, counting torch.load standing in for real
    # weight-load I/O.
    bundle_dir = tmp_path / "toy"
    _make_bundle(bundle_dir)

    calls = {"n": 0}
    calls_lock = threading.Lock()
    real_torch_load = torch.load

    def slow_counting_load(*args, **kwargs):
        with calls_lock:
            calls["n"] += 1
        time.sleep(0.05)  # simulate real, slow weight-load I/O
        return real_torch_load(*args, **kwargs)

    torch.load = slow_counting_load
    try:
        n_callers = 8
        barrier = threading.Barrier(n_callers)

        def load_one():
            barrier.wait()
            load_checkpoint_bundle(bundle_dir)

        threads = [threading.Thread(target=load_one) for _ in range(n_callers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        torch.load = real_torch_load

    assert calls["n"] == 1, (
        f"expected exactly 1 real torch.load call for {n_callers} concurrent callers "
        f"against one cold, unchanged bundle, got {calls['n']} -- the cache's own "
        "guarantee of loading each checkpoint once was defeated by a race"
    )


def test_discover_checkpoint_bundles_finds_only_complete_bundles(tmp_path: Path):
    _make_bundle(tmp_path / "real")
    (tmp_path / "incomplete").mkdir()
    (tmp_path / "incomplete" / "config.json").write_text("{}")  # missing tokenizer.json/model.pt

    found = discover_checkpoint_bundles(tmp_path)

    assert set(found) == {"real"}
    assert found["real"] == tmp_path / "real"


def test_discover_checkpoint_bundles_on_missing_directory_returns_empty(tmp_path: Path):
    assert discover_checkpoint_bundles(tmp_path / "does-not-exist") == {}


def test_model_info_for_bundle_reads_config_without_touching_torch(tmp_path: Path):
    _make_bundle(tmp_path / "toy")
    info = model_info_for_bundle("toy", tmp_path / "toy")

    assert info.id == "foundry/toy"
    assert info.provider == "foundry"
    assert info.local is True
    assert info.capabilities.modalities_in == {Modality.TEXT}
    assert info.capabilities.context_window == 32  # max_seq_len from _tiny_config
    assert info.cost.input_per_mtok == 0.0


def test_foundry_provider_construction_fails_clearly_on_an_empty_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="no valid foundry checkpoint bundles"):
        FoundryProvider(tmp_path)


def test_foundry_provider_skips_a_corrupted_bundle_instead_of_crashing_construction(
    tmp_path: Path,
):
    # A real bug found by actually corrupting a real bundle's model.pt
    # (truncating it, simulating an interrupted save or disk
    # corruption) and calling the real build_providers(): torch.load()
    # raised an uncaught OSError, and since FoundryProvider.__init__
    # used to be a plain dict comprehension over every discovered
    # bundle with no error handling, one bad bundle crashed
    # construction entirely -- and therefore every caller of
    # build_providers() (every CLI command, /chat, /ws/chat, server
    # startup), not just a request that tried to use that specific
    # checkpoint. discover_checkpoint_bundles only checks that the
    # three bundle files *exist*, never that they're actually valid.
    _make_bundle(tmp_path / "good")
    bad_dir = tmp_path / "bad"
    _make_bundle(bad_dir)
    model_pt = bad_dir / "model.pt"
    raw = model_pt.read_bytes()
    model_pt.write_bytes(raw[: len(raw) // 2])  # truncated, not just malformed

    provider = FoundryProvider(tmp_path)

    assert "good" in provider._loaded
    assert "bad" not in provider._loaded
    assert "bad" in provider.broken_bundles
    assert "good" not in provider.broken_bundles


def test_foundry_provider_construction_fails_clearly_when_every_bundle_is_broken(
    tmp_path: Path,
):
    bad_dir = tmp_path / "bad"
    _make_bundle(bad_dir)
    model_pt = bad_dir / "model.pt"
    raw = model_pt.read_bytes()
    model_pt.write_bytes(raw[: len(raw) // 2])

    with pytest.raises(ValueError, match="none could actually be loaded"):
        FoundryProvider(tmp_path)


async def test_foundry_provider_generate_produces_a_real_completion(tmp_path: Path):
    _make_bundle(tmp_path / "toy")
    provider = FoundryProvider(tmp_path)

    request = GenerateRequest(
        model="foundry/toy",
        messages=[Message(role="user", content=[TextBlock(text="the quick brown")])],
        config=GenerateConfig(max_tokens=8),
    )

    events = [event async for event in provider.generate(request)]
    done = events[-1]

    assert done.type == "done"
    assert done.stop_reason in (StopReason.END_TURN, StopReason.MAX_TOKENS)
    assert done.usage.input_tokens > 0
    # The DoneEvent's own message text must match whatever text deltas
    # were actually streamed -- not just internally consistent shapes.
    streamed_text = "".join(e.text for e in events if e.type == "text_delta")
    assert done.message.text() == streamed_text
    await provider.close()


async def test_foundry_provider_generate_handles_an_empty_message_cleanly(tmp_path: Path):
    # A real bug found by actually sending an empty message: nothing
    # upstream validates non-empty text (`chat()`'s CLI argument and
    # ChatRequest.message both accept ""), and ByteLevelBPETokenizer.
    # encode("") returns [] -- an empty prompt_ids list used to sail past
    # the budget<=0 guard (still positive with zero prompt tokens)
    # straight into generate_with_cache, where torch.tensor([[]]) has no
    # elements to infer a dtype from and defaults to float32 instead of
    # int64, crashing the token embedding lookup with a raw
    # RuntimeError ("Expected tensor for argument #1 'indices' ... Long,
    # Int; but got torch.FloatTensor"). Confirmed live before this fix.
    _make_bundle(tmp_path / "toy")
    provider = FoundryProvider(tmp_path)
    request = GenerateRequest(
        model="foundry/toy",
        messages=[Message(role="user", content=[TextBlock(text="")])],
        config=GenerateConfig(max_tokens=8),
    )

    events = [event async for event in provider.generate(request)]  # must not raise
    done = events[-1]

    assert done.type == "done"
    assert done.message.text() == ""
    assert done.usage.input_tokens == 0
    await provider.close()


async def test_foundry_provider_generate_rejects_an_unknown_model_id(tmp_path: Path):
    _make_bundle(tmp_path / "toy")
    provider = FoundryProvider(tmp_path)
    request = GenerateRequest(
        model="foundry/nonexistent",
        messages=[Message(role="user", content=[TextBlock(text="hi")])],
    )
    with pytest.raises(ModelNotFoundError):
        async for _ in provider.generate(request):
            pass


async def test_foundry_provider_raises_instead_of_silently_dropping_an_image(tmp_path: Path):
    # foundry checkpoints are text-only end to end (modalities_in={TEXT},
    # tool_use=False) -- Message.text() would otherwise silently drop an
    # ImageBlock, answering as if it had never been sent. Reachable only
    # via an explicit model override (the router's own modality check
    # would never route an image-bearing request here on its own), same
    # reachability note the Anthropic/OpenAI/Google adapters' own
    # untranslatable-block-type guards carry.
    from sarva.multimodal.content import ImageBlock

    _make_bundle(tmp_path / "toy")
    provider = FoundryProvider(tmp_path)
    request = GenerateRequest(
        model="foundry/toy",
        messages=[
            Message(
                role="user",
                content=[
                    TextBlock(text="what's in this image?"),
                    ImageBlock(media_type="image/png", data=b"\x89PNG\r\n\x1a\n"),
                ],
            )
        ],
    )
    with pytest.raises(ValueError, match="ImageBlock"):
        async for _ in provider.generate(request):
            pass


async def test_foundry_provider_is_gradable_through_the_real_eval_harness(tmp_path: Path):
    # eval/harness.py's own module docstring makes a direct claim: the
    # same run_benchmark() call "will grade a foundry-trained model too
    # the moment ... a real 'foundry adapter' ... exists." That adapter
    # has existed since a prior milestone, but nothing had ever actually
    # run a FoundryProvider through run_benchmark() in this automated
    # suite -- test_eval_harness.py only ever exercises MockProvider,
    # framing everything else as "live-only, exercised by whoever runs
    # `sarva eval` with a configured key." Foundry doesn't belong in
    # that bucket: unlike Anthropic/OpenAI/Google, it needs no API key
    # or network, so there was no real reason this had to stay
    # hand-verified-once instead of a real, permanent regression test.
    from sarva.eval.benchmarks import ARITHMETIC
    from sarva.eval.harness import run_benchmark

    _make_bundle(tmp_path / "toy")
    provider = FoundryProvider(tmp_path)

    report = await run_benchmark(ARITHMETIC, provider, model="foundry/toy")

    assert report.model == "foundry/toy"
    assert len(report.results) == len(ARITHMETIC.cases)
    # An untrained toy checkpoint getting arithmetic right would be the
    # real red flag here -- the honest, expected result is 0%, the same
    # no-fabrication discipline this project already applies to the
    # zero-config Mock provider's own eval score.
    assert report.accuracy == 0.0
    await provider.close()


def test_registry_register_adds_a_dynamic_entry_without_touching_static_ones(tmp_path: Path):
    _make_bundle(tmp_path / "toy")
    static_info = model_info_for_bundle("static", tmp_path / "toy")
    registry = Registry({static_info.id: static_info})

    dynamic_info = model_info_for_bundle("toy", tmp_path / "toy")
    registry.register(dynamic_info)

    assert {m.id for m in registry.all()} == {static_info.id, dynamic_info.id}
    assert registry.get(dynamic_info.id) == dynamic_info
    assert registry.get(static_info.id) == static_info


def test_runtime_wires_a_foundry_checkpoint_into_router_and_providers(tmp_path, monkeypatch):
    _make_bundle(tmp_path / "toy")
    monkeypatch.setenv("SARVA_FOUNDRY_CHECKPOINTS", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    import sarva.runtime as runtime

    monkeypatch.setattr(runtime, "ollama_reachable", lambda *a, **kw: False)

    router = runtime.build_router()
    providers = runtime.build_providers()

    assert "foundry/toy" in router.available
    assert router.registry.get("foundry/toy").provider == "foundry"
    assert "foundry" in providers
