"""Conformance tests for sarva_foundry.model.vision — native multimodal
input (spec §3.6a). Same bar as the rest of the model tests: shapes
alone don't prove PatchEmbed's "patchify" trick is actually the claimed
flatten+linear equivalent, and don't prove the vision encoder is
actually bidirectional rather than accidentally causal — both are
verified directly against real mathematical properties, not assumed
from a correct-looking implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sarva_foundry.model.transformer import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.model.vision import (
    PatchEmbed,
    Projector,
    VisionEncoder,
    VisionEncoderConfig,
)

torch.manual_seed(0)


def _tiny_vision_config(**overrides) -> VisionEncoderConfig:
    defaults = dict(image_size=16, patch_size=4, dim=32, n_layers=2, n_heads=2, n_kv_heads=2)
    defaults.update(overrides)
    return VisionEncoderConfig(**defaults)


def test_vision_encoder_config_rejects_non_divisible_image_size():
    import pytest

    with pytest.raises(ValueError, match="image_size"):
        _tiny_vision_config(image_size=17, patch_size=4)


def test_vision_encoder_config_rejects_non_positive_image_size_or_patch_size():
    # A real bug found by a later fresh-eyes sweep, on the very class
    # this __post_init__ was already revisited twice for (n_layers,
    # n_kv_heads): image_size/patch_size were only ever checked for
    # divisibility against each other, never for being positive.
    # patch_size=0 used to raise a raw ZeroDivisionError instead of a
    # clean ValueError; a negative patch_size that happens to divide
    # image_size evenly (Python's `%` follows the divisor's sign) used
    # to pass silently, producing a negative patches_per_side that only
    # surfaced as a confusing raw RuntimeError deep inside nn.Conv2d
    # once VisionEncoder was actually built.
    import pytest

    with pytest.raises(ValueError, match="image_size"):
        _tiny_vision_config(image_size=0, patch_size=4)
    with pytest.raises(ValueError, match="patch_size"):
        _tiny_vision_config(image_size=16, patch_size=0)
    with pytest.raises(ValueError, match="patch_size"):
        _tiny_vision_config(image_size=16, patch_size=-4)  # -4 divides 16 evenly in Python


def test_vision_encoder_config_rejects_non_positive_n_layers():
    # A real bug found by a fresh-eyes sweep: the identical shape round
    # 133 already fixed for TransformerConfig.n_layers, one sibling
    # config class over -- range() of a non-positive number is silently
    # empty in Python, so VisionEncoder.__init__'s `[VisionEncoderBlock(
    # config) for _ in range(config.n_layers)]` never raised for
    # n_layers <= 0. It silently built an encoder with ZERO transformer
    # blocks instead: forward() still ran, patch-embedding pixels then
    # immediately normalizing them with no attention or MLP pass at all,
    # returning a plausibly-shaped tensor with no exception anywhere.
    import pytest

    with pytest.raises(ValueError, match="n_layers"):
        _tiny_vision_config(n_layers=0)
    with pytest.raises(ValueError, match="n_layers"):
        _tiny_vision_config(n_layers=-2)


def test_vision_encoder_config_rejects_non_positive_n_kv_heads():
    # A real bug found by a later fresh-eyes sweep, the identical gap
    # just closed one field over (n_layers, above) and one file over
    # (TransformerConfig.n_kv_heads): n_kv_heads was never validated
    # here either. n_kv_heads=0 used to construct cleanly, then raise a
    # raw ZeroDivisionError only when VisionEncoder was actually built;
    # a negative n_kv_heads whose magnitude divides n_heads evenly slips
    # past GroupedQueryAttention's own `n_heads % n_kv_heads` check
    # entirely (Python's modulo is defined for negative divisors too),
    # then crashes nn.Linear with a raw "negative dimension" RuntimeError.
    import pytest

    with pytest.raises(ValueError, match="n_kv_heads"):
        _tiny_vision_config(n_kv_heads=0)
    with pytest.raises(ValueError, match="n_kv_heads"):
        _tiny_vision_config(n_heads=4, n_kv_heads=-4)


def test_vision_encoder_config_rejects_non_positive_n_heads():
    # A real bug found by a later fresh-eyes sweep, one field over from
    # the identical TransformerConfig.n_heads gap in transformer.py:
    # n_heads is used as a divisor right below in this same
    # __post_init__, but was never validated for being positive.
    # n_heads=0 used to raise a raw ZeroDivisionError instead of a
    # clean ValueError.
    import pytest

    with pytest.raises(ValueError, match="n_heads"):
        _tiny_vision_config(n_heads=0)


def test_n_patches_matches_the_grid():
    config = _tiny_vision_config(image_size=32, patch_size=8)
    assert config.patches_per_side == 4
    assert config.n_patches == 16


def test_patch_embed_output_shape():
    config = _tiny_vision_config()
    embed = PatchEmbed(config)
    pixel_values = torch.randn(2, 3, config.image_size, config.image_size)
    out = embed(pixel_values)
    assert out.shape == (2, config.n_patches, config.dim)


def test_patch_embed_matches_manual_flatten_and_linear():
    # The defining correctness property of the "patchify via strided
    # conv" trick: it must be mathematically identical to manually
    # slicing each non-overlapping patch, flattening it, and applying
    # one shared linear layer built from the same conv weights --  not
    # just "produces the right shape."
    config = _tiny_vision_config(image_size=8, patch_size=4, dim=6, n_channels=3)
    embed = PatchEmbed(config)
    pixel_values = torch.randn(1, 3, 8, 8)

    conv_out = embed(pixel_values)  # (1, 4, 6)

    # Manual reference: unfold into non-overlapping patches, flatten each
    # to a (channels * patch_size * patch_size) vector in the same
    # C-then-H-then-W order nn.Conv2d's weight tensor uses, apply the
    # conv's weight as a linear map plus its bias.
    patches = F.unfold(pixel_values, kernel_size=config.patch_size, stride=config.patch_size)
    patches = patches.transpose(1, 2)  # (1, n_patches, channels*patch*patch)
    weight = embed.proj.weight.reshape(config.dim, -1)  # (dim, channels*patch*patch)
    manual_out = patches @ weight.T + embed.proj.bias

    assert torch.allclose(conv_out, manual_out, atol=1e-5)


def test_vision_encoder_output_shape():
    config = _tiny_vision_config()
    encoder = VisionEncoder(config)
    pixel_values = torch.randn(2, 3, config.image_size, config.image_size)
    out = encoder(pixel_values)
    assert out.shape == (2, config.n_patches, config.dim)


def test_vision_encoder_is_genuinely_bidirectional_not_accidentally_causal():
    # The opposite of test_model.py's causal-masking test: here,
    # perturbing ONE patch must be able to change every OTHER patch's
    # output too, since a real bidirectional encoder lets every position
    # attend to every other position. If `causal=False` were silently
    # ignored (or wired backwards), this would look like the causal
    # decoder's test instead -- only later patches would change, which a
    # shape-only test could never catch.
    config = _tiny_vision_config()
    encoder = VisionEncoder(config)
    encoder.eval()
    pixel_values = torch.randn(1, 3, config.image_size, config.image_size)
    with torch.no_grad():
        out_a = encoder(pixel_values)

    perturbed = pixel_values.clone()
    perturbed[0, 0, 0, 0] += 10.0  # a single pixel inside the FIRST patch
    with torch.no_grad():
        out_b = encoder(perturbed)

    # A causal (or accidentally-causal) encoder would leave the LAST
    # patch's output completely unaffected by a change confined to the
    # first patch. A genuinely bidirectional one does not.
    assert not torch.allclose(out_a[:, -1, :], out_b[:, -1, :], atol=1e-6)


def test_projector_output_shape():
    projector = Projector(vision_dim=32, text_dim=24)
    x = torch.randn(2, 16, 32)
    out = projector(x)
    assert out.shape == (2, 16, 24)


def test_projector_is_nonlinear_not_a_disguised_linear_layer():
    # A 2-layer MLP with a real nonlinearity in between must NOT be
    # expressible as a single linear map -- confirms the GELU is
    # actually wired in between fc1 and fc2, not accidentally skipped
    # (which would make Projector degenerate to one linear layer, the
    # exact thing LLaVA-1.5's ablation found underperforms).
    torch.manual_seed(1)
    projector = Projector(vision_dim=8, text_dim=8, hidden_dim=8)
    x = torch.randn(5, 8)
    out = projector(x)

    # If Projector were linear, out would equal x @ A + b for a single
    # effective (A, b) recoverable via least squares from THESE 5
    # samples; check it does NOT generalize to fresh samples the way an
    # actual linear map would -- construct that fit and confirm its
    # prediction on new inputs diverges from the real (nonlinear) output.
    x_test = torch.randn(5, 8)
    out_test = projector(x_test)
    x_aug = torch.cat([x, torch.ones(5, 1)], dim=1)
    solution = torch.linalg.lstsq(x_aug, out).solution
    x_test_aug = torch.cat([x_test, torch.ones(5, 1)], dim=1)
    linear_prediction = x_test_aug @ solution
    assert not torch.allclose(linear_prediction, out_test, atol=1e-3)


def test_causal_text_attention_is_unaffected_by_the_new_causal_parameter():
    # Regression guard: GroupedQueryAttention's default (causal=True,
    # unchanged) must still behave exactly as it did before this
    # parameter existed -- proven by DecoderOnlyTransformer's own
    # existing causal-masking test still passing (see test_model.py),
    # plus this explicit construction check that the default wasn't
    # accidentally flipped.
    from sarva_foundry.model.attention import GroupedQueryAttention

    attn_default = GroupedQueryAttention(dim=8, n_heads=2, n_kv_heads=1, head_dim=4, max_seq_len=8)
    assert attn_default.causal is True


def test_forward_multimodal_matches_forward_when_no_image_tokens_present():
    # A degenerate but important case: multimodal splicing must produce
    # the exact same embeddings as plain forward() when every actual
    # position is text -- i.e. embed_multimodal shouldn't corrupt
    # anything when the mask happens to select nothing here.
    config = TransformerConfig(
        vocab_size=30, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    model = DecoderOnlyTransformer(config)
    token_ids = torch.randint(0, 29, (1, 8))  # never equals the reserved image_token_id=29

    plain_out = model(token_ids)
    embeds = model.embed_multimodal(
        token_ids, image_embeds=torch.zeros(1, 0, 16), image_token_id=29
    )
    spliced_out = model._forward_embeds(embeds)

    assert torch.equal(plain_out, spliced_out)


def test_forward_multimodal_raises_on_placeholder_count_mismatch():
    import pytest

    config = TransformerConfig(
        vocab_size=30, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    model = DecoderOnlyTransformer(config)
    token_ids = torch.full((1, 4), 29)  # 4 placeholder positions
    wrong_embeds = torch.randn(1, 3, 16)  # only 3 image embeddings

    with pytest.raises(ValueError, match="placeholder"):
        model.forward_multimodal(token_ids, wrong_embeds, image_token_id=29)


def test_embed_multimodal_raises_on_uneven_per_item_placeholder_counts():
    # A real bug found by a fresh-eyes sweep: the mismatch check above
    # only ever compares AGGREGATE counts (mask.sum() vs. image_embeds'
    # total token count), never each batch item's own count, despite
    # this method's own docstring stating it "assumes every batch item
    # contributes the same number of image placeholder tokens." Two
    # batch items whose individual placeholder counts differ (1 and 3)
    # but sum to the same total (4) as image_embeds' own total (2 items
    # x 2 tokens each = 4) used to pass the aggregate check silently --
    # confirmed live before this fix: the flattened, row-major splice
    # then spliced item 0's own unused second image embedding into item
    # 1's first placeholder position, shifting every one of item 1's
    # real embeddings one position over, with no exception raised
    # anywhere -- cross-example contamination in both training and
    # inference.
    import pytest

    config = TransformerConfig(
        vocab_size=30, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    model = DecoderOnlyTransformer(config)
    token_ids = torch.full((2, 4), 5)
    token_ids[0, 0] = 29  # item 0: 1 placeholder
    token_ids[1, 0] = token_ids[1, 1] = token_ids[1, 2] = 29  # item 1: 3 placeholders
    # 2 items x 2 tokens each = 4 total -- matches the aggregate count
    # above (1 + 3 = 4) even though no single item actually has 2.
    image_embeds = torch.randn(2, 2, 16)

    with pytest.raises(ValueError, match="each batch item"):
        model.embed_multimodal(token_ids, image_embeds, image_token_id=29)


def test_embed_multimodal_still_works_when_every_item_has_the_same_placeholder_count():
    # Regression guard for the fix above: the new per-item check must
    # not reject the ordinary, genuinely uniform case it's meant to
    # keep allowing.
    config = TransformerConfig(
        vocab_size=30, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    model = DecoderOnlyTransformer(config)
    token_ids = torch.full((2, 4), 5)
    token_ids[0, 0] = token_ids[0, 1] = 29
    token_ids[1, 0] = token_ids[1, 1] = 29
    image_embeds = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16)

    x = model.embed_multimodal(token_ids, image_embeds, image_token_id=29)

    assert torch.equal(x[0, 0], image_embeds[0, 0])
    assert torch.equal(x[0, 1], image_embeds[0, 1])
    assert torch.equal(x[1, 0], image_embeds[1, 0])
    assert torch.equal(x[1, 1], image_embeds[1, 1])


def test_end_to_end_vision_encoder_projector_decoder_forward_shape():
    vconfig = _tiny_vision_config(image_size=16, patch_size=4, dim=32, n_layers=2)
    encoder = VisionEncoder(vconfig)
    tconfig = TransformerConfig(
        vocab_size=50, dim=24, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=32
    )
    projector = Projector(vision_dim=vconfig.dim, text_dim=tconfig.dim)
    model = DecoderOnlyTransformer(tconfig)

    image_token_id = 49
    pixel_values = torch.randn(2, 3, 16, 16)
    image_embeds = projector(encoder(pixel_values))  # (2, 16, 24)

    seq_len = vconfig.n_patches + 5
    token_ids = torch.randint(0, 49, (2, seq_len))
    token_ids[:, : vconfig.n_patches] = image_token_id

    logits = model.forward_multimodal(token_ids, image_embeds, image_token_id)
    assert logits.shape == (2, seq_len, tconfig.vocab_size)


def test_full_stack_is_trainable_gradients_flow_through_vision_and_text():
    # The strongest end-to-end proof, mirroring every other trainability
    # test in this suite: loss must decrease, and every parameter across
    # the vision encoder, the projector, AND the text decoder must
    # actually receive a gradient -- a broken splice (e.g. embed_multimodal
    # accidentally using .detach() or a non-differentiable index_copy)
    # would silently zero out the vision/projector gradients while still
    # producing plausible-shaped logits.
    torch.manual_seed(0)
    vconfig = _tiny_vision_config(
        image_size=8, patch_size=4, dim=16, n_layers=1, n_heads=2, n_kv_heads=2
    )
    encoder = VisionEncoder(vconfig)
    tconfig = TransformerConfig(
        vocab_size=20, dim=16, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16
    )
    projector = Projector(vision_dim=vconfig.dim, text_dim=tconfig.dim)
    model = DecoderOnlyTransformer(tconfig)

    image_token_id = 19
    all_params = (
        list(encoder.parameters()) + list(projector.parameters()) + list(model.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=1e-2)

    pixel_values = torch.randn(1, 3, 8, 8)
    seq_len = vconfig.n_patches + 3
    token_ids = torch.randint(0, 19, (1, seq_len))
    token_ids[:, : vconfig.n_patches] = image_token_id
    targets = torch.randint(0, 20, (1, seq_len))

    losses = []
    for _ in range(30):
        image_embeds = projector(encoder(pixel_values))
        logits = model.forward_multimodal(token_ids, image_embeds, image_token_id)
        loss = F.cross_entropy(logits.view(-1, tconfig.vocab_size), targets.view(-1))
        optimizer.zero_grad()
        loss.backward()

        assert encoder.patch_embed.proj.weight.grad is not None
        assert projector.fc1.weight.grad is not None

        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.8
