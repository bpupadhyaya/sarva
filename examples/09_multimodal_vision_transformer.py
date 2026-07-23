"""Example 09 — Native multimodal input: vision encoder + projector +
text decoder, trained end to end.

The last named piece of §3.6a's architecture list (MoE and long-context
RoPE scaling came first — examples 07 and 08). Wires together
`sarva_foundry.model.vision`'s `VisionEncoder`/`Projector` with the same
`DecoderOnlyTransformer` every other example uses, via
`forward_multimodal`: image patches get embedded and projected into the
text model's embedding space, spliced into a token sequence at
placeholder positions, and the causal decoder processes the whole
sequence as one unified stream — exactly the LLaVA-class pattern this
project's docs describe.

The "task" is deliberately trivial and synthetic (a solid-color image
paired with a token sequence that should predict a specific answer
token) — the point isn't a useful vision-language model, it's proving
gradients genuinely flow through the *entire* stack: vision encoder ->
projector -> text decoder, and that loss actually decreases when they
do.

Run: uv run python examples/09_multimodal_vision_transformer.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sarva_foundry.model import Projector, TransformerConfig, VisionEncoder, VisionEncoderConfig
from sarva_foundry.model.transformer import DecoderOnlyTransformer

IMAGE_TOKEN_ID = 29  # reserved vocab id standing in for one image patch each
VOCAB_SIZE = 30  # 0-28 ordinary tokens, 29 = IMAGE_TOKEN_ID


def main() -> None:
    torch.manual_seed(0)

    vision_config = VisionEncoderConfig(
        image_size=8, patch_size=4, dim=32, n_layers=2, n_heads=4, n_kv_heads=2
    )
    text_config = TransformerConfig(
        vocab_size=VOCAB_SIZE, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=16
    )
    encoder = VisionEncoder(vision_config)
    projector = Projector(vision_dim=vision_config.dim, text_dim=text_config.dim)
    model = DecoderOnlyTransformer(text_config)
    print(
        f"Vision encoder: {sum(p.numel() for p in encoder.parameters()):,} params, "
        f"{vision_config.n_patches} patches/image"
    )
    print(f"Text decoder:   {model.num_parameters():,} params")

    # A trivial but real task: a solid RED image should be followed by
    # token 5; a solid BLUE image should be followed by token 10. The
    # model has to actually use the image to know which -- there's no
    # way to guess correctly from the text tokens alone.
    red_image = torch.zeros(1, 3, 8, 8)
    red_image[:, 0, :, :] = 1.0  # channel 0 = red
    blue_image = torch.zeros(1, 3, 8, 8)
    blue_image[:, 2, :, :] = 1.0  # channel 2 = blue

    n_patches = vision_config.n_patches
    red_tokens = torch.cat([torch.full((1, n_patches), IMAGE_TOKEN_ID), torch.tensor([[1]])], dim=1)
    blue_tokens = torch.cat(
        [torch.full((1, n_patches), IMAGE_TOKEN_ID), torch.tensor([[1]])], dim=1
    )
    red_target, blue_target = 5, 10

    all_params = (
        list(encoder.parameters()) + list(projector.parameters()) + list(model.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=3e-3)

    print("\nTraining: red image -> predict token 5, blue image -> predict token 10...")
    for step in range(150):
        total_loss = torch.zeros(())
        for image, tokens, target in [
            (red_image, red_tokens, red_target),
            (blue_image, blue_tokens, blue_target),
        ]:
            image_embeds = projector(encoder(image))
            logits = model.forward_multimodal(tokens, image_embeds, IMAGE_TOKEN_ID)
            last_logits = logits[:, -1, :]  # predict the token AFTER the prompt
            loss = F.cross_entropy(last_logits, torch.tensor([target]))
            total_loss = total_loss + loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if step % 30 == 0 or step == 149:
            print(f"  step {step:3d}  total loss {total_loss.item():.4f}")

    with torch.no_grad():
        red_pred = (
            model.forward_multimodal(red_tokens, projector(encoder(red_image)), IMAGE_TOKEN_ID)[
                :, -1, :
            ]
            .argmax()
            .item()
        )
        blue_pred = (
            model.forward_multimodal(blue_tokens, projector(encoder(blue_image)), IMAGE_TOKEN_ID)[
                :, -1, :
            ]
            .argmax()
            .item()
        )

    print(
        f"\nAfter training: red image -> predicted token {red_pred} (target {red_target}), "
        f"blue image -> predicted token {blue_pred} (target {blue_target})"
    )
    print(
        "The model can only get both right by actually using the image content -- "
        "the text tokens are identical in both cases."
    )


if __name__ == "__main__":
    main()
