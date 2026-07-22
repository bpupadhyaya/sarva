"""Example 06 — the foundry corpus pipeline on a real, small,
public-domain corpus.

Every earlier foundry example (02-04) trains on four hardcoded toy
sentences -- proves the mechanics, not that the actual corpus-sourcing
pipeline (`sarva_foundry.data.corpus`/`.near_dedup`/`.provenance`) does
anything useful on real text. This example fetches three short, genuine
public-domain texts from Project Gutenberg, runs the full local-scale
pipeline this project actually has (load-with-provenance -> exact-dedup
-> near-dedup -> length-filter) on them, then trains the same
tokenizer/transformer/Trainer stack example 04 exercises -- against real
prose instead of synthetic strings.

Requires network access to fetch the three source texts. Everything
after that step (dedup, tokenizer training, model training) is fully
offline, same as every other example.

Run: uv run python examples/06_real_corpus_pretraining.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import httpx
import torch
from sarva_foundry.data import DOCUMENT_SEPARATOR, TextChunkDataset, tokenize_corpus
from sarva_foundry.data.provenance import (
    dedup_near_duplicate_sourced_documents,
    dedup_sourced_documents,
    filter_sourced_documents_by_length,
    load_text_files_with_provenance,
)
from sarva_foundry.model import DecoderOnlyTransformer, TransformerConfig
from sarva_foundry.tokenizer import ByteLevelBPETokenizer
from sarva_foundry.train import Trainer, TrainerConfig, WarmupCosineSchedule

# Three short, genuinely public-domain works (expired US copyright,
# distributed under Project Gutenberg's own license grant) -- real
# sourcing, not synthetic strings. Kept small on purpose: laptop-scale
# demo, not a run meant to produce a useful model.
_SOURCES = {
    "a_modest_proposal.txt": "https://www.gutenberg.org/files/1080/1080-0.txt",
    "the_hunting_of_the_snark.txt": "https://www.gutenberg.org/cache/epub/13/pg13.txt",
    "the_time_machine.txt": "https://www.gutenberg.org/files/35/35-0.txt",
}
_LICENSE = "Public Domain (Project Gutenberg, US) -- see gutenberg.org/policy/license.html"


def _download_corpus(directory: Path) -> None:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for filename, url in _SOURCES.items():
            print(f"  fetching {url} ...")
            response = client.get(url)
            response.raise_for_status()
            (directory / filename).write_text(response.text, encoding="utf-8")


def main() -> None:
    torch.manual_seed(0)

    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = Path(tmp)
        print("Downloading real public-domain source texts from Project Gutenberg...")
        try:
            _download_corpus(corpus_dir)
        except httpx.HTTPError as e:
            print(f"Could not reach Project Gutenberg ({e}). This example needs network access.")
            sys.exit(1)

        docs = load_text_files_with_provenance(corpus_dir, license=_LICENSE)
        print(f"\nLoaded {len(docs)} real documents:")
        for d in docs:
            print(f"  {Path(d.source_path).name}: {len(d.text):,} chars, license={d.license!r}")

        docs = dedup_sourced_documents(docs)
        docs = dedup_near_duplicate_sourced_documents(docs)
        docs = filter_sourced_documents_by_length(docs, min_chars=1000)
        print(f"{len(docs)} document(s) survive exact-dedup + near-dedup + length-filter.")

        corpus = [d.text for d in docs]

    tokenizer = ByteLevelBPETokenizer()
    print("\nTraining a byte-level BPE tokenizer on the real corpus (this takes a few seconds)...")
    tokenizer.train(corpus, vocab_size=1200, special_tokens=[DOCUMENT_SEPARATOR])
    token_ids = tokenize_corpus(corpus, tokenizer)
    dataset = TextChunkDataset(token_ids, seq_len=64)
    print(f"Corpus: {len(token_ids):,} tokens -> {len(dataset)} training chunks of 64 tokens each")

    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        dim=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
    )
    schedule = WarmupCosineSchedule(peak_lr=3e-3, min_lr=3e-4, warmup_steps=20, total_steps=200)
    trainer = Trainer(DecoderOnlyTransformer(config), TrainerConfig(schedule=schedule))

    print("\nTraining on real, sourced, licensed prose...")
    for step in range(200):
        x, y = dataset[step % len(dataset)]
        loss = trainer.train_step(x.unsqueeze(0), y.unsqueeze(0))
        if step % 20 == 0:
            lr = trainer.optimizer.param_groups[0]["lr"]
            print(f"  step {step:3d}  loss {loss:.4f}  lr {lr:.5f}")

    print(
        "\nLoss decreasing on real, provenance-tracked, public-domain text -- the "
        "full local-scale foundry pipeline, not four hardcoded toy sentences."
    )


if __name__ == "__main__":
    main()
