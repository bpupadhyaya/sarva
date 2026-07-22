"""sarva_foundry.tokenizer — a from-scratch byte-level BPE tokenizer.

No HuggingFace `tokenizers`/`tiktoken`. See `bpe.py` for the implementation
and `docs/` (Part VI) for the accompanying chapter.
"""

from sarva_foundry.tokenizer.bpe import ByteLevelBPETokenizer

__all__ = ["ByteLevelBPETokenizer"]
