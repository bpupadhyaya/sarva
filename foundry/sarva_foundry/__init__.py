"""sarva_foundry — from-scratch model training: tokenizer, transformer,
pretraining, post-training, distillation, evals.

Status: F0 has a runnable (toy-scale) pretraining pipeline end to end —
`sarva_foundry.tokenizer.ByteLevelBPETokenizer`, `sarva_foundry.model`
(dense decoder-only transformer: RoPE, RMSNorm, SwiGLU, GQA),
`sarva_foundry.data` (local-file corpus sourcing, exact + MinHash
near-duplicate dedup, length-filtering, source/license provenance
tracking via `SourcedDocument` (including a per-file license manifest),
and corpus-to-batches chunking), and
`sarva_foundry.train.Trainer` (training loop with bit-identical
checkpoint/resume, plus `WarmupCosineSchedule` for a real LR curve
instead of a flat rate) all built and tested — see
`examples/04_pretrain_and_resume.py`. Web-scale corpus sourcing and
distributed training are not built yet.
No HuggingFace `transformers`/`peft`/`trl` — everything above the PyTorch/CUDA
substrate is written from scratch here, by design.
"""

__version__ = "0.1.0.dev0"
