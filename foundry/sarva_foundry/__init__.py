"""sarva_foundry — from-scratch model training: tokenizer, transformer,
pretraining, post-training, distillation, evals.

Status: F0 in progress. `sarva_foundry.tokenizer.ByteLevelBPETokenizer`
(a from-scratch byte-level BPE tokenizer) and `sarva_foundry.model`
(a dense decoder-only transformer: RoPE, RMSNorm, SwiGLU, GQA) are both
built and tested — see `examples/02_train_a_tokenizer.py` and
`examples/03_train_toy_transformer.py`. The pretraining data pipeline and
a real (non-toy, checkpointed) training loop are not built yet.
No HuggingFace `transformers`/`peft`/`trl` — everything above the PyTorch/CUDA
substrate is written from scratch here, by design.
"""

__version__ = "0.1.0.dev0"
