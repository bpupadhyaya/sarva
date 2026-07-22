"""sarva_foundry — from-scratch model training: tokenizer, transformer,
pretraining, post-training, distillation, evals.

Status: F0 started. `sarva_foundry.tokenizer.ByteLevelBPETokenizer` is the
first real component — a from-scratch, trainable byte-level BPE tokenizer
(see `examples/02_train_a_tokenizer.py`). The model architecture,
pretraining loop, and everything after it are not built yet.
No HuggingFace `transformers`/`peft`/`trl` — everything above the PyTorch/CUDA
substrate is written from scratch here, by design.
"""

__version__ = "0.1.0.dev0"
