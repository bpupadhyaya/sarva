"""sarva_foundry — from-scratch model training: tokenizer, transformer,
pretraining, post-training, distillation, evals.

Status: skeleton only. The core engine (this repo's `sarva` package) is the
T0/T1 focus; the foundry track begins once the provider registry and eval
harness exist to plug trained models into (see the roadmap's F0 milestone).
No HuggingFace `transformers`/`peft`/`trl` — everything above the PyTorch/CUDA
substrate is written from scratch here, by design.
"""

__version__ = "0.1.0.dev0"
