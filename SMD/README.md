# Shadow Mask Distillation (SMD)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**Shadow Mask Distillation** enables reliable Reinforcement Learning post-training (GRPO/PPO) under aggressive KV cache compression. Without correction, naive KV compression during rollout causes **complete learning collapse** — the model achieves zero reward. SMD fixes this via a dual-track loss that aligns the training signal with the compressed policy.

---

## Key Results

| Metric | Dense Baseline | SMD (SnapKV 50%) | Sparse-RL (SOTA) |
|--------|---------------|------------------|------------------|
| ROUGE-L (avg) | -0.0517 | **-0.0189** (+63%) | -0.0511 |
| ROUGE-L (tail-50) | +0.0348 | **+0.0869** | +0.0870 |
| Reward Variance | 0.0393 | **0.0351** (-10.7%) | 0.0393 |
| GSM8K Accuracy | 68.0% | **69.5%** (+1.5%) | — |

> **Naive KV compression = Total Failure.** Without SMD, both SnapKV 50% and Random 50% achieve **exactly zero reward** across 500 training steps (see [Experiment Results](./EXPERIMENT_RESULTS.md#exp_01)).

---

## Method Overview

```
┌─────────────────────────────────────────────────────────┐
│                   SMD Training Loop                     │
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ Rollout  │───▶│ Shadow Mask  │───▶│   Learner    │   │
│  │ (SGLang) │    │ Interceptor  │    │  (Megatron)  │   │
│  └──────────┘    └──────────────┘    └──────┬───────┘   │
│       │              Generates                │         │
│       │           shadow_masks          ┌─────┴─────┐   │
│       │              (T×T)              │ Dual-Track│   │
│       │                                 │    Loss   │   │
│       │                                 ├───────────┤   │
│       │                                 │ Track 1:  │   │
│       │                                 │ Shadow PG │   │
│       │                                 │ (on-policy│   │
│       │                                 │ faithful) │   │
│       │                                 ├───────────┤   │
│       │                                 │ Track 2:  │   │
│       │                                 │ KL Distill│   │
│       │                                 │ (dense →  │   │
│       │                                 │  sparse)  │   │
│       │                                 └───────────┘   │
└─────────────────────────────────────────────────────────┘

Total Loss = GRPO(π_shadow) + λ · D_KL(π_dense ‖ sg(π_shadow))
```

**Track 1 — Shadow Policy Gradient:** Restricts the GRPO gradient to tokens whose KV context was preserved by compression "on-policy faithful" tokens. This ensures the gradient signal is mathematically consistent with the compressed policy.

**Track 2 — KL Distillation:** Aligns the full dense model's predictions with the compressed policy via KL divergence. This acts as a powerful regularizer that transfers knowledge from the sparse reasoning pathway to the full model.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/RLKV.git
cd RLKV/SMD
pip install -r requirements.txt
```

### Prerequisites
- Python ≥ 3.10
- PyTorch ≥ 2.1 with CUDA support
- HuggingFace `transformers` ≥ 4.36
- [Slime](https://github.com/volcengine/verl) (for full training pipeline)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (for distributed training)

---

## Quick Start

### 1. Run SMD Training (Reddit TL;DR)

```bash
bash scripts/run_smd_tldr.sh
```

### 2. Run Dense Baseline

```bash
bash scripts/run_baseline.sh
```

### 3. Compare with Sparse-RL

```bash
bash scripts/run_sparse_rl.sh
```

---

## Core Components

| Module | Description |
|--------|-------------|
| [`shadow_distillation_loss.py`](src/shadow_distillation_loss.py) | Dual-track GRPO loss (Track 1: Shadow PG + Track 2: KL Distill) |
| [`shadow_mask_interceptor.py`](src/shadow_mask_interceptor.py) | Generates compression masks (SnapKV/Random/Recent strategies) |
| [`shadow_attention.py`](src/shadow_attention.py) | Injects shadow masks into attention computation |
| [`sparse_rl_loss.py`](src/sparse_rl_loss.py) | Sparse-RL baseline (rejection sampling + importance reweighting) |
| [`native_hf_rollout.py`](src/native_hf_rollout.py) | Native HF rollout with true physical KV cache eviction |

---

## Experiment Results

See **[EXPERIMENT_RESULTS.md](./EXPERIMENT_RESULTS.md)** for detailed analysis of all 7 experiments with publication-ready figures.

| Figure | Finding |
|--------|---------|
| Exp 01 | Naive KV compression → **total collapse** (zero reward) |
| Exp 02 | SMD outperforms Sparse-RL by **63%** in average ROUGE-L |
| Exp 03 | SMD improves downstream GSM8K accuracy by **+1.5%** |
| Exp 04 | True KV eviction causes **VRAM spike** (not-in-place allocation) |
| Exp 05 | 50% retention is the **optimal sweet spot** (U-shaped curve) |
| Exp 06 | Distillation λ=0.1 achieves **best convergence** |
| Exp 07 | SnapKV attention-guided selection >> Random ≈ Recent |

---

## Citation

```bibtex
@article{smd2026,
  title={Shadow Mask Distillation: Breaking the Memory Wall for
         RL Post-Training with KV Cache Compression},
  year={2026}
}
```

## License

Apache 2.0
