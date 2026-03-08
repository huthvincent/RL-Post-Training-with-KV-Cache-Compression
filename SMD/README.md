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

## Environment Setup

SMD is built on top of a distributed RL training stack. All experiments run inside a **Docker container** with the following pre-installed software.

### Software Stack (Tested Versions)

| Component | Version | Purpose |
|-----------|---------|---------|
| Ubuntu | 24.04 LTS | Base OS |
| Python | 3.12 | Runtime |
| PyTorch | 2.9.1+cu129 | Deep learning framework |
| CUDA | 12.9 | GPU acceleration |
| [Slime (veRL)](https://github.com/volcengine/verl) | 0.2.2 | RLHF training orchestration |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 0.16.0rc0 (core) | Distributed model parallelism |
| [SGLang](https://github.com/sgl-project/sglang) | 0.5.9 | High-performance rollout engine |
| [Ray](https://github.com/ray-project/ray) | 2.54.0 | Distributed task scheduling |
| HuggingFace Transformers | 4.57.1 | Model loading & tokenization |

### Step 1: Set Up the Docker Container

```bash
# Start from an NVIDIA PyTorch base image
docker run --gpus all --shm-size=64g --name smd_training \
  -v /path/to/your/workspace:/workspace \
  -it nvcr.io/nvidia/pytorch:25.02-py3

# Inside the container, install the required frameworks:

# 1. Install Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM
cd /root/Megatron-LM && pip install -e .
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

# 2. Install Slime (veRL fork with GRPO support)
git clone https://github.com/volcengine/verl.git /root/slime
cd /root/slime && pip install -e .

# 3. Install SGLang (for rollout inference)
pip install sglang[all]

# 4. Install Ray
pip install ray[default]

# 5. Install other dependencies
pip install rouge-score transformers accelerate
```

> **Note:** The exact Docker setup may require additional configuration depending on your GPU driver version and CUDA compatibility. We recommend starting from `nvcr.io/nvidia/pytorch:25.02-py3` which includes PyTorch + CUDA pre-configured.

### Step 2: Prepare Model Weights

```bash
# Download the base model (example: Qwen2.5-1.5B-Instruct)
mkdir -p shared_resources/models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir shared_resources/models/Qwen2.5-1.5B-Instruct

# Convert to Megatron torch_dist format (required for distributed training)
python /root/slime/tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint shared_resources/models/Qwen2.5-1.5B-Instruct \
  --save-path shared_resources/models/Qwen2.5-1.5B-Instruct_torch_dist
```

### Step 3: Prepare Datasets

```bash
mkdir -p shared_resources/datasets

# For TL;DR summarization (ROUGE reward)
# Format: JSONL with "prompt" and "label" fields
# Source: https://huggingface.co/datasets/CarperAI/openai_summarize_tldr

# For GSM8K math reasoning (Math accuracy reward)
# Format: JSONL with "prompt" and "label" fields
# Source: https://huggingface.co/datasets/openai/gsm8k
```

### Hardware Requirements

| Model Size | Min GPU VRAM | Recommended GPU |
|-----------|-------------|-----------------|
| 1.5B | 40 GB | NVIDIA A100 / H100 |
| 7B | 80 GB | NVIDIA H100 / H200 |
| 14B | 2 × 80 GB | 2 × H100 (TP=2) |

---

## Quick Start

```bash
# Run all 7 experiments (see experiments/README_EXPERIMENTS.md for details)
bash experiments/exp_02_sota_showdown/run_exp.sh

# Or run the micro sanity check first (10 rollouts, ~2 min per test)
bash experiments/run_micro_sanity_check.sh
```

Each experiment script has a **Configuration** section at the top — just change `MODEL_SIZE` and `MODEL_PATH` to scale up:

```bash
# ── Configuration (Modify here for scale-up) ───────────────
MODEL_SIZE="1.5B"        # ← Change to "7B" or "14B"
MODEL_PATH="..."         # ← Point to your model weights
CONTAINER="smd_training" # ← Your Docker container name
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
