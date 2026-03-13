# Baselines

This directory contains two categories of baselines, organized by their role in the system.

## RL Training Methods (`baselines/`)

These are **RL training algorithms** that compete with SMD. Each addresses the off-policy mismatch problem (KV compression during rollout breaks standard GRPO) with a different correction strategy.

| Method | File | Core Mechanism |
|--------|------|----------------|
| **Sparse-RL** | `sparse_rl.py` | Reject 20% most-divergent rollouts + importance reweighting (ρ clipped to [0.8, 1.2]) |
| **QuRL** | `qurl.py` | Update-Aware Quantization scaling + adaptive PPO clipping |
| **RLHFless** | `rlhfless.py` | Deduplicated prefill + length-aware elastic scheduling |

```python
# Usage: plug into Slime's custom loss
python train.py \
    --loss-type custom_loss \
    --custom-loss-function-path baselines.sparse_rl.sparse_rl_loss_function
```

## KV Compression Strategies (`baselines/kv_compression/`)

These are **token selection policies** that decide WHICH KV cache entries to keep during compression. They are **not** RL training methods — they are pluggable components used **by** SMD (or any other training method) as the shadow mask generation strategy.

| Strategy | File | How It Selects Tokens |
|----------|------|-----------------------|
| **SnapKV** | `snapkv.py` | Attention-guided: keep positions with highest cumulative attention mass |
| **R-KV** | `r_kv.py` | Joint scoring: importance (attention) × redundancy (key similarity) |
| **Random** | `random_eviction.py` | Uniform random selection (with protected sink/recent tokens) |
| **Recent** | `recent_eviction.py` | Keep only the most recent tokens + sink tokens |

```python
# Usage: choose a strategy when configuring SMD
from baselines.kv_compression import STRATEGY_REGISTRY

selector = STRATEGY_REGISTRY["snapkv"](retention_ratio=0.5)
keep_indices = selector.select(prompt_length=200)
```

> **Key distinction:** RL training methods (Sparse-RL, QuRL, etc.) answer *"how to train with compressed KV"*. Compression strategies (SnapKV, R-KV, etc.) answer *"which tokens to keep"*. SMD can use **any** compression strategy — the choice is orthogonal to the training algorithm.

## Testing

```bash
python baselines/test_baselines.py
```
