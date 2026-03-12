# Baselines: SOTA KV Cache Compression for RL Post-Training

This directory contains implementations of 4 baseline methods for comparison with Shadow Mask Distillation (SMD).

## Methods

| Method | File | Type | Key Mechanism |
|--------|------|------|---------------|
| **Sparse-RL** | `sparse_rl.py` | Loss | Rejection sampling + importance reweighting |
| **QuRL** | `qurl.py` | Loss + Weights | Update-Aware Quantization + adaptive PPO clipping |
| **RLHFless** | `rlhfless.py` | Loss + Scheduler | Deduplicated prefill + length-aware scheduling |
| **R-KV** | `r_kv.py` | Loss + KV Eviction | Importance × Redundancy joint scoring |

## Interface

All loss functions share the same Slime custom loss interface:

```python
def method_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
```

## Usage

```bash
# Use via Slime's --custom-loss-function-path:
python train.py \
    --loss-type custom_loss \
    --custom-loss-function-path baselines.sparse_rl.sparse_rl_loss_function

# Or import directly:
from baselines import BASELINE_REGISTRY
loss_fn = BASELINE_REGISTRY["qurl"]
```

## Testing

```bash
python baselines/test_baselines.py
```
