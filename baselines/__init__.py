"""Baseline methods for KV cache compression in RL post-training.

All methods follow the Slime custom loss interface:
    loss_fn(args, batch, logits, sum_of_sample_mean) -> (loss, metrics)

Registry:
    - sparse_rl: Statistical correction via rejection sampling + importance reweighting
    - qurl:      Quantized rollout with UAQ scaling + adaptive clipping
    - rlhfless:  Serverless elastic scheduling with deduplicated prefill
    - r_kv:      Redundancy-aware KV cache eviction during decode
"""

from .sparse_rl import sparse_rl_loss_function
from .qurl import qurl_loss_function
from .rlhfless import rlhfless_loss_function
from .r_kv import r_kv_loss_function

BASELINE_REGISTRY = {
    "sparse_rl": sparse_rl_loss_function,
    "qurl": qurl_loss_function,
    "rlhfless": rlhfless_loss_function,
    "r_kv": r_kv_loss_function,
}
