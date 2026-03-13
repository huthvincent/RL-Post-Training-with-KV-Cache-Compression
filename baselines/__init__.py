"""Baseline RL training methods for KV-compressed post-training.

These are RL training algorithms that compete with SMD. They address the
off-policy mismatch when RL rollouts use compressed KV caches, each with
a different correction strategy.

All methods follow the Slime custom loss interface:
    loss_fn(args, batch, logits, sum_of_sample_mean) -> (loss, metrics)

Registry:
    - sparse_rl: Rejection sampling + importance reweighting (arXiv:2601.10079)
    - qurl:      Update-Aware Quantization + adaptive PPO clipping
    - rlhfless:  Serverless elastic scheduling with deduplicated prefill

Note:
    KV cache compression strategies (SnapKV, R-KV, Random, Recent) are in
    baselines/kv_compression/. Those are token selection policies — pluggable
    components that decide WHICH tokens to keep. They are NOT RL training
    methods and do NOT belong here.
"""

from .sparse_rl import sparse_rl_loss_function
from .qurl import qurl_loss_function
from .rlhfless import rlhfless_loss_function

BASELINE_REGISTRY = {
    "sparse_rl": sparse_rl_loss_function,
    "qurl": qurl_loss_function,
    "rlhfless": rlhfless_loss_function,
}
