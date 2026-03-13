"""KV Cache Compression Strategies.

This subpackage contains token selection strategies for KV cache compression.
These are NOT RL training methods — they are pluggable components that decide
WHICH tokens to keep when compressing the KV cache.

These strategies can be used by:
  - SMD (Shadow Mask Distillation) as the shadow mask generation strategy
  - Any RL training method that needs a token selection policy

Available strategies:
  - snapkv:  Attention-guided selection (keep high-attention tokens)
  - r_kv:    Redundancy-aware eviction (importance × redundancy joint scoring)
  - random:  Uniformly random selection (with protected sink/recent tokens)
  - recent:  Keep only the most recent tokens + sink tokens
"""

from .r_kv import RKVCacheCompressor
from .snapkv import SnapKVSelector
from .random_eviction import RandomSelector
from .recent_eviction import RecentSelector

STRATEGY_REGISTRY = {
    "snapkv": SnapKVSelector,
    "r_kv": RKVCacheCompressor,
    "random": RandomSelector,
    "recent": RecentSelector,
}
