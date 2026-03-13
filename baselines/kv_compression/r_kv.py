"""R-KV Baseline: Redundancy-Aware KV Cache Compression.

Core Idea:
    Dynamically evict KV cache entries during decode by jointly scoring
    tokens on Importance (attention-based) and Redundancy (key similarity).
    Particularly suited for long chain-of-thought generation.

Algorithm:
    1. Trigger: Compress when buffer (128 tokens) fills up.
       Always retain the last α=8 newly generated tokens.

    2. Importance Score I:
       - Extract attention weights for recent α query tokens
       - Apply max-pooling with sliding window (2W) to smooth outliers
       - Average across recent queries → per-key importance

    3. Redundancy Score R:
       - Compute cosine similarity between normalized key vectors
       - Zero diagonal and connections to nearest β similar tokens
       - Softmax over average similarities → per-token redundancy

    4. Joint Score: Z = λ·I - (1-λ)·R  (λ=0.1, importance-downweighted)
       Keep top-Budget tokens; evict the rest.

Note:
    R-KV operates during decode (modifying KV cache), not in the loss.
    The loss function is standard GRPO. The KVCacheCompressor class below
    implements the eviction logic that wraps the decode step.
"""

import logging
import math
from argparse import Namespace
from collections.abc import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Hyperparameters ─────────────────────────────────────────────────────
BUFFER_SIZE = 128       # Compress every 128 new tokens
RECENT_PROTECT = 8      # α: always keep the last 8 tokens
IMPORTANCE_WINDOW = 16  # W: half-width for max-pooling smoothing
REDUNDANCY_BETA = 4     # β: ignore top-β most similar neighbors
LAMBDA = 0.1            # Joint score weight (low = favor redundancy removal)


def r_kv_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Standard GRPO loss — R-KV optimizes KV eviction during decode, not the loss.

    The actual R-KV innovation is in the RKVCacheCompressor class below.
    This loss tracks additional KV-compression metrics when available.
    """
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths, response_lengths=response_lengths,
        with_entropy=True, max_seq_lens=max_seq_lens,
    )
    log_probs = log_probs_and_entropy["log_probs"]
    entropy_list = log_probs_and_entropy["entropy"]

    old_log_probs_list = (
        batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    )
    advantages_list = batch["advantages"]
    loss_masks_list = batch["loss_masks"]
    num_samples = len(log_probs)
    eps = getattr(args, "eps_clip", 0.2)

    pg_losses, pg_clipfracs, all_entropies = [], [], []

    for i in range(num_samples):
        lp = log_probs[i]
        old_lp = old_log_probs_list[i].to(lp.device)
        adv = advantages_list[i].to(lp.device)
        lm = loss_masks_list[i].float().to(lp.device)

        ratio = torch.exp(lp - old_lp)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        pg_loss = torch.max(pg_loss1, pg_loss2) * lm

        pg_losses.append(pg_loss)
        pg_clipfracs.append(((ratio - 1.0).abs() > eps).float() * lm)
        if i < len(entropy_list):
            all_entropies.append(entropy_list[i] * lm)

    pg_loss = sum_of_sample_mean(torch.cat(pg_losses, dim=0))
    pg_clipfrac = sum_of_sample_mean(torch.cat(pg_clipfracs, dim=0))
    entropy_loss = sum_of_sample_mean(torch.cat(all_entropies, dim=0)) if all_entropies else torch.tensor(0.0, device=logits.device)

    total_loss = pg_loss - args.entropy_coef * entropy_loss
    if torch.cat(log_probs, dim=0).numel() == 0:
        total_loss += 0 * logits.sum()

    metrics = {
        "loss": total_loss.clone().detach(),
        "r_kv/pg_loss": pg_loss.clone().detach(),
        "r_kv/entropy": entropy_loss.clone().detach(),
        "r_kv/clipfrac": pg_clipfrac.clone().detach(),
    }
    return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════
# KV Cache Compression Engine
# ═══════════════════════════════════════════════════════════════════════

class RKVCacheCompressor:
    """Redundancy-aware KV cache compressor for decode-time eviction.

    Usage:
        compressor = RKVCacheCompressor(budget=512, buffer=128)
        # During decode loop:
        for token in generate():
            compressor.add_token(key_states, attn_weights)
            if compressor.should_compress():
                keep_indices = compressor.compute_eviction(key_states, attn_weights)
                kv_cache = kv_cache[:, :, keep_indices, :]

    Args:
        budget: Maximum number of KV entries to retain
        buffer_size: Number of new tokens before triggering compression
        alpha: Number of recent tokens to always protect
        window: Half-width for importance score max-pooling
        beta: Number of top-similar neighbors to ignore in redundancy
        lam: Joint score weight λ (low = favor redundancy removal)
    """

    def __init__(
        self,
        budget: int = 512,
        buffer_size: int = BUFFER_SIZE,
        alpha: int = RECENT_PROTECT,
        window: int = IMPORTANCE_WINDOW,
        beta: int = REDUNDANCY_BETA,
        lam: float = LAMBDA,
    ):
        self.budget = budget
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.window = window
        self.beta = beta
        self.lam = lam
        self.tokens_since_compress = 0
        self.total_evictions = 0

    def should_compress(self, current_cache_len: int) -> bool:
        """Check if compression should be triggered."""
        self.tokens_since_compress += 1
        if self.tokens_since_compress >= self.buffer_size and current_cache_len > self.budget:
            return True
        return False

    def compute_importance_scores(
        self,
        attn_weights: torch.Tensor,
        num_keys: int,
    ) -> torch.Tensor:
        """Compute importance score I for each key position.

        Args:
            attn_weights: Attention weights (num_heads, num_queries, num_keys)
                          Use only the last α query positions.

        Returns:
            Importance scores shape (num_keys,)
        """
        # Use last α queries only
        recent_attn = attn_weights[:, -self.alpha:, :num_keys]  # (H, α, K)

        # Average across heads and recent queries
        avg_attn = recent_attn.mean(dim=(0, 1))  # (K,)

        # Max-pooling with sliding window to smooth outliers
        if self.window > 0 and avg_attn.numel() > 2 * self.window:
            padded = F.pad(avg_attn.unsqueeze(0).unsqueeze(0),
                          (self.window, self.window), mode="replicate")
            smoothed = F.max_pool1d(padded, kernel_size=2 * self.window + 1, stride=1)
            avg_attn = smoothed.squeeze()[:num_keys]

        return avg_attn

    def compute_redundancy_scores(
        self,
        key_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute redundancy score R for each key position.

        Args:
            key_states: Key vectors (num_keys, head_dim) or averaged across heads

        Returns:
            Redundancy scores shape (num_keys,)
        """
        # Normalize keys
        key_norm = F.normalize(key_states.float(), dim=-1)  # (K, D)

        # Cosine similarity matrix
        sim_matrix = torch.mm(key_norm, key_norm.t())  # (K, K)

        # Zero diagonal (self-similarity)
        sim_matrix.fill_diagonal_(0.0)

        # Zero top-β most similar neighbors per token
        if self.beta > 0 and sim_matrix.shape[0] > self.beta:
            topk_vals, topk_idx = sim_matrix.topk(self.beta, dim=-1)
            sim_matrix.scatter_(1, topk_idx, 0.0)

        # Average remaining similarities → redundancy score
        avg_sim = sim_matrix.mean(dim=-1)  # (K,)

        # Softmax to normalize to probability distribution
        redundancy = F.softmax(avg_sim, dim=0)

        return redundancy

    def compute_eviction(
        self,
        key_states: torch.Tensor,
        attn_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute which KV entries to keep after eviction.

        Args:
            key_states: Key vectors (num_keys, head_dim)
            attn_weights: Attention matrix (num_heads, num_queries, num_keys)

        Returns:
            Indices of KV entries to keep, shape (budget,)
        """
        num_keys = key_states.shape[0]
        device = key_states.device
        attn_weights = attn_weights.to(device)

        if num_keys <= self.budget:
            return torch.arange(num_keys, device=device)

        # Always protect the last α tokens
        candidate_end = num_keys - self.alpha
        candidates = torch.arange(candidate_end, device=device)

        # Compute scores for candidates only
        importance = self.compute_importance_scores(attn_weights, candidate_end)
        redundancy = self.compute_redundancy_scores(key_states[:candidate_end])

        # Ensure shapes match
        min_len = min(importance.shape[0], redundancy.shape[0], candidate_end)
        importance = importance[:min_len]
        redundancy = redundancy[:min_len]

        # Normalize importance to [0, 1]
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())

        # Joint score: Z = λ·I - (1-λ)·R
        # Higher Z = more important + less redundant → keep
        z_scores = self.lam * importance - (1 - self.lam) * redundancy

        # Select top candidates to fill budget (minus protected slots)
        keep_count = self.budget - self.alpha
        _, top_indices = z_scores.topk(min(keep_count, min_len))
        keep_candidates = candidates[top_indices]

        # Combine: kept candidates + protected recent tokens
        protected = torch.arange(candidate_end, num_keys, device=device)
        keep_indices = torch.cat([keep_candidates, protected])
        keep_indices = keep_indices.sort().values

        self.tokens_since_compress = 0
        self.total_evictions += (num_keys - keep_indices.shape[0])

        return keep_indices
