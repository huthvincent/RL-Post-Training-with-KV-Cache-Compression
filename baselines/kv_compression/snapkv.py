"""SnapKV: Attention-Guided KV Cache Compression.

Reference: Li et al., "SnapKV: LLM Knows What You are Looking for Before Generation"
           (arXiv:2404.14469)

Core Idea:
    During prefill, compute cumulative attention scores across all heads.
    Keep the top-k positions with the highest attention mass. Always protect
    "sink tokens" (initial positions) and recent observation window tokens.

This implementation provides a standalone selector class that can be used
independently of SMD's shadow mask mechanism.
"""

import torch


class SnapKVSelector:
    """Attention-guided token selector based on SnapKV.

    Args:
        retention_ratio: Fraction of prompt KV positions to retain (e.g., 0.5).
        observation_window: Number of most-recent tokens always retained.
        sink_tokens: Number of initial tokens always retained.
    """

    def __init__(
        self,
        retention_ratio: float = 0.5,
        observation_window: int = 64,
        sink_tokens: int = 4,
    ):
        self.retention_ratio = retention_ratio
        self.observation_window = observation_window
        self.sink_tokens = sink_tokens

    def select(
        self,
        prompt_length: int,
        attention_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select which prompt positions to retain.

        Args:
            prompt_length: Total number of prompt tokens.
            attention_scores: Optional (num_heads, seq_len, seq_len) tensor.
                If provided, uses actual attention weights for scoring.
                If None, uses a position-based importance heuristic.

        Returns:
            Sorted tensor of retained position indices.
        """
        num_keep = max(
            self.sink_tokens + self.observation_window,
            int(prompt_length * self.retention_ratio),
        )
        num_keep = min(num_keep, prompt_length)

        if num_keep >= prompt_length:
            return torch.arange(prompt_length)

        if attention_scores is not None:
            return self._select_by_attention(prompt_length, num_keep, attention_scores)
        else:
            return self._select_by_position(prompt_length, num_keep)

    def _select_by_attention(
        self,
        prompt_length: int,
        num_keep: int,
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Select using actual attention weights (true SnapKV)."""
        # Sum attention received by each key position across all heads and queries
        importance = attention_scores[:, :, :prompt_length].sum(dim=(0, 1))  # (prompt_length,)

        # Boost protected positions (save max before any boost to avoid pollution)
        max_imp = importance.max().item()
        importance[:self.sink_tokens] += max_imp * 10
        importance[-self.observation_window:] += max_imp * 5

        _, top_indices = importance.topk(num_keep)
        return top_indices.sort().values

    def _select_by_position(
        self,
        prompt_length: int,
        num_keep: int,
    ) -> torch.Tensor:
        """Approximate SnapKV using position-based heuristic.

        Uses harmonic-mean distance: tokens near either end score higher.
        This approximation is used when actual attention scores are unavailable
        (e.g., when rollout is done by a black-box engine like SGLang).
        """
        importance = torch.zeros(prompt_length, dtype=torch.float32)

        for i in range(prompt_length):
            dist_start = i + 1
            dist_end = prompt_length - i
            importance[i] = 2.0 / (1.0 / dist_start + 1.0 / dist_end)

            if i < self.sink_tokens:
                importance[i] += prompt_length
            if i >= prompt_length - self.observation_window:
                importance[i] += prompt_length * 0.5

        _, top_indices = importance.topk(num_keep)
        return top_indices.sort().values
