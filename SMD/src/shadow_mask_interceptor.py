"""Shadow Mask Interceptor: Simulates KV cache compression decisions.

This module generates shadow masks — boolean attention masks that encode which
KV cache positions would be retained by a given compression algorithm (SnapKV,
Random, or Recent). The masks are generated AFTER standard rollout generation,
without modifying the generation process itself.

Architecture Decision: Why simulate instead of actually compressing?
    Production inference engines like SGLang/vLLM manage KV cache at the CUDA
    kernel level with paged memory, making it impossible to inject custom eviction
    logic from Python. Instead, we simulate the compression decisions externally
    and inject the resulting masks into the learner's attention during training.
    This "shadow" approach achieves the same mathematical effect (on-policy
    alignment) without requiring access to inference engine internals.

The generated shadow_mask is a (total_len, total_len) boolean tensor:
    - Row i, Column j = True means "token i can attend to token j"
    - This is a SUBSET of the standard causal mask
    - Prompt tokens attend to each other normally (no compression within prompt)
    - Response tokens can only attend to the retained prompt KV positions + all
      previous response tokens (response-internal KV is never compressed)
"""

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class ShadowMaskConfig:
    """Configuration for the shadow mask generation strategy.

    Attributes:
        enabled: Whether shadow mask generation is active.
        retention_ratio: Fraction of prompt KV positions to retain (e.g., 0.5 = 50%).
        strategy: Token selection algorithm. One of:
            - "snapkv": Attention-guided selection (uses real attention when available,
              falls back to position heuristic).
            - "r_kv": Redundancy-aware eviction. Joint importance (attention) ×
              redundancy (key cosine similarity) scoring. Requires key_states.
            - "random": Uniformly random selection (with protected sink/recent tokens).
            - "recent": Keep only the most recent tokens + sink tokens.
        observation_window: Number of most-recent prompt tokens always retained.
        sink_tokens: Number of initial prompt tokens always retained (attention sinks).
        r_kv_lambda: R-KV joint score weight λ. Low = favor removing redundant tokens.
        r_kv_beta: R-KV: ignore top-β most similar neighbors in redundancy calc.
    """
    enabled: bool = False
    retention_ratio: float = 0.5
    strategy: str = "snapkv"
    observation_window: int = 64
    sink_tokens: int = 4
    r_kv_lambda: float = 0.1
    r_kv_beta: int = 4


class ShadowMaskInterceptor:
    """Generates shadow masks that simulate KV cache compression decisions.

    This class operates as a post-processing step after rollout generation.
    It doesn't modify the actual generation — instead, it computes what a
    compression algorithm WOULD have done, and encodes those decisions as
    boolean attention masks for use during training.

    The mask structure ensures:
    1. Prompt-to-prompt attention is always full causal (no compression).
    2. Response tokens attend to a SUBSET of prompt KV (the retained positions).
    3. Response tokens always attend to ALL previous response tokens.

    Example:
        >>> config = ShadowMaskConfig(enabled=True, retention_ratio=0.5, strategy="snapkv")
        >>> interceptor = ShadowMaskInterceptor(config)
        >>> mask = interceptor.generate_shadow_mask(prompt_length=100, response_length=50)
        >>> mask.shape
        torch.Size([150, 150])
    """

    def __init__(self, config: ShadowMaskConfig):
        self.config = config

    def generate_shadow_mask(
        self,
        prompt_length: int,
        response_length: int,
        seed: int | None = None,
        attention_scores: torch.Tensor | None = None,
        key_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate a shadow mask for a single sample.

        The mask simulates KV cache compression applied at the prefill stage:
        a subset of prompt KV positions is selected for retention, while the
        rest are "evicted." During autoregressive generation, each new response
        token can only attend to the retained prompt positions.

        Args:
            prompt_length: Number of tokens in the prompt.
            response_length: Number of tokens in the generated response.
            seed: Optional random seed for reproducibility (used by "random" strategy).
            attention_scores: Optional per-key importance scores of shape (seq_len,)
                or attention weights of shape (num_heads, seq_len, seq_len).
                When provided with strategy="snapkv" or "r_kv", uses these real
                attention scores instead of the position-based heuristic.
            key_states: Optional key vectors of shape (seq_len, head_dim).
                Required by strategy="r_kv" for redundancy scoring.

        Returns:
            Boolean tensor of shape (total_len, total_len) where True indicates
            that attention is allowed at that (query, key) position.
        """
        total_len = prompt_length + response_length
        config = self.config

        if not config.enabled or config.retention_ratio >= 1.0:
            return torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))

        # Determine how many prompt KV positions to retain
        num_prompt_keep = max(
            config.sink_tokens + config.observation_window,
            int(prompt_length * config.retention_ratio),
        )
        num_prompt_keep = min(num_prompt_keep, prompt_length)

        # Select which prompt positions to retain
        retained_prompt_indices = self._select_prompt_positions(
            prompt_length, num_prompt_keep, seed, attention_scores, key_states
        )

        # Build the shadow mask
        shadow_mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # Prompt tokens: standard causal attention (no compression within prompt)
        for i in range(prompt_length):
            shadow_mask[i, : i + 1] = True

        # Response tokens: attend to retained prompt + all previous response
        for i in range(response_length):
            gen_pos = prompt_length + i
            shadow_mask[gen_pos, retained_prompt_indices] = True  # Retained prompt KV
            shadow_mask[gen_pos, prompt_length : gen_pos + 1] = True  # All prior response

        return shadow_mask

    def _select_prompt_positions(
        self,
        prompt_length: int,
        num_keep: int,
        seed: int | None = None,
        attention_scores: torch.Tensor | None = None,
        key_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select which prompt KV positions to retain based on the configured strategy.

        All strategies protect "sink tokens" (initial positions that accumulate
        disproportionate attention mass) and recent tokens within the observation window.

        Args:
            prompt_length: Total number of prompt tokens.
            num_keep: Number of positions to retain.
            seed: Random seed for the "random" strategy.
            attention_scores: Optional real attention scores for SnapKV.
                Shape: (seq_len,) for per-key importance, or
                       (num_heads, seq_len, seq_len) for raw attention weights.

        Returns:
            Sorted tensor of retained position indices.
        """
        config = self.config

        if num_keep >= prompt_length:
            return torch.arange(prompt_length)

        # Guard against empty prompt
        if prompt_length == 0:
            return torch.tensor([], dtype=torch.long)

        if config.strategy == "recent":
            # Keep sink tokens + most recent tokens
            sink = torch.arange(min(config.sink_tokens, prompt_length))
            recent_start = max(config.sink_tokens, prompt_length - (num_keep - config.sink_tokens))
            recent = torch.arange(recent_start, prompt_length)
            indices = torch.cat([sink, recent]).unique()

        elif config.strategy == "random":
            # Random selection with protected sink + recent tokens
            if seed is not None:
                gen = torch.Generator().manual_seed(seed)
            else:
                gen = None

            always_keep = set(range(min(config.sink_tokens, prompt_length)))
            always_keep.update(range(max(0, prompt_length - config.observation_window), prompt_length))

            middle_positions = [i for i in range(prompt_length) if i not in always_keep]
            num_random = max(0, num_keep - len(always_keep))
            if num_random > 0 and middle_positions:
                perm = torch.randperm(len(middle_positions), generator=gen)
                selected = [middle_positions[p] for p in perm[:num_random]]
                always_keep.update(selected)

            indices = torch.tensor(sorted(always_keep), dtype=torch.long)

        elif config.strategy == "snapkv":
            if attention_scores is not None:
                # ── Real SnapKV: use actual attention weights ────────────
                indices = self._select_by_real_attention(
                    prompt_length, num_keep, attention_scores
                )
            else:
                # ── Fallback: position-based heuristic ───────────────────
                logger.debug(
                    "No attention scores provided for SnapKV, using position heuristic"
                )
                indices = self._select_by_position_heuristic(
                    prompt_length, num_keep
                )

        elif config.strategy == "r_kv":
            if attention_scores is not None:
                indices = self._select_by_r_kv(
                    prompt_length, num_keep, attention_scores, key_states
                )
            else:
                logger.debug(
                    "No attention scores for R-KV, falling back to position heuristic"
                )
                indices = self._select_by_position_heuristic(
                    prompt_length, num_keep
                )

        else:
            raise ValueError(f"Unknown shadow mask strategy: {config.strategy}")

        return indices

    def _select_by_real_attention(
        self,
        prompt_length: int,
        num_keep: int,
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        """True SnapKV: select prompt positions using real attention weights.

        This is the proper SnapKV algorithm as described in the paper:
        1. Take attention from the observation window (last N query tokens)
        2. Sum attention received by each key position across heads and queries
        3. Always protect sink tokens and recent tokens
        4. Select top-k positions by cumulative attention

        Args:
            prompt_length: Number of prompt tokens.
            num_keep: Number of positions to retain.
            attention_scores: Either:
                - (seq_len,) per-key importance scores (pre-aggregated)
                - (num_heads, seq_len, seq_len) raw attention weights

        Returns:
            Sorted tensor of retained position indices.
        """
        config = self.config

        if attention_scores.dim() == 1:
            # Pre-aggregated importance scores (with padding if needed)
            if attention_scores.shape[0] < prompt_length:
                logger.warning(
                    f"Attention scores length ({attention_scores.shape[0]}) < "
                    f"prompt_length ({prompt_length}), padding with zeros"
                )
                padded = torch.zeros(prompt_length, dtype=attention_scores.dtype,
                                     device=attention_scores.device)
                padded[:attention_scores.shape[0]] = attention_scores
                importance = padded.float()
            else:
                importance = attention_scores[:prompt_length].float().clone()
        elif attention_scores.dim() == 3:
            # Raw attention weights: (H, S, S)
            # Use last observation_window queries to score keys (SnapKV paper)
            obs_start = max(0, prompt_length - config.observation_window)
            recent_attn = attention_scores[:, obs_start:prompt_length, :prompt_length]
            # Sum across heads and recent queries → per-key importance
            importance = recent_attn.sum(dim=(0, 1))[:prompt_length].float()
        elif attention_scores.dim() == 4:
            # (B, H, S, S) — take first batch element
            obs_start = max(0, prompt_length - config.observation_window)
            recent_attn = attention_scores[0, :, obs_start:prompt_length, :prompt_length]
            importance = recent_attn.sum(dim=(0, 1))[:prompt_length].float()
        else:
            logger.warning(
                f"Unexpected attention shape {attention_scores.shape}, "
                "falling back to position heuristic"
            )
            return self._select_by_position_heuristic(prompt_length, num_keep)

        # Boost protected positions to guarantee retention
        max_imp = importance.max().item() + 1.0
        importance[:config.sink_tokens] += max_imp * 10  # Sink tokens
        importance[-config.observation_window:] += max_imp * 5  # Recent tokens

        # Top-k selection
        _, top_indices = importance.topk(min(num_keep, prompt_length))
        return top_indices.sort().values

    def _select_by_r_kv(
        self,
        prompt_length: int,
        num_keep: int,
        attention_scores: torch.Tensor,
        key_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """R-KV: select prompt positions using joint importance + redundancy scoring.

        Implements the R-KV algorithm:
        1. Importance: attention-based scoring (same as SnapKV)
        2. Redundancy: cosine similarity between key vectors
        3. Joint score: Z = λ·I - (1-λ)·R — keep high-importance, low-redundancy

        Args:
            prompt_length: Number of prompt tokens.
            num_keep: Number of positions to retain.
            attention_scores: Per-key importance (seq_len,) or raw weights (H,S,S).
            key_states: Key vectors (seq_len, head_dim). If None, falls back to
                pure attention-based selection (no redundancy scoring).

        Returns:
            Sorted tensor of retained position indices.
        """
        config = self.config
        import torch.nn.functional as F

        # ── Step 1: Compute importance (same as SnapKV) ──────────────
        if attention_scores.dim() == 1:
            importance = attention_scores[:prompt_length].float().clone()
        elif attention_scores.dim() == 3:
            obs_start = max(0, prompt_length - config.observation_window)
            recent_attn = attention_scores[:, obs_start:prompt_length, :prompt_length]
            importance = recent_attn.sum(dim=(0, 1)).float()
        elif attention_scores.dim() == 4:
            obs_start = max(0, prompt_length - config.observation_window)
            recent_attn = attention_scores[0, :, obs_start:prompt_length, :prompt_length]
            importance = recent_attn.sum(dim=(0, 1)).float()
        else:
            logger.warning(f"Unexpected attention shape, falling back to SnapKV")
            return self._select_by_real_attention(prompt_length, num_keep, attention_scores)

        # Normalize importance to [0, 1]
        if importance.max() > importance.min():
            importance = (importance - importance.min()) / (importance.max() - importance.min())

        # ── Step 2: Compute redundancy (if key_states available) ─────
        if key_states is not None and key_states.shape[0] >= prompt_length:
            keys = key_states[:prompt_length].float()
            key_norm = F.normalize(keys, dim=-1)
            sim_matrix = torch.mm(key_norm, key_norm.t())  # (P, P)
            sim_matrix.fill_diagonal_(0.0)

            # Zero top-β most similar neighbors
            beta = config.r_kv_beta
            if beta > 0 and sim_matrix.shape[0] > beta:
                _, topk_idx = sim_matrix.topk(beta, dim=-1)
                sim_matrix.scatter_(1, topk_idx, 0.0)

            avg_sim = sim_matrix.mean(dim=-1)
            redundancy = F.softmax(avg_sim, dim=0)
        else:
            # No key states → redundancy = 0 (pure attention-based selection)
            if key_states is None:
                logger.debug("No key_states for R-KV redundancy, using attention only")
            redundancy = torch.zeros(prompt_length, dtype=torch.float32)

        # ── Step 3: Joint score Z = λ·I - (1-λ)·R ──────────────────
        lam = config.r_kv_lambda
        z_scores = lam * importance - (1 - lam) * redundancy.to(importance.device)

        # Boost protected positions
        max_z = z_scores.max().item() + 1.0
        z_scores[:config.sink_tokens] += max_z * 10
        z_scores[-config.observation_window:] += max_z * 5

        # Top-k selection
        _, top_indices = z_scores.topk(min(num_keep, prompt_length))
        return top_indices.sort().values

    def _select_by_position_heuristic(
        self,
        prompt_length: int,
        num_keep: int,
    ) -> torch.Tensor:
        """Fallback SnapKV approximation using position-based importance.

        Uses harmonic-mean distance: tokens closer to either end score higher.
        Used when real attention scores are unavailable (e.g., SGLang rollout).
        """
        config = self.config
        importance = torch.zeros(prompt_length, dtype=torch.float32)

        for i in range(prompt_length):
            dist_start = i + 1
            dist_end = prompt_length - i
            importance[i] = 2.0 / (1.0 / dist_start + 1.0 / dist_end)

            if i < config.sink_tokens:
                importance[i] += prompt_length  # Boost sink tokens
            if i >= prompt_length - config.observation_window:
                importance[i] += prompt_length * 0.5  # Boost recent tokens

        _, top_indices = importance.topk(num_keep)
        return top_indices.sort().values

    def generate_batch_shadow_masks(
        self,
        prompt_lengths: list[int],
        response_lengths: list[int],
        max_seq_len: int | None = None,
        seed: int | None = None,
    ) -> list[torch.Tensor]:
        """Generate shadow masks for an entire batch of samples.

        Args:
            prompt_lengths: List of prompt lengths, one per sample.
            response_lengths: List of response lengths, one per sample.
            max_seq_len: Optional maximum sequence length for padding (unused currently).
            seed: Random seed base; each sample uses seed + sample_index.

        Returns:
            List of (total_len, total_len) boolean tensors, one per sample.
        """
        masks = []
        for i, (pl, rl) in enumerate(zip(prompt_lengths, response_lengths)):
            sample_seed = seed + i if seed is not None else None
            mask = self.generate_shadow_mask(pl, rl, seed=sample_seed)
            masks.append(mask)
        return masks
