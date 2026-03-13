"""Random KV Cache Eviction Strategy.

Baseline compression strategy: randomly select which prompt tokens to retain.
Protected positions (sink tokens and recent observation window) are always kept.
"""

import torch


class RandomSelector:
    """Random token selector for KV cache compression.

    Args:
        retention_ratio: Fraction of prompt KV positions to retain.
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
        seed: int | None = None,
    ) -> torch.Tensor:
        """Randomly select which prompt positions to retain.

        Args:
            prompt_length: Total number of prompt tokens.
            seed: Optional random seed for reproducibility.

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

        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        # Positions that are always protected
        always_keep = set(range(min(self.sink_tokens, prompt_length)))
        always_keep.update(range(max(0, prompt_length - self.observation_window), prompt_length))

        # Random selection from middle positions
        middle = [i for i in range(prompt_length) if i not in always_keep]
        num_random = max(0, num_keep - len(always_keep))
        if num_random > 0 and middle:
            perm = torch.randperm(len(middle), generator=gen)
            selected = [middle[p] for p in perm[:num_random]]
            always_keep.update(selected)

        return torch.tensor(sorted(always_keep), dtype=torch.long)
