"""Recent Token KV Cache Eviction Strategy.

Compression strategy: keep only the most recent tokens plus sink tokens.
All middle-context tokens are evicted. This is the simplest baseline and
tends to perform worst on tasks requiring early-context information
(e.g., summarization titles, question context).
"""

import torch


class RecentSelector:
    """Recent-token selector for KV cache compression.

    Args:
        retention_ratio: Fraction of prompt KV positions to retain.
        sink_tokens: Number of initial tokens always retained (attention sinks).
    """

    def __init__(
        self,
        retention_ratio: float = 0.5,
        sink_tokens: int = 4,
    ):
        self.retention_ratio = retention_ratio
        self.sink_tokens = sink_tokens

    def select(self, prompt_length: int) -> torch.Tensor:
        """Select sink + most recent positions to retain.

        Args:
            prompt_length: Total number of prompt tokens.

        Returns:
            Sorted tensor of retained position indices.
        """
        num_keep = max(self.sink_tokens, int(prompt_length * self.retention_ratio))
        num_keep = min(num_keep, prompt_length)

        if num_keep >= prompt_length:
            return torch.arange(prompt_length)

        sink = torch.arange(min(self.sink_tokens, prompt_length))
        recent_start = max(self.sink_tokens, prompt_length - (num_keep - self.sink_tokens))
        recent = torch.arange(recent_start, prompt_length)

        return torch.cat([sink, recent]).unique().sort().values
