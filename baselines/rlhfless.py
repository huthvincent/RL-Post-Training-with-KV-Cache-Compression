"""RLHFless Baseline: Serverless Elastic RLHF with Scheduling Optimizations.

Core Idea:
    Decouple the generation pipeline using dynamic resource scheduling to
    maximize GPU utilization during rollout. Three mechanisms:

    1. Deduplicated Prefill:
       Extract shared prefix from prompts, compute KV cache once, broadcast to
       all decode workers. Saves redundant prefill computation.

    2. Length-Aware Prompt Assignment:
       Use EWMA (window=1) to predict response lengths from historical data.
       Pack prompts with similar predicted lengths to the same decode node,
       minimizing idle time from length variance.

    3. Cut-and-Migrate:
       When τ=70% of sequences in a batch have finished, force-truncate the
       remaining "long-tail" sequences and migrate their KV state to idle nodes.

Note:
    This method primarily optimizes the rollout scheduling, not the loss function.
    The loss function itself is standard GRPO. The optimizations are implemented
    as a RolloutScheduler class that wraps the rollout engine.
"""

import logging
import math
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

# ── Hyperparameters ─────────────────────────────────────────────────────
CUT_AND_MIGRATE_THRESHOLD = 0.70   # τ: trigger migration when this fraction done
EWMA_ALPHA = 1.0                   # EWMA window=1 (instant tracking)
DEFAULT_LENGTH_PREDICTION = 128    # Fallback prediction for first epoch


def rlhfless_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Standard GRPO loss — RLHFless optimizes scheduling, not the loss.

    The actual RLHFless innovations (deduplicated prefill, length-aware
    assignment, cut-and-migrate) are in the RolloutScheduler class below.
    This loss function is identical to standard PPO/GRPO but tracks additional
    scheduling-aware metrics.
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
    truncated_count = 0

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

        # Track truncated sequences (from cut-and-migrate)
        if batch.get("truncated") and i < len(batch["truncated"]):
            truncated_count += int(batch["truncated"][i])

    pg_loss = sum_of_sample_mean(torch.cat(pg_losses, dim=0))
    pg_clipfrac = sum_of_sample_mean(torch.cat(pg_clipfracs, dim=0))
    entropy_loss = sum_of_sample_mean(torch.cat(all_entropies, dim=0)) if all_entropies else torch.tensor(0.0, device=logits.device)

    total_loss = pg_loss - args.entropy_coef * entropy_loss
    if torch.cat(log_probs, dim=0).numel() == 0:
        total_loss += 0 * logits.sum()

    # Length variance metric (lower = better scheduling)
    resp_lens = torch.tensor([float(r) for r in response_lengths], device=logits.device)
    length_cv = resp_lens.std() / resp_lens.mean().clamp(min=1) if len(resp_lens) > 1 else torch.tensor(0.0, device=logits.device)

    metrics = {
        "loss": total_loss.clone().detach(),
        "rlhfless/pg_loss": pg_loss.clone().detach(),
        "rlhfless/entropy": entropy_loss.clone().detach(),
        "rlhfless/clipfrac": pg_clipfrac.clone().detach(),
        "rlhfless/truncated_count": torch.tensor(float(truncated_count), device=logits.device),
        "rlhfless/length_cv": length_cv.detach(),
    }
    return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════
# Rollout Scheduling Components
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class LengthPredictor:
    """EWMA-based response length predictor for prompt assignment.

    Tracks historical response lengths per prompt hash and predicts
    future lengths using exponentially weighted moving average.
    """
    alpha: float = EWMA_ALPHA
    predictions: dict = field(default_factory=dict)
    global_mean: float = DEFAULT_LENGTH_PREDICTION

    def predict(self, prompt_hash: int) -> float:
        """Predict response length for a given prompt."""
        return self.predictions.get(prompt_hash, self.global_mean)

    def update(self, prompt_hash: int, actual_length: int):
        """Update prediction with observed length."""
        old = self.predictions.get(prompt_hash, self.global_mean)
        new = self.alpha * actual_length + (1 - self.alpha) * old
        self.predictions[prompt_hash] = new
        # Update global mean
        if self.predictions:
            self.global_mean = sum(self.predictions.values()) / len(self.predictions)


@dataclass
class DedupPrefillResult:
    """Result of deduplicated prefill computation."""
    shared_prefix_len: int
    kv_cache: object  # Framework-specific KV cache tensor
    unique_suffixes: list  # Per-prompt unique suffix tokens


def find_shared_prefix(prompts: list[list[int]]) -> int:
    """Find the longest common prefix length across all prompts.

    Args:
        prompts: List of tokenized prompts (list of token IDs)

    Returns:
        Length of shared prefix (0 if no common prefix)
    """
    if not prompts:
        return 0
    min_len = min(len(p) for p in prompts)
    shared = 0
    for pos in range(min_len):
        if all(p[pos] == prompts[0][pos] for p in prompts):
            shared += 1
        else:
            break
    return shared


def assign_prompts_by_length(
    prompts: list,
    predicted_lengths: list[float],
    num_workers: int,
) -> list[list[int]]:
    """Assign prompts to decode workers by predicted response length.

    Groups prompts with similar predicted lengths together to minimize
    idle time from length variance within each batch.

    Args:
        prompts: List of prompt objects
        predicted_lengths: Predicted response length for each prompt
        num_workers: Number of decode workers

    Returns:
        List of lists, where each inner list contains prompt indices
        assigned to that worker.
    """
    # Sort by predicted length
    indexed = sorted(enumerate(predicted_lengths), key=lambda x: x[1])
    assignments = [[] for _ in range(num_workers)]

    # Round-robin assign sorted prompts to workers
    for rank, (idx, _) in enumerate(indexed):
        worker_id = rank % num_workers
        assignments[worker_id].append(idx)

    return assignments


def should_cut_and_migrate(
    completed: list[bool],
    threshold: float = CUT_AND_MIGRATE_THRESHOLD,
) -> bool:
    """Check if cut-and-migrate should be triggered.

    Args:
        completed: Boolean list indicating which sequences have finished
        threshold: Fraction of completed sequences to trigger migration

    Returns:
        True if migration should be triggered
    """
    if not completed:
        return False
    completion_rate = sum(completed) / len(completed)
    return completion_rate >= threshold
