"""Sparse-RL Baseline: Post-hoc statistical correction for off-policy RL.

Reference: arXiv:2601.10079

Mechanism A — Rejection Sampling:
    Computes sequence-level divergence between dense π_dense and sparse π_sparse.
    Top REJECTION_RATIO (20%) most divergent rollouts are discarded.

Mechanism B — Importance Reweighting:
    Per-token importance ratios ρ = π_dense / π_sparse, clipped and used to
    reweight GRPO advantages.

Key Limitation:
    Rejection sampling discards 20% of expensive rollout compute.
"""

import logging
from argparse import Namespace
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

# ── Hyperparameters ─────────────────────────────────────────────────────
REJECTION_RATIO = 0.20
IMPORTANCE_CLIP_LOW = 0.8
IMPORTANCE_CLIP_HIGH = 1.2


def sparse_rl_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sparse-RL GRPO loss with rejection sampling + importance reweighting."""
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    # Step 1: Dense log-probs
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
    rollout_lp_list = batch.get("rollout_log_probs", old_log_probs_list)

    # Step 2: Rejection sampling — measure divergence per rollout
    sample_divergences = []
    for i in range(num_samples):
        dense_lp = log_probs[i].detach()
        sparse_lp = rollout_lp_list[i].to(dense_lp.device)
        mask = loss_masks_list[i].float().to(dense_lp.device)
        diff = (dense_lp - sparse_lp).abs() * mask
        avg_div = diff.sum() / mask.sum().clamp(min=1)
        sample_divergences.append(avg_div)

    divergences = torch.stack(sample_divergences)
    num_reject = max(1, int(num_samples * REJECTION_RATIO))
    num_keep = num_samples - num_reject

    if num_samples > 1:
        _, sorted_indices = torch.sort(divergences)
        keep_set = set(sorted_indices[:num_keep].tolist())
        reject_mask = torch.tensor(
            [1.0 if i in keep_set else 0.0 for i in range(num_samples)],
            device=logits.device,
        )
    else:
        reject_mask = torch.ones(num_samples, device=logits.device)

    # Step 3: Importance reweighting + PPO loss
    pg_losses, pg_clipfracs, all_entropies, importance_ratios_all = [], [], [], []

    for i in range(num_samples):
        lp = log_probs[i]
        old_lp = old_log_probs_list[i].to(lp.device)
        sparse_lp = rollout_lp_list[i].to(lp.device)
        adv = advantages_list[i].to(lp.device)
        lm = loss_masks_list[i].float().to(lp.device)
        w = reject_mask[i]

        rho = torch.exp(lp.detach() - sparse_lp)
        clipped_rho = torch.clamp(rho, IMPORTANCE_CLIP_LOW, IMPORTANCE_CLIP_HIGH)
        importance_ratios_all.append(rho.detach())

        reweighted_adv = adv * clipped_rho
        ratio = torch.exp(lp - old_lp)
        pg_loss1 = -reweighted_adv * ratio
        pg_loss2 = -reweighted_adv * torch.clamp(ratio, 1.0 - args.eps_clip, 1.0 + args.eps_clip)
        pg_loss = torch.max(pg_loss1, pg_loss2) * lm * w

        pg_losses.append(pg_loss)
        pg_clipfracs.append(((ratio - 1.0).abs() > args.eps_clip).float() * lm * w)
        if i < len(entropy_list):
            all_entropies.append(entropy_list[i] * lm * w)

    # Step 4: Aggregate
    pg_loss = sum_of_sample_mean(torch.cat(pg_losses, dim=0))
    pg_clipfrac = sum_of_sample_mean(torch.cat(pg_clipfracs, dim=0))
    entropy_loss = sum_of_sample_mean(torch.cat(all_entropies, dim=0)) if all_entropies else torch.tensor(0.0, device=logits.device)

    total_loss = pg_loss - args.entropy_coef * entropy_loss
    if torch.cat(log_probs, dim=0).numel() == 0:
        total_loss += 0 * logits.sum()

    all_rho = torch.cat(importance_ratios_all)
    metrics = {
        "loss": total_loss.clone().detach(),
        "sparse_rl/pg_loss": pg_loss.clone().detach(),
        "sparse_rl/entropy": entropy_loss.clone().detach(),
        "sparse_rl/clipfrac": pg_clipfrac.clone().detach(),
        "sparse_rl/rejection_rate": torch.tensor(1.0 - reject_mask.mean().item(), device=logits.device),
        "sparse_rl/importance_ratio_mean": all_rho.mean(),
        "sparse_rl/importance_ratio_std": all_rho.std() if all_rho.numel() > 1 else torch.tensor(0.0, device=logits.device),
        "sparse_rl/avg_divergence": divergences.mean().detach(),
    }
    return total_loss, metrics
