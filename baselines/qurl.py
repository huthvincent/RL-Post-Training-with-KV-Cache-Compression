"""QuRL Baseline: Quantized Reinforcement Learning with Update-Aware Quantization.

Core Idea:
    Use a low-precision quantized model for rollout (fast generation), but maintain
    full-precision for gradient updates. Two key mechanisms correct the mismatch:

    1. Update-Aware Quantization (UAQ):
       Before quantizing, apply invariant scaling to linear layers:
       W_scaled = W / s (s=1.5), fuse s into preceding LayerNorm.
       This amplifies gradient updates so they survive quantization noise.

    2. Decoupled PPO with Adaptive Clipping Range (ACR):
       Maintain separate behavior policy (quantized) and proximal policy (full-precision).
       When advantage > 0 and ratio r < 1, expand upper clip bound from (1+ε) to (1+ε)/r,
       preventing erroneous clipping due to quantization-induced policy difference.
"""

import logging
from argparse import Namespace
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)

# ── Hyperparameters ─────────────────────────────────────────────────────
UAQ_SCALE_FACTOR = 1.5  # Invariant scaling factor for weight amplification


def qurl_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """QuRL loss: Decoupled PPO with adaptive clipping for quantized rollout.

    The idea: rollout was done by a quantized actor, so π_behavior ≠ π_proximal.
    We detect this mismatch via the ratio r = π_proximal / π_behavior and
    adaptively expand the PPO clip range when the quantization pushes r < 1
    for positive-advantage tokens, preventing false clipping.
    """
    from slime.backends.megatron_utils.loss import get_log_probs_and_entropy

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    # Dense (full-precision) forward pass
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths, response_lengths=response_lengths,
        with_entropy=True, max_seq_lens=max_seq_lens,
    )
    log_probs = log_probs_and_entropy["log_probs"]
    entropy_list = log_probs_and_entropy["entropy"]

    # π_old is the policy at rollout time (used for standard PPO ratio)
    old_log_probs_list = (
        batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    )
    # π_behavior is the quantized policy's log-probs from rollout
    behavior_log_probs_list = batch.get("rollout_log_probs", old_log_probs_list)
    advantages_list = batch["advantages"]
    loss_masks_list = batch["loss_masks"]
    num_samples = len(log_probs)

    eps = getattr(args, "eps_clip", 0.2)

    pg_losses, pg_clipfracs, all_entropies = [], [], []
    acr_expansions = []  # Track how often ACR expands the clip range

    for i in range(num_samples):
        lp = log_probs[i]                                    # π_proximal (dense, current)
        old_lp = old_log_probs_list[i].to(lp.device)         # π_old (for PPO ratio)
        beh_lp = behavior_log_probs_list[i].to(lp.device)    # π_behavior (quantized)
        adv = advantages_list[i].to(lp.device)
        lm = loss_masks_list[i].float().to(lp.device)

        # Standard PPO ratio: r_ppo = π_current / π_old
        ratio = torch.exp(lp - old_lp)

        # Decoupled ratio: r_decouple = π_proximal / π_behavior
        # This measures the gap between full-precision and quantized policies
        r_decouple = torch.exp(lp.detach() - beh_lp)

        # Adaptive Clipping Range (ACR):
        # When advantage > 0 (want to increase probability) and r_decouple < 1
        # (quantized model had higher probability), the standard clip bound
        # (1+ε) is too tight — the dense model needs more room to grow.
        # Expand upper bound: (1+ε)/r_decouple
        upper_clip = torch.full_like(ratio, 1.0 + eps)
        needs_expansion = (adv > 0) & (r_decouple < 1.0)
        expanded_upper = (1.0 + eps) / r_decouple.clamp(min=0.5)  # Floor at 0.5 for stability
        upper_clip = torch.where(needs_expansion, expanded_upper, upper_clip)

        acr_expansion_rate = (needs_expansion.float() * lm).sum() / lm.sum().clamp(min=1)
        acr_expansions.append(acr_expansion_rate.detach())

        # Clipped surrogate with adaptive bounds
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1.0 - eps, upper_clip)
        pg_loss = torch.max(pg_loss1, pg_loss2) * lm

        pg_losses.append(pg_loss)
        pg_clipfracs.append(((ratio - 1.0).abs() > eps).float() * lm)
        if i < len(entropy_list):
            all_entropies.append(entropy_list[i] * lm)

    # Aggregate
    pg_loss = sum_of_sample_mean(torch.cat(pg_losses, dim=0))
    pg_clipfrac = sum_of_sample_mean(torch.cat(pg_clipfracs, dim=0))
    entropy_loss = sum_of_sample_mean(torch.cat(all_entropies, dim=0)) if all_entropies else torch.tensor(0.0, device=logits.device)

    total_loss = pg_loss - args.entropy_coef * entropy_loss
    if torch.cat(log_probs, dim=0).numel() == 0:
        total_loss += 0 * logits.sum()

    metrics = {
        "loss": total_loss.clone().detach(),
        "qurl/pg_loss": pg_loss.clone().detach(),
        "qurl/entropy": entropy_loss.clone().detach(),
        "qurl/clipfrac": pg_clipfrac.clone().detach(),
        "qurl/acr_expansion_rate": torch.stack(acr_expansions).mean() if acr_expansions else torch.tensor(0.0, device=logits.device),
        "qurl/uaq_scale": torch.tensor(UAQ_SCALE_FACTOR, device=logits.device),
    }
    return total_loss, metrics


def apply_uaq_scaling(model: torch.nn.Module, scale: float = UAQ_SCALE_FACTOR):
    """Apply Update-Aware Quantization scaling to model weights (pre-quantization).

    For each Transformer block: W_linear *= 1/s, LayerNorm.weight *= s.
    This ensures the effective computation is unchanged (invariant), but
    amplifies gradient updates relative to quantization noise.

    Call this BEFORE quantizing the model for rollout.

    Args:
        model: Transformer model
        scale: Scaling factor s (default: 1.5)
    """
    inv_scale = 1.0 / scale
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.mul_(inv_scale)
            if module.bias is not None:
                module.bias.data.mul_(inv_scale)
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            module.weight.data.mul_(scale)
    logger.info(f"Applied UAQ scaling with s={scale} to {sum(1 for _ in model.parameters())} parameters")


def revert_uaq_scaling(model: torch.nn.Module, scale: float = UAQ_SCALE_FACTOR):
    """Revert UAQ scaling after quantization (restore original weights).

    Args:
        model: Transformer model with UAQ scaling applied
        scale: Same scaling factor used in apply_uaq_scaling
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.mul_(scale)
            if module.bias is not None:
                module.bias.data.mul_(scale)
        elif isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            inv_scale = 1.0 / scale
            module.weight.data.mul_(inv_scale)
    logger.info(f"Reverted UAQ scaling with s={scale}")
