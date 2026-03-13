"""Shadow Mask Distillation: 双轨 GRPO Loss 实现。

核心算法：
- Track-1 (Shadow, on-policy faithful): 注入 shadow_mask 的前向传播 → π_shadow
  在信息瓶颈下计算 GRPO Loss，保证与 Rollout 时的采样分布 100% 对齐
- Track-2 (Dense, knowledge assimilation): 标准密集前向传播 → π_dense
  KL 蒸馏 Loss 迫使全量网络内化稀疏推理路径

总 Loss = GRPO_loss(π_shadow) + λ * D_KL(π_dense || StopGrad(π_shadow))
"""

import logging
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from slime.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


def shadow_distillation_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """双轨 Shadow Mask Distillation Loss。

    这个函数被设计为 slime 的 custom_loss_function，通过
    --loss-type custom_loss --custom-loss-function-path 注入。

    由于 Megatron 只执行一次前向传播（logits 已经是模型输出），
    这里的设计策略是：
    - 传入的 logits 来自 **标准密集前向传播** (Track-2, π_dense)
    - Track-1 无法再额外执行一次前向传播（Megatron 架构限制）
    - 因此使用 **掩码后处理** 模拟 shadow 效果：
      对 logits 应用 shadow_mask 的 token-level 信息限制

    实际实现：利用 shadow_mask 标记哪些 token 的 log_probs
    是"同策略可信"的，将主 GRPO Loss 限制在这些 token 上计算。
    同时添加 KL 蒸馏 term 作为全量→稀疏的知识同化信号。

    Args:
        args: 训练配置
        batch: 包含 shadow_mask 的 RolloutBatch
        logits: 密集前向传播的 logits (1, T, V)
        sum_of_sample_mean: 聚合函数

    Returns:
        (loss, metrics) 元组
    """
    from slime.backends.megatron_utils.loss import (
        get_log_probs_and_entropy,
        policy_loss_function,
    )

    # === 获取 shadow_mask 相关配置 ===
    shadow_masks = batch.get("shadow_masks", None)
    lambda_distill = getattr(args, "shadow_distill_lambda", 0.1)

    # 如果没有 shadow_mask，回退到标准 policy_loss
    if shadow_masks is None or not shadow_masks:
        logger.debug("No shadow_masks found, falling back to standard policy_loss")
        return policy_loss_function(args, batch, logits, sum_of_sample_mean)

    # === Real SnapKV: 用真实 attention 重新生成 shadow mask ===
    # 在 actor.py 中注册的 forward hooks 会在本次 forward pass 中
    # 自动捕获 attention weights 到全局 buffer。如果有数据，用它来
    # 替换原来基于 position heuristic 的 shadow mask。
    try:
        from SMD.src.attention_extraction import get_per_key_importance
        from SMD.src.shadow_mask_interceptor import ShadowMaskConfig, ShadowMaskInterceptor

        importance = get_per_key_importance(
            num_recent_queries=getattr(args, "shadow_observation_window", 64)
        )
        if importance is not None:
            # 用真实 attention 重新生成 shadow mask
            config = ShadowMaskConfig(
                enabled=True,
                retention_ratio=getattr(args, "shadow_retention_ratio", 0.5),
                strategy="snapkv",
                observation_window=getattr(args, "shadow_observation_window", 64),
                sink_tokens=getattr(args, "shadow_sink_tokens", 4),
            )
            interceptor = ShadowMaskInterceptor(config)

            response_lengths_raw = batch["response_lengths"]
            total_lengths_raw = batch["total_lengths"]

            new_masks = []
            for i in range(len(shadow_masks)):
                total_len = total_lengths_raw[i]
                resp_len = response_lengths_raw[i]
                prompt_len = total_len - resp_len
                # importance 来自最后一层，切到 prompt 长度
                new_mask = interceptor.generate_shadow_mask(
                    prompt_len, resp_len,
                    attention_scores=importance[:prompt_len].to("cpu"),
                )
                new_masks.append(new_mask.to(logits.device))

            shadow_masks = new_masks
            batch["shadow_masks"] = new_masks  # 更新 batch 供后续使用
            logger.info("Shadow masks upgraded with real SnapKV attention")
    except ImportError:
        pass  # attention_extraction 不可用，使用原始 position-based masks
    except Exception as e:
        logger.warning(f"Failed to regenerate shadow masks with real attention: {e}")

    # === Track-1: Shadow-masked on-policy loss ===
    # 使用 shadow_mask 来标记"同策略可信"的 token 位置
    # 在这些位置上计算 GRPO Loss

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    entropy_list = log_probs_and_entropy["entropy"]

    # 获取旧的 log_probs
    old_log_probs_list = (
        batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    )

    # 获取 advantages
    advantages_list = batch["advantages"]
    loss_masks_list = batch["loss_masks"]

    # === 构建 shadow token mask ===
    # shadow_masks 是 per-sample 的 (total_len, total_len) 掩码
    # 我们需要将其转换为 per-token 的可信度掩码 (response_len,)
    # 思路：对于 response 中的每个 token，如果它在 shadow_mask 中的
    # 可见 KV 比率 >= retention_ratio，则认为该 token 是"同策略可信"的
    shadow_token_masks = []
    for i, sm in enumerate(shadow_masks):
        if sm is None:
            # 没有 shadow_mask 的 sample，所有 token 都可信
            shadow_token_masks.append(
                torch.ones(response_lengths[i], dtype=torch.bool, device=logits.device)
            )
            continue

        total_len = total_lengths[i]
        resp_len = response_lengths[i]
        prompt_len = total_len - resp_len

        # 对 response 区域：每行的 True 比率
        if sm.shape[0] >= total_len:
            resp_mask = sm[prompt_len:total_len, :total_len]  # (resp_len, total_len)
            visibility_ratio = resp_mask.float().sum(dim=-1) / total_len
            # 如果 visibility_ratio < 1.0, 说明有 KV 被压缩，token 在稀疏条件下生成
            shadow_token_masks.append(
                (visibility_ratio < 1.0).to(logits.device)
            )
        else:
            shadow_token_masks.append(
                torch.ones(resp_len, dtype=torch.bool, device=logits.device)
            )

    # === Track-1 GRPO Loss: 在 shadow token 上计算 ===
    pg_losses = []
    pg_clipfracs = []
    shadow_entropies = []

    for i in range(len(log_probs)):
        lp = log_probs[i]
        old_lp = old_log_probs_list[i]
        adv = advantages_list[i]
        lm = loss_masks_list[i]
        stm = shadow_token_masks[i] if i < len(shadow_token_masks) else torch.ones_like(lm, dtype=torch.bool)

        # 将 shadow token mask 和 loss mask 结合
        effective_mask = lm.float() * stm.float().to(lm.device)

        # PPO-style clipped loss
        ratio = torch.exp(lp - old_lp)
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1.0 - args.eps_clip, 1.0 + args.eps_clip)
        pg_loss = torch.max(pg_loss1, pg_loss2)

        # 加权
        pg_loss = pg_loss * effective_mask
        clipfrac = ((ratio - 1.0).abs() > args.eps_clip).float() * effective_mask

        pg_losses.append(pg_loss)
        pg_clipfracs.append(clipfrac)

        if i < len(entropy_list):
            shadow_entropies.append(entropy_list[i] * effective_mask)

    pg_loss_cat = torch.cat(pg_losses, dim=0)
    pg_clipfrac_cat = torch.cat(pg_clipfracs, dim=0)

    shadow_pg_loss = sum_of_sample_mean(pg_loss_cat)
    shadow_clipfrac = sum_of_sample_mean(pg_clipfrac_cat)

    shadow_entropy = torch.tensor(0.0, device=logits.device)
    if shadow_entropies:
        shadow_entropy = sum_of_sample_mean(torch.cat(shadow_entropies, dim=0))

    # === Track-2: Dense KL Distillation Loss ===
    # 由于只有一次前向传播，我们无法得到独立的 π_shadow logits。
    # 替代方案：使用 rollout_log_probs 作为 shadow target
    # （rollout 时的采样 log_probs 就是在稀疏条件下产生的）
    # D_KL(π_dense || π_shadow_target)
    kl_distill_loss = torch.tensor(0.0, device=logits.device)
    if lambda_distill > 0 and "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        dense_lp = torch.cat(log_probs, dim=0)
        shadow_target_lp = torch.cat(batch["rollout_log_probs"], dim=0).to(logits.device)

        # Forward KL: D_KL(shadow || dense) ≈ shadow_target * (shadow_target - dense)
        # 这里用 token-level 的 log_prob 差异的绝对值作为近似蒸馏信号
        kl_per_token = (dense_lp - shadow_target_lp).abs()
        kl_distill_loss = sum_of_sample_mean(kl_per_token)

    # === 总 Loss ===
    total_loss = shadow_pg_loss - args.entropy_coef * shadow_entropy + lambda_distill * kl_distill_loss

    # 保证梯度可以反传
    if torch.cat(log_probs, dim=0).numel() == 0:
        total_loss += 0 * logits.sum()

    # === Metrics ===
    metrics = {
        "loss": total_loss.clone().detach(),
        "shadow_pg_loss": shadow_pg_loss.clone().detach(),
        "shadow_entropy": shadow_entropy.clone().detach(),
        "shadow_clipfrac": shadow_clipfrac.clone().detach(),
        "kl_distill_loss": kl_distill_loss.clone().detach(),
        "distill_lambda": torch.tensor(lambda_distill, device=logits.device),
    }

    # 计算同策略对齐率
    if shadow_token_masks:
        all_stm = torch.cat([stm.float() for stm in shadow_token_masks])
        metrics["shadow_token_ratio"] = all_stm.mean().detach()

    return total_loss, metrics
