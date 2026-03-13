#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Exp 09: PPO vs GRPO for Shadow Mask Distillation
# ═══════════════════════════════════════════════════════════════
#
# Question: How does the advantage estimation method affect SMD?
#   - GRPO: group-relative advantage (no critic network)
#   - PPO:  critic network for value estimation
#
# Setup: R-KV 50% + λ=0.1, vary advantage method
# Model: Qwen3-1.7B | Dataset: TL;DR | 500 rollouts
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
source "$(dirname "$0")/../_shared/run_helper.sh"

EXPERIMENT="exp_09_ppo_vs_grpo"
METHODS=("smd_grpo" "smd_ppo")

for method in "${METHODS[@]}"; do
    run_experiment "$EXPERIMENT" "$method" \
        "--dataset tldr --num-rollouts 500 --shadow-strategy r_kv --shadow-retention-ratio 0.5 --shadow-distill-lambda 0.1"
done
