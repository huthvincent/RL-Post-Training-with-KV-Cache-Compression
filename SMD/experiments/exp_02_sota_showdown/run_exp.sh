#!/bin/bash
# ==========================================
# Exp_02: SOTA Showdown — SMD vs Sparse-RL vs Dense Baseline
# ==========================================
# PURPOSE: Head-to-head comparison of SMD against Sparse-RL (arXiv:2601.10079)
#          and the dense baseline on Reddit TL;DR summarization.
#
# EXPECTED RESULT: SMD achieves 63% higher avg ROUGE-L than Sparse-RL.
# ==========================================

# ── Configuration (Modify here for scale-up) ───────────────────────────
MODEL_SIZE="1.5B"
MODEL_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/models"
DATA_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/datasets/tldr_train.jsonl"
CONTAINER="slime_shadow"
MAX_ROLLOUTS=500
MAX_RESP_LEN=256
BATCH_SIZE=2
LR="1e-7"
KL_COEF="0.5"
COMPRESSION_RATIO=0.5
LAMBDA_COEF=0.1

RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
# ── End Configuration ───────────────────────────────────────────────────

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_shared/run_helper.sh"

mkdir -p "${RESULTS_DIR}"
echo "=== Exp_02: SOTA Showdown ($(date)) ===" | tee "${RESULTS_DIR}/run.log"

# 1. Dense Baseline (no compression, no shadow mask)
run_sglang_experiment "baseline_dense" "" "${DATA_PATH}" "rouge" "${MAX_ROLLOUTS}"

# 2. SMD SnapKV 50%
run_sglang_experiment "smd_snapkv50" \
  "--use-shadow-mask --shadow-retention-ratio ${COMPRESSION_RATIO} --shadow-strategy snapkv --shadow-distill-lambda ${LAMBDA_COEF} --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.shadow_distillation_loss.shadow_distillation_loss_function" \
  "${DATA_PATH}" "rouge" "${MAX_ROLLOUTS}"

# 3. Sparse-RL Baseline
run_sglang_experiment "sparse_rl" \
  "--use-shadow-mask --shadow-retention-ratio ${COMPRESSION_RATIO} --shadow-strategy snapkv --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.sparse_rl_loss.sparse_rl_loss_function" \
  "${DATA_PATH}" "rouge" "${MAX_ROLLOUTS}"

echo "=== Exp_02 Complete! $(date) ===" | tee -a "${RESULTS_DIR}/run.log"
