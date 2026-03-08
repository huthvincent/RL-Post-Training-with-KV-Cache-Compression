#!/bin/bash
# ==========================================
# Exp_05: Compression Ratio Ablation (25% / 50% / 75% / 100%)
# ==========================================
# PURPOSE: Find the optimal KV retention ratio for SMD.
#
# EXPECTED RESULT: 50% is the sweet spot (U-shaped curve).
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
LAMBDA_COEF=0.1

RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
# ── End Configuration ───────────────────────────────────────────────────

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_shared/run_helper.sh"

mkdir -p "${RESULTS_DIR}"
echo "=== Exp_05: Compression Ratio Ablation ($(date)) ===" | tee "${RESULTS_DIR}/run.log"

SMD_ARGS_TEMPLATE="--use-shadow-mask --shadow-strategy snapkv --shadow-distill-lambda ${LAMBDA_COEF} --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.shadow_distillation_loss.shadow_distillation_loss_function"

# Dense baseline (ratio=1.0, no compression)
run_sglang_experiment "dense_r1.0" "" "${DATA_PATH}" "rouge" "${MAX_ROLLOUTS}"

for ratio in 0.25 0.50 0.75; do
  run_sglang_experiment "snapkv_r${ratio}" \
    "--shadow-retention-ratio ${ratio} ${SMD_ARGS_TEMPLATE}" \
    "${DATA_PATH}" "rouge" "${MAX_ROLLOUTS}"
done

echo "=== Exp_05 Complete! $(date) ===" | tee -a "${RESULTS_DIR}/run.log"
