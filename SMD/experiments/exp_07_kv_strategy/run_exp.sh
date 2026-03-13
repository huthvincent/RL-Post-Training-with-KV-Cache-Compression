#!/bin/bash
# ==========================================
# Exp_07: KV Selection Strategy Comparison (SnapKV / Random / Recent)
# ==========================================
# PURPOSE: Compare different token selection algorithms for shadow mask generation.
#
# EXPECTED RESULT: SnapKV >> Random ≈ Recent.
#                  Attention-guided selection creates a semantically meaningful bottleneck.
# ==========================================

# ── Configuration (Modify here for scale-up) ───────────────────────────
MODEL_SIZE="1.7B"
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
echo "=== Exp_07: KV Strategy Comparison ($(date)) ===" | tee "${RESULTS_DIR}/run.log"

for strategy in snapkv random recent; do
  run_sglang_experiment "${strategy}_r0.5" \
    "--use-shadow-mask --shadow-retention-ratio ${COMPRESSION_RATIO} --shadow-strategy ${strategy} --shadow-distill-lambda ${LAMBDA_COEF} --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.shadow_distillation_loss.shadow_distillation_loss_function" \
    "${DATA_PATH}" "rouge" "${MAX_ROLLOUTS}"
done

echo "=== Exp_07 Complete! $(date) ===" | tee -a "${RESULTS_DIR}/run.log"
