#!/bin/bash
# ==========================================
# Exp_06: Distillation Lambda Ablation (λ = 0, 0.1, 1.0)
# ==========================================
# PURPOSE: Evaluate the impact of Track 2 (KL distillation) weight.
#
# EXPECTED RESULT: λ=0.1 is optimal. λ=0 still works (shadow mask alone helps).
#                  λ=1.0 degrades performance (KL overwhelms RL signal).
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

RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
# ── End Configuration ───────────────────────────────────────────────────

set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../_shared/run_helper.sh"

mkdir -p "${RESULTS_DIR}"
echo "=== Exp_06: Distillation Lambda Ablation ($(date)) ===" | tee "${RESULTS_DIR}/run.log"

for lambda in 0.0 0.1 1.0; do
  run_sglang_experiment "snapkv_lambda${lambda}" \
    "--use-shadow-mask --shadow-retention-ratio ${COMPRESSION_RATIO} --shadow-strategy snapkv --shadow-distill-lambda ${lambda} --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.shadow_distillation_loss.shadow_distillation_loss_function" \
    "${DATA_PATH}" "rouge" "${MAX_ROLLOUTS}"
done

echo "=== Exp_06 Complete! $(date) ===" | tee -a "${RESULTS_DIR}/run.log"
