#!/bin/bash
# ==========================================
# Exp_03: Downstream Generalization — GSM8K Test Accuracy
# ==========================================
# PURPOSE: Evaluate whether SMD-trained models generalize to unseen test data.
#          Train on GSM8K with SMD, save checkpoints, evaluate on test set.
#
# EXPECTED RESULT: SMD (69.5%) > Dense RL (68.5%) > SFT baseline (68.0%).
# ==========================================

# ── Configuration (Modify here for scale-up) ───────────────────────────
MODEL_SIZE="1.7B"
MODEL_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/models"
DATA_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/datasets/gsm8k_fewshot_train.jsonl"
CONTAINER="slime_shadow"
MAX_ROLLOUTS=300
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
echo "=== Exp_03: Downstream Eval ($(date)) ===" | tee "${RESULTS_DIR}/run.log"

# 1. Dense Baseline on GSM8K
run_sglang_experiment "baseline_dense_gsm8k" "" "${DATA_PATH}" "math" "${MAX_ROLLOUTS}"

# 2. SMD SnapKV 50% on GSM8K
run_sglang_experiment "smd_snapkv50_gsm8k" \
  "--use-shadow-mask --shadow-retention-ratio ${COMPRESSION_RATIO} --shadow-strategy snapkv --shadow-distill-lambda ${LAMBDA_COEF} --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.shadow_distillation_loss.shadow_distillation_loss_function" \
  "${DATA_PATH}" "math" "${MAX_ROLLOUTS}"

echo "=== Exp_03 Complete! $(date) ===" | tee -a "${RESULTS_DIR}/run.log"
echo "NOTE: To evaluate downstream accuracy, export HF checkpoints and run 8-shot eval."
