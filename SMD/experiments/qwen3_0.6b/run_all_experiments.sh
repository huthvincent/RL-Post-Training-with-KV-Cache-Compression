#!/bin/bash
# ==========================================
# Exp 1 & 2: Qwen3-0.6B on all 4 datasets
# ==========================================
# Usage:
#   bash run_all_experiments.sh [sanity|full]
#   sanity = 10 rollouts (quick test)
#   full   = 500 rollouts (full experiment)

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-sanity}"
CONTAINER="smd_training"

if [ "$MODE" = "sanity" ]; then
  NUM_ROLLOUTS=10
  echo "=== SANITY CHECK MODE: 10 rollouts ==="
elif [ "$MODE" = "full" ]; then
  NUM_ROLLOUTS=500
  echo "=== FULL MODE: 500 rollouts ==="
else
  echo "Usage: $0 [sanity|full]"; exit 1
fi

RESULTS_BASE="/workspace/RLKV/SMD/figures"
DATASETS="tldr gsm8k govreport hotpotqa"
LOG_FILE="${SCRIPT_DIR}/experiment_run.log"

run_one() {
  local exp=$1 dataset=$2 method=$3
  local output_dir="${RESULTS_BASE}/qwen3_0.6b_${exp}/${dataset}_${method}"
  local marker="${output_dir}/DONE"

  if [ -f "${marker}" ]; then
    echo "[SKIP] ${exp}/${dataset}_${method} — already done" | tee -a "${LOG_FILE}"
    return 0
  fi

  echo "$(date '+%H:%M:%S') [START] ${exp} | ${dataset} | ${method}" | tee -a "${LOG_FILE}"

  docker exec "${CONTAINER}" bash -c "
    python /workspace/RLKV/SMD/experiments/qwen3_0.6b/run_grpo_training.py \
      --dataset ${dataset} \
      --method ${method} \
      --num-rollouts ${NUM_ROLLOUTS} \
      --output-dir ${output_dir}
  " 2>&1 | tail -20

  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "$(date '+%H:%M:%S') [DONE ✅] ${exp}/${dataset}_${method}" | tee -a "${LOG_FILE}"
  else
    echo "$(date '+%H:%M:%S') [FAIL ❌] ${exp}/${dataset}_${method} (exit=$exit_code)" | tee -a "${LOG_FILE}"
  fi
}

echo "=== Qwen3-0.6B Experiments ($(date)) ===" | tee "${LOG_FILE}"

# ── Exp 01: Naive KV Collapse ──────────────────────────
echo "" | tee -a "${LOG_FILE}"
echo "=== EXP 01: Naive KV Collapse ===" | tee -a "${LOG_FILE}"
for ds in $DATASETS; do
  run_one "exp01" "$ds" "dense"
  run_one "exp01" "$ds" "naive_snapkv"
  run_one "exp01" "$ds" "naive_random"
done

# ── Exp 02: SOTA Showdown (with all baselines) ────────
echo "" | tee -a "${LOG_FILE}"
echo "=== EXP 02: SOTA Showdown ===" | tee -a "${LOG_FILE}"
for ds in $DATASETS; do
  run_one "exp02" "$ds" "dense"
  run_one "exp02" "$ds" "smd"
  run_one "exp02" "$ds" "sparse_rl"
  run_one "exp02" "$ds" "qurl"
  run_one "exp02" "$ds" "rlhfless"
done

echo "" | tee -a "${LOG_FILE}"
echo "=== All experiments complete! $(date) ===" | tee -a "${LOG_FILE}"
