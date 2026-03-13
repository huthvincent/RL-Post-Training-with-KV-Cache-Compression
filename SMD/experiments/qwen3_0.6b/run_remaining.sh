#!/bin/bash
# Auto-run all remaining Exp 01 + Exp 02 experiments for Qwen3-0.6B
# Usage: nohup bash run_remaining.sh > /tmp/exp_remaining.log 2>&1 &
set -uo pipefail

CONTAINER="smd_training"
SCRIPT="/workspace/RLKV/SMD/experiments/qwen3_0.6b/run_grpo_training.py"
FIGURES="/workspace/RLKV/SMD/figures"
NUM_ROLLOUTS=500

run_one() {
  local exp=$1 dataset=$2 method=$3
  local out="${FIGURES}/qwen3_0.6b_${exp}/${dataset}_${method}"
  if [ -f "$(echo $out | sed 's|/workspace/RLKV|/data2/zhu11/RL-Post-Training-with-KV-Cache-Compression|')/metrics.json" ]; then
    echo "[SKIP] ${exp}/${dataset}_${method} — already done"
    return 0
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${exp} | ${dataset} | ${method}"
  docker exec "${CONTAINER}" python "${SCRIPT}" \
    --dataset "${dataset}" --method "${method}" \
    --num-rollouts ${NUM_ROLLOUTS} --output-dir "${out}" 2>&1 | \
    grep -E "Step|Training complete|Total time|Error|Traceback" | tail -5
  echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE]  ${exp} | ${dataset} | ${method} (exit=$?)"
  echo ""
}

DATASETS="tldr gsm8k govreport hotpotqa"

echo "=== Starting remaining experiments $(date) ==="

# Exp 01: Dense + Naive SnapKV + Naive Random × 4 datasets
echo "--- EXP 01 ---"
for ds in $DATASETS; do
  run_one "exp01" "$ds" "dense"
  run_one "exp01" "$ds" "naive_snapkv"
  run_one "exp01" "$ds" "naive_random"
done

# Exp 02: Dense + SMD + Sparse-RL + QuRL + RLHFless × 4 datasets
echo "--- EXP 02 ---"
for ds in $DATASETS; do
  run_one "exp02" "$ds" "dense"
  run_one "exp02" "$ds" "smd"
  run_one "exp02" "$ds" "sparse_rl"
  run_one "exp02" "$ds" "qurl"
  run_one "exp02" "$ds" "rlhfless"
done

echo "=== All experiments complete! $(date) ==="
