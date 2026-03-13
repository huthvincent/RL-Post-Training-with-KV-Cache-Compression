#!/bin/bash
# ==========================================
# Exp 01: Naive KV Compression → Reward Collapse
# Model: Qwen3-1.7B | Dataset: TL;DR
# ==========================================
# Usage: bash run_exp01.sh [sanity|full]

set -uo pipefail
CONTAINER="smd_training"
SCRIPT="/workspace/RLKV/SMD/experiments/exp01_qwen3_1.7b/run_exp01.py"
FIGURES="/workspace/RLKV/SMD/figures/exp01"
MODE="${1:-sanity}"

if [ "$MODE" = "sanity" ]; then
  N=10; echo "=== SANITY (10 rollouts) ==="
elif [ "$MODE" = "full" ]; then
  N=500; echo "=== FULL (500 rollouts) ==="
else
  echo "Usage: $0 [sanity|full]"; exit 1
fi

METHODS="dense naive_snapkv naive_random naive_recent naive_r_kv"

for method in $METHODS; do
  out="${FIGURES}/${method}"
  host_out="$(echo $out | sed 's|/workspace/RLKV|/data2/zhu11/RL-Post-Training-with-KV-Cache-Compression|')"
  if [ -f "${host_out}/metrics.json" ]; then
    echo "[SKIP] ${method} — already done"
    continue
  fi
  echo "$(date '+%H:%M:%S') [START] ${method}"
  docker exec "${CONTAINER}" python "${SCRIPT}" \
    --method "${method}" --num-rollouts ${N} --output-dir "${out}" 2>&1 | \
    grep -E "Step|Done|Error|Traceback" | tail -5
  echo "$(date '+%H:%M:%S') [DONE] ${method}"
  echo ""
done

echo "=== Exp 01 complete! ==="
