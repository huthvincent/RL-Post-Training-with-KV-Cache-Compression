#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Exp 08: Quantization Mismatch vs KV Compression Mismatch
# ═══════════════════════════════════════════════════════════════
#
# Question: Which mismatch is more severe — INT8 quantization
# (QuRL) or KV cache compression (SMD with R-KV)?
#
# Setup:
#   - QuRL: INT8 quantized rollout, dense training
#   - SMD:  R-KV 50% compressed rollout, shadow mask training
#   - Dense baseline: no mismatch
#
# Model: Qwen3-1.7B | Dataset: TL;DR | 500 rollouts
# ═══════════════════════════════════════════════════════════════
set -euo pipefail
source "$(dirname "$0")/../_shared/run_helper.sh"

EXPERIMENT="exp_08_quant_vs_kv"
METHODS=("dense" "smd" "qurl")

for method in "${METHODS[@]}"; do
    run_experiment "$EXPERIMENT" "$method" "--dataset tldr --num-rollouts 500"
done
