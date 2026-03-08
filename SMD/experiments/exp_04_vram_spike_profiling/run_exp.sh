#!/bin/bash
# ==========================================
# Exp_04: VRAM Spike Profiling — True KV Eviction Memory Analysis
# ==========================================
# PURPOSE: Measure peak GPU memory during native KV cache eviction to demonstrate
#          the not-in-place allocation spike (PyTorch tensor slicing creates a copy
#          before freeing the original).
#
# EXPECTED RESULT: True eviction INCREASES peak VRAM by ~0.017 GB (counter-intuitive).
# ==========================================

# ── Configuration (Modify here for scale-up) ───────────────────────────
MODEL_SIZE="1.5B"
MODEL_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/models"
DATA_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/datasets/tldr_train.jsonl"
CONTAINER="slime_shadow"
MAX_ROLLOUTS=5         # Only need a few rollouts to measure peak VRAM
MAX_RESP_LEN=128
BATCH_SIZE=2
LR="1e-7"
KL_COEF="0.5"

RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
# ── End Configuration ───────────────────────────────────────────────────

set -uo pipefail

run_vram_test() {
  local name=$1 extra_args=$2
  local exp_dir="${RESULTS_DIR}/${name}"
  local log_file="${exp_dir}/train.log"

  mkdir -p "${exp_dir}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${name}" | tee -a "${RESULTS_DIR}/run.log"

  local hf_ckpt="${MODEL_PATH}/Qwen2.5-${MODEL_SIZE}-Instruct"
  local td_ckpt="${MODEL_PATH}/Qwen2.5-${MODEL_SIZE}-Instruct_torch_dist"

  docker exec "${CONTAINER}" bash -c "
    set -e
    export MASTER_ADDR=127.0.0.1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONPATH=/root/Megatron-LM:\${PYTHONPATH:-}
    export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

    ray stop --force 2>/dev/null || true
    sleep 2
    rm -rf /tmp/ray/session_* 2>/dev/null || true
    ray start --head --node-ip-address \${MASTER_ADDR} --num-gpus 1 --disable-usage-stats 2>&1 | tail -3
    sleep 5

    source /root/slime/scripts/models/qwen2.5-${MODEL_SIZE}.sh
    MODEL_ARGS+=(--rotary-base 1000000)

    python3 /root/slime/train.py \\
       --actor-num-nodes 1 --actor-num-gpus-per-node 1 \\
       --colocate --no-offload-train --no-offload-rollout \\
       \${MODEL_ARGS[@]} \\
       --hf-checkpoint ${hf_ckpt} --ref-load ${td_ckpt} --load ${td_ckpt} \\
       --prompt-data ${DATA_PATH} \\
       --input-key prompt --label-key label --apply-chat-template --rm-type rouge \\
       --num-rollout ${MAX_ROLLOUTS} --rollout-batch-size 1 --n-samples-per-prompt 2 \\
       --rollout-max-response-len ${MAX_RESP_LEN} --rollout-temperature 1 \\
       --global-batch-size ${BATCH_SIZE} --micro-batch-size 1 \\
       --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \\
       --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \\
       --advantage-estimator grpo --eps-clip 0.2 --entropy-coef 0.01 --kl-coef ${KL_COEF} \\
       --optimizer adam --lr ${LR} --lr-decay-style constant --weight-decay 0.1 \\
       --attention-dropout 0.0 --hidden-dropout 0.0 \\
       --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \\
       --rollout-function-path slime.rollout.native_hf_rollout.generate_rollout \\
       --rollout-num-gpus-per-engine 1 \\
       ${extra_args}
  " > "${log_file}" 2>&1

  local exit_code=$?
  echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] ${name} (exit=${exit_code})" | tee -a "${RESULTS_DIR}/run.log"

  # Extract peak memory
  local peak=$(grep "PEAK MEMORY" "${log_file}" | tail -1 | grep -oP '[0-9.]+' | tail -1)
  echo "  Peak VRAM: ${peak:-N/A} GB" | tee -a "${RESULTS_DIR}/run.log"
  touch "${exp_dir}/DONE"
}

mkdir -p "${RESULTS_DIR}"
echo "=== Exp_04: VRAM Spike Profiling ($(date)) ===" | tee "${RESULTS_DIR}/run.log"

# 1. Dense baseline (no compression)
run_vram_test "dense_baseline" ""

# Restart container to get clean VRAM measurement
docker restart "${CONTAINER}" && sleep 15

# 2. SnapKV 50% with true eviction
run_vram_test "snapkv_50_eviction" "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv"

echo "=== Exp_04 Complete! $(date) ===" | tee -a "${RESULTS_DIR}/run.log"
