#!/bin/bash
# ==========================================
# Exp_01: Naive KV Compression → Total Reward Collapse
# ==========================================
# PURPOSE: Demonstrate that naive KV compression (true physical eviction)
#          without SMD correction causes complete learning failure (zero reward).
#          This is the "motivating failure" that justifies the SMD method.
#
# WHAT'S SPECIAL: Uses native HuggingFace rollout engine (NOT SGLang) to perform
#                 true physical KV cache eviction via tensor slicing.
#
# EXPECTED RESULT: Naive SnapKV/Random → reward = 0 for all 500 steps.
#                  Dense baseline → normal learning curve.
# ==========================================

# ── Configuration (Modify here for scale-up) ───────────────────────────
MODEL_SIZE="1.5B"                                                       # Change to 7B/14B
MODEL_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/models"      # HF model root
DATA_PATH_TLDR="/home/zhu11/RLKV/RLKV_github/shared_resources/datasets/tldr_train.jsonl"
DATA_PATH_GSM8K="/home/zhu11/RLKV/RLKV_github/shared_resources/datasets/gsm8k_fewshot_train.jsonl"
CONTAINER="slime_shadow"
MAX_ROLLOUTS=500
MAX_RESP_LEN_TLDR=128
MAX_RESP_LEN_GSM8K=256
BATCH_SIZE=2
LR="1e-7"
KL_COEF="0.5"

# Results directory
RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
# ── End Configuration ───────────────────────────────────────────────────

set -uo pipefail

run_naive_experiment() {
  local name=$1 strategy=$2 ratio=$3 data_file=$4 num_rollout=$5 max_resp_len=$6 rm_type=$7
  local exp_dir="${RESULTS_DIR}/${name}"
  local log_file="${exp_dir}/train.log"

  if [ -f "${exp_dir}/DONE" ]; then
    echo "[SKIP] ${name} — already done"
    return 0
  fi

  mkdir -p "${exp_dir}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${name}" | tee -a "${RESULTS_DIR}/run.log"

  local hf_ckpt="${MODEL_PATH}/Qwen2.5-${MODEL_SIZE}-Instruct"
  local td_ckpt="${MODEL_PATH}/Qwen2.5-${MODEL_SIZE}-Instruct_torch_dist"

  local shadow_args="--use-shadow-mask --shadow-retention-ratio ${ratio} --shadow-strategy ${strategy}"

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
       --prompt-data ${data_file} \\
       --input-key prompt --label-key label --apply-chat-template --rm-type ${rm_type} \\
       --num-rollout ${num_rollout} --rollout-batch-size 1 --n-samples-per-prompt 2 \\
       --rollout-max-response-len ${max_resp_len} --rollout-temperature 1 \\
       --global-batch-size ${BATCH_SIZE} --micro-batch-size 1 \\
       --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \\
       --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \\
       --advantage-estimator grpo --eps-clip 0.2 --entropy-coef 0.01 --kl-coef ${KL_COEF} \\
       --optimizer adam --lr ${LR} --lr-decay-style constant --weight-decay 0.1 \\
       --attention-dropout 0.0 --hidden-dropout 0.0 \\
       --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \\
       --rollout-function-path slime.rollout.native_hf_rollout.generate_rollout \\
       --rollout-num-gpus-per-engine 1 \\
       ${shadow_args}
  " > "${log_file}" 2>&1

  local exit_code=$?
  local actual_rollouts=$(grep -c "'rollout/raw_reward'" "${log_file}" 2>/dev/null || echo 0)
  echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] ${name} (${actual_rollouts} rollouts, exit=${exit_code})" | tee -a "${RESULTS_DIR}/run.log"
  [ "${actual_rollouts}" -ge 10 ] && touch "${exp_dir}/DONE"
  docker exec "${CONTAINER}" bash -c "ray stop --force 2>/dev/null; sleep 3" || true
}

mkdir -p "${RESULTS_DIR}"
echo "=== Exp_01: Naive KV Collapse ($(date)) ===" | tee -a "${RESULTS_DIR}/run.log"

# NOTE: Dense baselines are reused from Exp_02/05 (SGLang rollout).
# Only the 4 naive compressed configs need this native HF engine.
run_naive_experiment "tldr_naive_snapkv50"  "snapkv" "0.5" "${DATA_PATH_TLDR}"  ${MAX_ROLLOUTS} ${MAX_RESP_LEN_TLDR}  "rouge"
run_naive_experiment "tldr_naive_random50"  "random" "0.5" "${DATA_PATH_TLDR}"  ${MAX_ROLLOUTS} ${MAX_RESP_LEN_TLDR}  "rouge"
run_naive_experiment "gsm8k_naive_snapkv50" "snapkv" "0.5" "${DATA_PATH_GSM8K}" ${MAX_ROLLOUTS} ${MAX_RESP_LEN_GSM8K} "math"
run_naive_experiment "gsm8k_naive_random50" "random" "0.5" "${DATA_PATH_GSM8K}" ${MAX_ROLLOUTS} ${MAX_RESP_LEN_GSM8K} "math"

echo "=== Exp_01 Complete! $(date) ===" | tee -a "${RESULTS_DIR}/run.log"
