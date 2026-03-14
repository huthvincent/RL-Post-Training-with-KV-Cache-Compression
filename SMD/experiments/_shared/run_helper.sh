#!/bin/bash
# _shared/run_helper.sh — Shared SGLang-based training function
# Source this file from experiment scripts: source "../_shared/run_helper.sh"
#
# Required variables (set by the calling script):
#   MODEL_SIZE, MODEL_PATH, CONTAINER, BATCH_SIZE, LR, KL_COEF,
#   MAX_RESP_LEN, RESULTS_DIR

run_sglang_experiment() {
  local name=$1 extra_args=$2 data_file=$3 rm_type=$4 num_rollout=$5
  local exp_dir="${RESULTS_DIR}/${name}"
  local log_file="${exp_dir}/train.log"

  if [ -f "${exp_dir}/DONE" ]; then
    echo "[SKIP] ${name} — already done"
    return 0
  fi

  mkdir -p "${exp_dir}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${name}" | tee -a "${RESULTS_DIR}/run.log"

  local hf_ckpt="${MODEL_PATH}/Qwen3-${MODEL_SIZE}"
  local td_ckpt="${MODEL_PATH}/Qwen3-${MODEL_SIZE}_torch_dist"

  docker exec "${CONTAINER}" bash -c "
    set -e
    export MASTER_ADDR=127.0.0.1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONPATH=/root/Megatron-LM:\${PYTHONPATH:-}
    export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
    export RAY_ADDRESS=auto

    ray stop --force 2>/dev/null || true
    sleep 3
    rm -rf /tmp/ray/session_* 2>/dev/null || true

    ray start --head --num-gpus 1 --disable-usage-stats 2>&1 | tail -5
    sleep 8

    source /root/slime/scripts/models/qwen3-${MODEL_SIZE}.sh
    MODEL_ARGS+=(--rotary-base 1000000)

    python3 /root/slime/train.py \\
       --actor-num-nodes 1 --actor-num-gpus-per-node 1 \\
       --colocate --no-offload-train --no-offload-rollout \\
       \${MODEL_ARGS[@]} \\
       --hf-checkpoint ${hf_ckpt} --ref-load ${td_ckpt} --load ${td_ckpt} \\
       --prompt-data ${data_file} \\
       --input-key prompt --label-key label --apply-chat-template --rm-type ${rm_type} \\
       --num-rollout ${num_rollout} --rollout-batch-size 1 --n-samples-per-prompt 2 \\
       --rollout-max-response-len ${MAX_RESP_LEN} --rollout-temperature 1 \\
       --global-batch-size ${BATCH_SIZE} --micro-batch-size 1 \\
       --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \\
       --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \\
       --advantage-estimator grpo --eps-clip 0.2 --entropy-coef 0.01 --kl-coef ${KL_COEF} \\
       --optimizer adam --lr ${LR} --lr-decay-style constant --weight-decay 0.1 \\
       --attention-dropout 0.0 --hidden-dropout 0.0 \\
       --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \\
       --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.3 \\
       ${extra_args}
  " > "${log_file}" 2>&1

  local exit_code=$?
  local actual_rollouts=$(grep -c "'rollout/raw_reward'" "${log_file}" 2>/dev/null || echo 0)
  echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] ${name} (${actual_rollouts} rollouts, exit=${exit_code})" | tee -a "${RESULTS_DIR}/run.log"
  [ "${actual_rollouts}" -ge 5 ] && touch "${exp_dir}/DONE"
  docker exec "${CONTAINER}" bash -c "ray stop --force 2>/dev/null; sleep 2" || true
}
