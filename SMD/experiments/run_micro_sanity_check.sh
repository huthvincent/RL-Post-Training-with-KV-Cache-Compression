#!/bin/bash
# run_micro_sanity_check.sh — 10-rollout validation for ALL experiment types
# Tests path resolution, reward computation, and gradient updates
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="slime_shadow"
LOG="${SCRIPT_DIR}/micro_check.log"

echo "============================================" | tee "${LOG}"
echo "  Micro Sanity Check — ALL Experiment Types" | tee -a "${LOG}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"             | tee -a "${LOG}"
echo "============================================" | tee -a "${LOG}"

# ── Docker-visible paths (adjust for your environment) ──────────────────
# The Docker container has /home/zhu11/RLKV/manifold mounted.
# If your model/data paths differ inside the container, change these:
MODEL_SIZE="1.5B"
MODEL_PATH="/home/zhu11/RLKV/manifold/models"
TLDR_DATA="/home/zhu11/RLKV/manifold/data/tldr_train.jsonl"
GSM8K_DATA="/home/zhu11/RLKV/manifold/data/gsm8k_fewshot_train.jsonl"
# ── End paths ───────────────────────────────────────────────────────────

MICRO_RESULTS="${SCRIPT_DIR}/micro_results"
rm -rf "${MICRO_RESULTS}"
mkdir -p "${MICRO_RESULTS}"

PASS_COUNT=0
FAIL_COUNT=0

run_micro() {
  local name=$1 data=$2 rm_type=$3 rollout_fn=$4 extra_args=$5
  local exp_dir="${MICRO_RESULTS}/${name}"
  local log_file="${exp_dir}/train.log"
  mkdir -p "${exp_dir}"

  echo "" | tee -a "${LOG}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${name}" | tee -a "${LOG}"

  local hf_ckpt="${MODEL_PATH}/Qwen2.5-${MODEL_SIZE}-Instruct"
  local td_ckpt="${MODEL_PATH}/Qwen2.5-${MODEL_SIZE}-Instruct_torch_dist"

  local rollout_args=""
  if [ "${rollout_fn}" = "native" ]; then
    rollout_args="--rollout-function-path slime.rollout.native_hf_rollout.generate_rollout --rollout-num-gpus-per-engine 1"
  else
    rollout_args="--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.3"
  fi

  docker exec "${CONTAINER}" bash -c "
    set -e
    export MASTER_ADDR=127.0.0.1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONPATH=/root/Megatron-LM:\${PYTHONPATH:-}
    export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

    ray stop --force 2>/dev/null || true
    sleep 3
    rm -rf /tmp/ray/session_* 2>/dev/null || true

    ray start --head --num-gpus 1 --disable-usage-stats 2>&1 | tail -5
    sleep 8

    source /root/slime/scripts/models/qwen2.5-${MODEL_SIZE}.sh
    MODEL_ARGS+=(--rotary-base 1000000)

    python3 /root/slime/train.py \\
       --actor-num-nodes 1 --actor-num-gpus-per-node 1 \\
       --colocate --no-offload-train --no-offload-rollout \\
       \${MODEL_ARGS[@]} \\
       --hf-checkpoint ${hf_ckpt} --ref-load ${td_ckpt} --load ${td_ckpt} \\
       --prompt-data ${data} \\
       --input-key prompt --label-key label --apply-chat-template --rm-type ${rm_type} \\
       --num-rollout 10 --rollout-batch-size 1 --n-samples-per-prompt 2 \\
       --rollout-max-response-len 128 --rollout-temperature 1 \\
       --global-batch-size 2 --micro-batch-size 1 \\
       --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \\
       --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \\
       --advantage-estimator grpo --eps-clip 0.2 --entropy-coef 0.01 --kl-coef 0.5 \\
       --optimizer adam --lr 1e-7 --lr-decay-style constant --weight-decay 0.1 \\
       --attention-dropout 0.0 --hidden-dropout 0.0 \\
       --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \\
       ${rollout_args} \\
       ${extra_args}
  " > "${log_file}" 2>&1

  local exit_code=$?
  local n_rollouts
  n_rollouts=$(grep -c "'rollout/raw_reward'" "${log_file}" 2>/dev/null) || n_rollouts=0
  local has_grad
  has_grad=$(grep -c "grad_norm" "${log_file}" 2>/dev/null) || has_grad=0
  local has_oom
  has_oom=$(grep -ci "OutOfMemoryError" "${log_file}" 2>/dev/null) || has_oom=0

  if [ "${n_rollouts}" -ge 3 ] && [ "${has_oom}" -eq 0 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [PASS ✅] ${name} (${n_rollouts} rollouts, ${has_grad} grads, exit=${exit_code})" | tee -a "${LOG}"
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [FAIL ❌] ${name} (${n_rollouts} rollouts, OOM=${has_oom}, exit=${exit_code})" | tee -a "${LOG}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi

  docker exec "${CONTAINER}" bash -c "ray stop --force 2>/dev/null; sleep 2" || true
}

# ── Test 1: Dense baseline (SGLang, standard loss, TL;DR/ROUGE) ────────
run_micro "sglang_dense_tldr" "${TLDR_DATA}" "rouge" "sglang" ""

# ── Test 2: SMD SnapKV 50% (SGLang, shadow_distillation_loss) ───────────
run_micro "sglang_smd_snapkv50" "${TLDR_DATA}" "rouge" "sglang" \
  "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv --shadow-distill-lambda 0.1 --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.shadow_distillation_loss.shadow_distillation_loss_function"

# ── Test 3: Sparse-RL (SGLang, sparse_rl_loss) ─────────────────────────
run_micro "sglang_sparse_rl" "${TLDR_DATA}" "rouge" "sglang" \
  "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv --loss-type custom_loss --custom-loss-function-path slime.backends.megatron_utils.sparse_rl_loss.sparse_rl_loss_function"

# ── Test 4: Dense baseline (SGLang, math reward, GSM8K) ────────────────
run_micro "sglang_dense_gsm8k" "${GSM8K_DATA}" "math" "sglang" ""

# Restart container for native HF tests (needs clean GPU)
docker restart "${CONTAINER}" && sleep 15

# ── Test 5: Native HF SnapKV 50% (native rollout, true KV eviction) ───
run_micro "native_snapkv50_tldr" "${TLDR_DATA}" "rouge" "native" \
  "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv"

echo "" | tee -a "${LOG}"
echo "============================================"                     | tee -a "${LOG}"
echo "  Micro Sanity Check Complete! $(date '+%Y-%m-%d %H:%M:%S')"     | tee -a "${LOG}"
echo "  Results: ${PASS_COUNT} PASS, ${FAIL_COUNT} FAIL"               | tee -a "${LOG}"
echo "============================================"                     | tee -a "${LOG}"
