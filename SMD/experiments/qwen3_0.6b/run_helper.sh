#!/bin/bash
# ==========================================
# Shared helper for Qwen3-0.6B experiments
# ==========================================
# Runs training inside Docker container smd_training using veRL + Megatron

CONTAINER="smd_training"
MODEL_NAME="Qwen3-0.6B"
HF_CKPT="/workspace/RLKV/shared_resources/models/Qwen3-0.6B"
TD_CKPT="/workspace/RLKV/shared_resources/models/Qwen3-0.6B_torch_dist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset paths inside container
TLDR_DATA="/workspace/RLKV/shared_resources/datasets/tldr/train.jsonl"
GSM8K_DATA="/workspace/RLKV/shared_resources/datasets/gsm8k/train.jsonl"
GOVREPORT_DATA="/workspace/RLKV/shared_resources/datasets/gov_report/train.jsonl"
HOTPOTQA_DATA="/workspace/RLKV/shared_resources/datasets/hotpot_qa/train.jsonl"

# Default training hyperparams
MAX_ROLLOUTS=${MAX_ROLLOUTS:-500}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-"1e-7"}
KL_COEF=${KL_COEF:-"0.5"}

get_data_path() {
  local dataset=$1
  case $dataset in
    tldr)      echo "$TLDR_DATA" ;;
    gsm8k)     echo "$GSM8K_DATA" ;;
    govreport) echo "$GOVREPORT_DATA" ;;
    hotpotqa)  echo "$HOTPOTQA_DATA" ;;
    *) echo "ERROR: unknown dataset $dataset"; exit 1 ;;
  esac
}

get_rm_type() {
  local dataset=$1
  case $dataset in
    tldr)      echo "rouge" ;;
    gsm8k)     echo "math" ;;
    govreport) echo "govreport" ;;
    hotpotqa)  echo "hotpotqa" ;;
    *) echo "ERROR: unknown dataset $dataset"; exit 1 ;;
  esac
}

get_max_resp_len() {
  local dataset=$1
  case $dataset in
    tldr)      echo 128 ;;
    gsm8k)     echo 256 ;;
    govreport) echo 512 ;;
    hotpotqa)  echo 256 ;;
    *) echo 256 ;;
  esac
}

run_experiment() {
  local name=$1 extra_args=$2 dataset=$3 num_rollout=$4
  local data_file=$(get_data_path "$dataset")
  local rm_type=$(get_rm_type "$dataset")
  local max_resp_len=$(get_max_resp_len "$dataset")
  local exp_dir="${RESULTS_DIR}/${name}"
  local log_file="${exp_dir}/train.log"

  if [ -f "${exp_dir}/DONE" ]; then
    echo "[SKIP] ${name} — already done"
    return 0
  fi

  mkdir -p "${exp_dir}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${name}" | tee -a "${RESULTS_DIR}/run.log"

  docker exec "${CONTAINER}" bash -c "
    set -e
    export MASTER_ADDR=127.0.0.1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONPATH=/workspace/Megatron-LM:/workspace/RLKV:\${PYTHONPATH:-}
    export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

    ray stop --force 2>/dev/null || true
    sleep 3
    rm -rf /tmp/ray/session_* 2>/dev/null || true

    ray start --head --num-gpus 1 --disable-usage-stats 2>&1 | tail -5
    sleep 8

    source /workspace/RLKV/SMD/experiments/qwen3_0.6b/model_config.sh

    python3 -m verl.trainer.main_ppo \\
       algorithm.adv_estimator=grpo \\
       data.train_files=${data_file} \\
       data.train_batch_size=${BATCH_SIZE} \\
       data.max_prompt_length=2048 \\
       data.max_response_length=${max_resp_len} \\
       actor_rollout_ref.model.path=${HF_CKPT} \\
       actor_rollout_ref.model.enable_gradient_checkpointing=True \\
       actor_rollout_ref.actor.optim.lr=${LR} \\
       actor_rollout_ref.actor.ppo_mini_batch_size=${BATCH_SIZE} \\
       actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
       actor_rollout_ref.actor.use_kl_loss=True \\
       actor_rollout_ref.actor.kl_loss_coef=${KL_COEF} \\
       actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
       actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \\
       actor_rollout_ref.rollout.n=2 \\
       actor_rollout_ref.rollout.temperature=1.0 \\
       actor_rollout_ref.rollout.max_num_seqs=512 \\
       actor_rollout_ref.rollout.max_model_len=4096 \\
       trainer.n_gpus_per_node=1 \\
       trainer.nnodes=1 \\
       trainer.total_epochs=${num_rollout} \\
       trainer.logger='[\"console\"]' \\
       ${extra_args}
  " > "${log_file}" 2>&1

  local exit_code=$?
  local actual_rollouts=$(grep -c "'rollout/raw_reward'" "${log_file}" 2>/dev/null || echo 0)
  echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] ${name} (${actual_rollouts} rollouts, exit=${exit_code})" | tee -a "${RESULTS_DIR}/run.log"
  [ "${actual_rollouts}" -ge 5 ] && touch "${exp_dir}/DONE"
  docker exec "${CONTAINER}" bash -c "ray stop --force 2>/dev/null; sleep 2" || true
}
