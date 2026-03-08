#!/bin/bash
# run_sanity_check.sh — Task 5: Quick 50-step GRPO validation
# Verifies: (1) Baseline Dense, (2) SMD SnapKV 50% both work correctly
set -uo pipefail

MODELS_DIR="/home/zhu11/RLKV/manifold/models"
DATA_DIR="/home/zhu11/RLKV/manifold/data"
RESULTS_DIR="/home/zhu11/RLKV/RLKV_github/SMD/sanity_check_results"
CONTAINER="slime_shadow"
MODEL="1.5B"

echo "============================================"
echo "  SMD Sanity Check (50-step GRPO)"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

run_experiment() {
  local name=$1
  local extra_args=$2
  local exp_dir="${RESULTS_DIR}/${name}"
  local log_file="${exp_dir}/train.log"

  if [ -f "${exp_dir}/DONE" ]; then
    echo "[SKIP] ${name} — already done"
    return 0
  fi

  mkdir -p "${exp_dir}"
  echo "$(date '+%Y-%m-%d %H:%M:%S') [START] ${name}"

  local hf_ckpt="${MODELS_DIR}/Qwen2.5-${MODEL}-Instruct"
  local td_ckpt="${MODELS_DIR}/Qwen2.5-${MODEL}-Instruct_torch_dist"
  local data_file="${DATA_DIR}/tldr_train.jsonl"

  docker exec "${CONTAINER}" bash -c "
    set -e
    export MASTER_ADDR=127.0.0.1
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export PYTHONPATH=/root/Megatron-LM:\${PYTHONPATH:-}
    export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

    # Clean up stale ray sessions
    ray stop --force 2>/dev/null || true
    sleep 2
    rm -rf /tmp/ray/session_* 2>/dev/null || true

    # Start fresh ray head
    ray start --head --node-ip-address \${MASTER_ADDR} --num-gpus 1 --disable-usage-stats 2>&1 | tail -3
    sleep 5

    source /root/slime/scripts/models/qwen2.5-${MODEL}.sh
    MODEL_ARGS+=( --rotary-base 1000000 )

    python3 /root/slime/train.py \\
       --actor-num-nodes 1 --actor-num-gpus-per-node 1 \\
       --colocate --no-offload-train --no-offload-rollout \\
       \${MODEL_ARGS[@]} \\
       --hf-checkpoint ${hf_ckpt} --ref-load ${td_ckpt} --load ${td_ckpt} \\
       --prompt-data ${data_file} \\
       --input-key prompt --label-key label --apply-chat-template --rm-type rouge \\
       --num-rollout 50 --rollout-batch-size 1 --n-samples-per-prompt 2 \\
       --rollout-max-response-len 128 --rollout-temperature 1 \\
       --global-batch-size 2 --micro-batch-size 1 \\
       --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \\
       --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \\
       --advantage-estimator grpo --eps-clip 0.2 --entropy-coef 0.01 --kl-coef 0.5 \\
       --optimizer adam --lr 1e-7 --lr-decay-style constant --weight-decay 0.1 \\
       --attention-dropout 0.0 --hidden-dropout 0.0 \\
       --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \\
       --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.3 \\
       ${extra_args}
  " > "${log_file}" 2>&1

  local exit_code=$?
  local actual_rollouts=$(grep -c "'rollout/raw_reward'" "${log_file}" 2>/dev/null || echo 0)
  echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] ${name} (${actual_rollouts} rollouts, exit=${exit_code})"

  if [ "${actual_rollouts}" -ge 10 ]; then
    touch "${exp_dir}/DONE"
  fi
}

rm -rf "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}"

# === Experiment 1: Baseline Dense ===
run_experiment "baseline_dense" ""

# Restart container between experiments to free GPU
docker restart "${CONTAINER}" && sleep 15

# === Experiment 2: SMD SnapKV 50% ===
run_experiment "smd_snapkv50" "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv --shadow-distill-lambda 0.1"

echo ""
echo "============================================"
echo "  Validation Results"
echo "============================================"

for name in baseline_dense smd_snapkv50; do
  log="${RESULTS_DIR}/${name}/train.log"
  if [ ! -f "$log" ]; then
    echo "[FAIL] ${name}: no log file"
    continue
  fi

  rollouts=$(grep -c "'rollout/raw_reward'" "$log" 2>/dev/null || echo 0)
  first5=$(grep "'rollout/raw_reward'" "$log" | head -5 | sed "s/.*'rollout\/raw_reward': //" | sed 's/,.*//' | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}')
  last5=$(grep "'rollout/raw_reward'" "$log" | tail -5 | sed "s/.*'rollout\/raw_reward': //" | sed 's/,.*//' | awk '{sum+=$1; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}')

  echo ""
  echo "  ${name}:"
  echo "    Rollouts completed: ${rollouts}"
  echo "    First-5 avg reward: ${first5}"
  echo "    Last-5 avg reward:  ${last5}"

  errors=$(grep -ci "OutOfMemoryError\|CUDA error\|segfault" "$log" 2>/dev/null || echo 0)
  if [ "$errors" -gt 0 ]; then
    echo "    ⚠️  Critical errors found!"
  else
    echo "    ✅ No critical errors"
  fi
done

echo ""
echo "============================================"
echo "  Sanity Check Complete! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
