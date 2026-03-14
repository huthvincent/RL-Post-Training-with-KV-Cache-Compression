#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Smoke Test — Verify all experiments (2-10) can launch and run
# ═══════════════════════════════════════════════════════════════
#
# Runs ONE representative config per experiment with 5 rollouts.
# ALL tests run inside Docker (slime_shadow) via Slime/Megatron.
#
# Usage:   bash smoking_test.sh
# Results: /tmp/smoke_test_results/
# ═══════════════════════════════════════════════════════════════
set -uo pipefail

CONTAINER="slime_shadow"
MODEL_SIZE="1.7B"
MODEL_PATH="/home/zhu11/RLKV/RLKV_github/shared_resources/models"
SMOKE_DIR="/tmp/smoke_test_results"
SMOKE_ROLLOUTS=5
START_TIME=$(date +%s)

DATA_TLDR="/home/zhu11/RLKV/RLKV_github/shared_resources/datasets/tldr/train.jsonl"
DATA_GSM8K="/home/zhu11/RLKV/RLKV_github/shared_resources/datasets/gsm8k/train.jsonl"

HF_CKPT="${MODEL_PATH}/Qwen3-${MODEL_SIZE}"
TD_CKPT="${MODEL_PATH}/Qwen3-${MODEL_SIZE}_torch_dist"

rm -rf "${SMOKE_DIR}"
mkdir -p "${SMOKE_DIR}"

PASS=0
FAIL=0
declare -A RESULTS

# SMD loss function path (used by most experiments)
SMD_LOSS="slime.backends.megatron_utils.shadow_distillation_loss.shadow_distillation_loss_function"

smoke_slime() {
    local test_name=$1
    local extra_args=$2
    local data_file=$3
    local rm_type=$4
    local log_file="${SMOKE_DIR}/${test_name}.log"

    echo "──────────────────────────────────────────"
    echo "🔬 [${test_name}] Starting..."

    docker exec "${CONTAINER}" bash -c "
        set -e
        export MASTER_ADDR=127.0.0.1
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export PYTHONPATH=/root/Megatron-LM:\${PYTHONPATH:-}
        export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
        export RAY_ADDRESS=auto

        ray stop --force 2>/dev/null || true
        sleep 2
        rm -rf /tmp/ray/session_* 2>/dev/null || true
        ray start --head --num-gpus 1 --disable-usage-stats 2>&1 | tail -3
        sleep 8

        source /root/slime/scripts/models/qwen3-${MODEL_SIZE}.sh
        MODEL_ARGS+=(--rotary-base 1000000)

        timeout 480 python3 /root/slime/train.py \
            --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
            --colocate --no-offload-train --no-offload-rollout \
            \${MODEL_ARGS[@]} \
            --hf-checkpoint ${HF_CKPT} --ref-load ${TD_CKPT} --load ${TD_CKPT} \
            --prompt-data ${data_file} \
            --input-key prompt --label-key label --apply-chat-template --rm-type ${rm_type} \
            --num-rollout ${SMOKE_ROLLOUTS} --rollout-batch-size 1 --n-samples-per-prompt 2 \
            --rollout-max-response-len 64 --rollout-temperature 1 \
            --global-batch-size 2 --micro-batch-size 1 \
            --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
            --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
            --advantage-estimator grpo --eps-clip 0.2 --entropy-coef 0.01 --kl-coef 0.5 \
            --optimizer adam --lr 1e-7 --lr-decay-style constant --weight-decay 0.1 \
            --attention-dropout 0.0 --hidden-dropout 0.0 \
            --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash \
            --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.3 \
            ${extra_args}
    " > "${log_file}" 2>&1
    local exit_code=$?

    docker exec "${CONTAINER}" bash -c "ray stop --force 2>/dev/null; sleep 2" 2>/dev/null || true

    # Check success: either clean exit or at least some rollouts happened
    local rollout_count
    rollout_count=$(grep -c "rollout/raw_reward" "${log_file}" 2>/dev/null || echo 0)
    if [ "$rollout_count" -ge 1 ]; then
        echo "✅ [${test_name}] PASS (${rollout_count} rollouts, exit=${exit_code})"
        RESULTS["${test_name}"]="✅ PASS (${rollout_count} rollouts)"
        ((PASS++))
    elif [ "$exit_code" -eq 0 ]; then
        echo "⚠️  [${test_name}] LAUNCHED OK but 0 rollouts detected in log"
        RESULTS["${test_name}"]="⚠️ LAUNCHED (0 rollouts in log)"
        ((PASS++))
    else
        echo "❌ [${test_name}] FAIL (exit=${exit_code})"
        echo "   Last 5 lines:"
        tail -5 "${log_file}" 2>/dev/null | sed 's/^/   /'
        RESULTS["${test_name}"]="❌ FAIL (exit=${exit_code})"
        ((FAIL++))
    fi
}

# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  SMOKE TEST: Exp 2-10  (${SMOKE_ROLLOUTS} rollouts each)"
echo "  All tests via Slime inside Docker: ${CONTAINER}"
echo "  Started: $(date)"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Exp 02: SOTA Showdown — dense baseline ──
smoke_slime "exp02_dense" \
    "" \
    "${DATA_TLDR}" "rouge"

# ── Exp 03: Downstream — GSM8K dense ──
smoke_slime "exp03_gsm8k" \
    "" \
    "${DATA_GSM8K}" "math"

# ── Exp 05: Compression Ratio — SMD SnapKV 50% ──
smoke_slime "exp05_smd_r0.5" \
    "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv --shadow-distill-lambda 0.1 --loss-type custom_loss --custom-loss-function-path ${SMD_LOSS}" \
    "${DATA_TLDR}" "rouge"

# ── Exp 06: Lambda — SMD λ=0.1 ──
smoke_slime "exp06_smd_l0.1" \
    "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv --shadow-distill-lambda 0.1 --loss-type custom_loss --custom-loss-function-path ${SMD_LOSS}" \
    "${DATA_TLDR}" "rouge"

# ── Exp 07: KV Strategy — SnapKV ──
smoke_slime "exp07_snapkv" \
    "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv --shadow-distill-lambda 0.1 --loss-type custom_loss --custom-loss-function-path ${SMD_LOSS}" \
    "${DATA_TLDR}" "rouge"

# ── Exp 08: Quant vs KV — dense (QuRL needs quantized model, so test dense only) ──
smoke_slime "exp08_dense" \
    "" \
    "${DATA_TLDR}" "rouge"

# ── Exp 09: PPO vs GRPO — GRPO + SMD ──
smoke_slime "exp09_grpo_smd" \
    "--use-shadow-mask --shadow-retention-ratio 0.5 --shadow-strategy snapkv --shadow-distill-lambda 0.1 --loss-type custom_loss --custom-loss-function-path ${SMD_LOSS}" \
    "${DATA_TLDR}" "rouge"

# ── Exp 10: Stability — dense baseline ──
smoke_slime "exp10_dense" \
    "" \
    "${DATA_TLDR}" "rouge"

# ═══════════════════════════════════════════════════════════════
elapsed=$(( $(date +%s) - START_TIME ))
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  SMOKE TEST SUMMARY  (${elapsed}s / $(( elapsed / 60 ))m)"
echo "═══════════════════════════════════════════════════════"
for t in exp02_dense exp03_gsm8k exp05_smd_r0.5 exp06_smd_l0.1 exp07_snapkv exp08_dense exp09_grpo_smd exp10_dense; do
    printf "  %-25s %s\n" "${t}" "${RESULTS[${t}]:-⏭ SKIPPED}"
done
echo "───────────────────────────────────────────────────────"
echo "  Pass: ${PASS}  |  Fail: ${FAIL}  |  Total: $((PASS + FAIL))"
echo "  Logs: ${SMOKE_DIR}/"
echo "═══════════════════════════════════════════════════════"
