# Experiment Scripts

This directory contains all 7 reproducible experiments for the SMD paper.

## Experiment Map

| Directory | Purpose | Dataset | Key Variable |
|-----------|---------|---------|-------------|
| `exp_01_reward_collapse` | Naive KV collapse (motivating failure) | TL;DR + GSM8K | Compression method |
| `exp_02_sota_showdown` | SMD vs Sparse-RL vs Dense | TL;DR | Loss function |
| `exp_03_downstream_eval` | Downstream generalization | GSM8K | Training method |
| `exp_04_vram_spike_profiling` | VRAM spike measurement | TL;DR | Compression on/off |
| `exp_05_compression_ratio` | Retention ratio ablation | TL;DR | ratio ∈ {0.25, 0.5, 0.75, 1.0} |
| `exp_06_distillation_lambda` | Distillation weight ablation | TL;DR | λ ∈ {0, 0.1, 1.0} |
| `exp_07_kv_strategy` | Token selection strategy | TL;DR | strategy ∈ {snapkv, random, recent} |

## For Collaborators: How to Scale Up

Each `run_exp.sh` has a clearly marked **Configuration** section at the top:

```bash
# ── Configuration (Modify here for scale-up) ───────────────────────────
MODEL_SIZE="1.5B"        # ← Change to "7B" or "14B"
MODEL_PATH="..."         # ← Point to your model weights
DATA_PATH="..."          # ← Point to your dataset
CONTAINER="slime_shadow" # ← Your Docker container name
MAX_ROLLOUTS=500         # ← Increase for longer training
BATCH_SIZE=2             # ← Increase with more GPUs
```

### To run on a 7B model:
1. Download `Qwen2.5-7B-Instruct` weights
2. Convert to Megatron format with `tools/convert_hf_to_torch_dist.py`
3. Edit `MODEL_SIZE="7B"` and update `MODEL_PATH`
4. Increase `BATCH_SIZE` if you have multiple GPUs

### Multi-GPU scaling:
For TP=2 or TP=4, also update `--tensor-model-parallel-size` in `_shared/run_helper.sh`.

## Running Experiments

```bash
# Run a single experiment
bash experiments/exp_02_sota_showdown/run_exp.sh

# Run all experiments sequentially
for exp in exp_0{1..7}_*; do
  bash experiments/$exp/run_exp.sh
done
```

## Dependencies
- Docker container with Slime + Megatron-LM installed
- NVIDIA GPU with ≥ 40GB VRAM (for 1.5B model)
- Model weights in `shared_resources/models/`
- Datasets in `shared_resources/datasets/`
