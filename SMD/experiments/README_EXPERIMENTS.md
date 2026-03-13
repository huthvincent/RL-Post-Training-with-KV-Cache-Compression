# SMD Experiment Matrix

> **Model**: Qwen3-1.7B  |  **Datasets**: TL;DR (ROUGE-L), GSM8K (Exact Match)

---

## Overview

| #  | Experiment | Question | Key Variables |
|----|-----------|----------|---------------|
| 1  | **Naive KV Collapse** | Why does naive KV compression break RL? | Dense vs 4× naive KV strategies |
| 2  | **SOTA Showdown** | Is SMD better than other corrections? | SMD vs Sparse-RL vs RLHFless vs Dense |
| 3  | **Downstream Generalization** | Does SMD preserve downstream accuracy? | GSM8K train → test eval |
| 5  | **Compression Ratio** | How much can we compress? | R-KV retention = {25%, 50%, 75%, 100%} |
| 6  | **Distillation Lambda** | How much KL distillation? | λ = {0, 0.1, 0.5, 1.0} |
| 7  | **KV Strategy** | Which compression strategy is best? | R-KV vs SnapKV vs Random vs Recent |
| 8  | **Quant vs KV Compression** | Which mismatch is more severe? | QuRL (INT8) vs SMD (R-KV 50%) |
| 9  | **PPO vs GRPO for SMD** | Does advantage estimation method matter? | PPO (critic) vs GRPO (group-relative) |
| 10 | **Stability** | Is training reproducible? | 5 seeds × {Dense, SMD}, mean ± std |

> **Note**: Exp 4 was removed (former VRAM spike profiling, not relevant to final paper).

---

## Directory Structure

```
experiments/
├── _shared/                    # Shared helper scripts
│   └── run_helper.sh
├── exp_01_reward_collapse/     # Exp 1: Naive KV → collapse
│   ├── run_exp.sh              # Slime/Megatron version
│   ├── run_exp01.py            # Standalone HuggingFace version
│   └── run_exp01.sh
├── exp_02_sota_showdown/       # Exp 2: SMD vs competitors
│   └── run_exp.sh
├── exp_03_downstream_eval/     # Exp 3: GSM8K generalization
│   └── run_exp.sh
├── exp_05_compression_ratio/   # Exp 5: retention ratio sweep
│   └── run_exp.sh
├── exp_06_distillation_lambda/ # Exp 6: λ sweep
│   └── run_exp.sh
├── exp_07_kv_strategy/         # Exp 7: compression strategy
│   └── run_exp.sh
├── exp_08_quant_vs_kv/         # Exp 8: quantization vs KV
│   └── run_exp.sh
├── exp_09_ppo_vs_grpo/         # Exp 9: advantage method
│   └── run_exp.sh
├── exp_10_stability/           # Exp 10: multi-seed stability
│   ├── run_stability.py        # Python: 5 seeds + report
│   └── run_exp.sh
└── qwen3_0.6b/                 # Standalone 0.6B experiments
    ├── run_grpo_training.py
    └── run_all_experiments.sh
```

---

## How to Run

### Slime/Megatron (inside Docker `slime_shadow`)

```bash
docker exec slime_shadow bash -c "
  cd /home/zhu11/RLKV/RLKV_github/SMD/experiments/exp_01_reward_collapse
  bash run_exp.sh
"
```

### Standalone HuggingFace (Exp 1, 10)

```bash
docker exec slime_shadow bash -c "
  export RLKV_ROOT=/home/zhu11/RLKV/RLKV_github
  export PYTHONPATH=\$RLKV_ROOT:\$PYTHONPATH
  python \$RLKV_ROOT/SMD/experiments/exp_10_stability/run_stability.py \
    --output-dir /workspace/results/exp10 --method smd
"
```

---

## Exp 10: Stability Test Details

Runs the same configuration with **5 random seeds** (42, 123, 456, 789, 2026) and reports:

| Metric | Formula |
|--------|---------|
| Reward (last 50) | `mean(avg_reward[-50:])` across seeds |
| Std | `std(avg_reward[-50:])` across seeds |
| Range | `[min, max]` across seeds |

**Output**: `stability_report.json` with per-seed and aggregated statistics.

Supports `--report-only` to regenerate report from existing results.
