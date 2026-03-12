# RLKV: Reinforcement Learning with KV Cache Compression

A research monorepo for developing and evaluating **KV cache compression methods** integrated with **Reinforcement Learning post-training** (GRPO/PPO).

## Motivation

Large Language Models require enormous GPU memory for their KV caches during inference. While compression methods like SnapKV can reduce this footprint, naively applying them during RL training causes **catastrophic reward collapse** due to off-policy mismatch. This repository hosts **Shadow Mask Distillation (SMD)**, a dual-track training algorithm that eliminates this mismatch.

## Projects

| Project | Status | Description |
|---------|--------|-------------|
| **[SMD](./SMD/)** | ✅ Active | Shadow Mask Distillation — Dual-track GRPO loss for KV-compressed RL |
| **[Baselines](./baselines/)** | ✅ Active | 4 SOTA comparison methods (Sparse-RL, QuRL, RLHFless, R-KV) |

## Roadmap

- 🔜 **Multi-GPU branch** — A new branch for multi-GPU distributed training is coming soon, enabling scaling experiments on larger models (7B+) with tensor/pipeline parallelism.

## Supported Models & Datasets

### Models
| Model | Parameters | Architecture | Status |
|-------|-----------|--------------|--------|
| Qwen3-0.6B | 0.6B | Transformer | ✅ Validated |
| Qwen2.5-1.5B-Instruct | 1.5B | Transformer | ✅ Validated |
| Qwen3-1.7B | 1.7B | Transformer | ✅ Validated |
| Qwen3-4B-Instruct-2507 | 4B | Transformer | ✅ Validated |

### Datasets
| Dataset | Task | Avg Context | Reward Type |
|---------|------|-------------|-------------|
| TL;DR (Reddit) | Summarization | ~300 tokens | ROUGE-L |
| GSM8K | Math Reasoning | ~500 tokens | Exact Match |
| GovReport | Long-Doc Summarization | ~9,000 tokens | ROUGE-L |
| HotpotQA | Multi-hop QA + Distractors | ~800 tokens | Fuzzy Match |

## Repository Structure

```
RLKV_github/
├── shared_resources/          # Local models & datasets (git-ignored, download via prep scripts)
│   ├── models/                # HuggingFace model weights + torch_dist checkpoints
│   └── datasets/              # Training/eval data (JSONL format)
│       ├── tldr/              #   Reddit TL;DR summarization
│       ├── gsm8k/             #   Grade school math
│       ├── gov_report/        #   Government report summarization
│       └── hotpot_qa/         #   Multi-hop QA with distractors
├── baselines/                 # SOTA comparison methods
│   ├── sparse_rl.py           #   Rejection sampling + importance reweighting
│   ├── qurl.py                #   Update-Aware Quantization + adaptive PPO clipping
│   ├── rlhfless.py            #   Serverless elastic scheduling
│   └── r_kv.py                #   Redundancy-aware KV eviction
└── SMD/                       # Shadow Mask Distillation (our method)
    ├── src/                   # Core Python modules (shadow mask, losses, rewards)
    │   └── rewards/           # Unified reward functions for all datasets
    ├── data_prep/             # Dataset download & formatting scripts
    ├── experiments/           # Parameterized experiment scripts (Exp_01-07)
    ├── figures/               # Publication-ready plots
    └── EXPERIMENT_RESULTS.md  # Full results with cross-model scaling analysis
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/huthvincent/RL-Post-Training-with-KV-Cache-Compression.git
cd RL-Post-Training-with-KV-Cache-Compression

# 2. Download datasets
pip install datasets rouge-score
python SMD/data_prep/prep_tldr.py
python SMD/data_prep/prep_gsm8k.py
python SMD/data_prep/prep_govreport.py
python SMD/data_prep/prep_hotpotqa.py

# 3. See SMD project for experiment instructions
```

See the [SMD README](./SMD/README.md) for detailed setup, Docker environment, and experiment instructions.

## Citation

```bibtex
@article{smd2026,
  title={Shadow Mask Distillation: Breaking the Memory Wall for RL Post-Training with KV Cache Compression},
  author={Zhu, Rui},
  year={2026}
}
```

## License

Apache 2.0

