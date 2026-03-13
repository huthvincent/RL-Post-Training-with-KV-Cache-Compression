# Server Sync Guide: 完整项目部署指南

> **源服务器**: `sev-cxl` (`zhu11@sev-cxl`)
> **目标**: 在另一台 server 上完全复制本项目的运行环境

---

## 架构概览

```
/home/zhu11/RLKV/
├── RLKV_github/          ← GitHub 仓库 (代码、实验脚本、模型)
│   ├── SMD/src/           ← 核心算法 (Shadow Mask Distillation)
│   ├── baselines/         ← Baseline 实验
│   ├── experiments/       ← 实验配置
│   └── shared_resources/
│       └── models/        ← HuggingFace模型权重 (~22GB, 不在 git 中)
│
└── manifold/              ← Docker 挂载目录
    ├── slime/             ← 定制版 Slime 源码 (覆盖 Docker 内的 /root/slime)
    ├── Megatron-LM/       ← Megatron-LM 源码
    ├── data/              ← 训练数据
    └── models/            ← Megatron 格式模型
```

**Docker 容器 `slime_shadow`**:
- 镜像: `slimerl/slime:latest`
- 3 个 Volume Mounts:
  - `manifold/ → /home/zhu11/RLKV/manifold` (数据+模型)
  - `manifold/slime/ → /root/slime` (⚠️ 覆盖容器内 Slime)
  - `RLKV_github/ → /home/zhu11/RLKV/RLKV_github` (代码)
- GPU: 全部 (`--gpus all`)
- SHM: 64GB (`--shm-size=64g`)

---

## Step-by-Step 部署

### Step 1: 创建目录结构

```bash
mkdir -p /home/zhu11/RLKV/manifold
mkdir -p /home/zhu11/RLKV/RLKV_github
```

### Step 2: 克隆 GitHub 仓库

```bash
cd /home/zhu11/RLKV
git clone https://github.com/huthvincent/KV-Cache-Compression-in-Reinforcement-Learning.git RLKV_github
```

### Step 3: 从源服务器 SCP 关键文件

以下文件**不在 GitHub 中**，必须从 `sev-cxl` 拷贝：

```bash
# ⚠️ 关键: 定制版 Slime 源码 (~12MB)
# 这是我们修改过的 Slime，包含 shadow mask 参数和 attention hooks
scp -r zhu11@sev-cxl:/home/zhu11/RLKV/manifold/slime /home/zhu11/RLKV/manifold/slime

# Megatron-LM (~108MB)
scp -r zhu11@sev-cxl:/home/zhu11/RLKV/manifold/Megatron-LM /home/zhu11/RLKV/manifold/Megatron-LM

# HuggingFace 模型权重 (~22GB, 需要时间)
scp -r zhu11@sev-cxl:/home/zhu11/RLKV/RLKV_github/shared_resources/models/ /home/zhu11/RLKV/RLKV_github/shared_resources/models/
```

**SCP 的具体模型列表** (按需选择):
| 模型 | 大小 | 路径 |
|------|------|------|
| Qwen3-1.7B | 3.8GB | `shared_resources/models/Qwen3-1.7B/` |
| Qwen3-1.7B_torch_dist | 3.3GB | `shared_resources/models/Qwen3-1.7B_torch_dist/` |
| Qwen3-4B-Instruct-2507 | 7.6GB | `shared_resources/models/Qwen3-4B-Instruct-2507/` |
| Qwen3-4B-Instruct-2507_torch_dist | 7.5GB | `shared_resources/models/Qwen3-4B-Instruct-2507_torch_dist/` |

### Step 4: 拉取 Docker 镜像并创建容器

```bash
docker pull slimerl/slime:latest

docker run -d \
  --gpus all \
  --shm-size=64g \
  --name slime_shadow \
  -v /home/zhu11/RLKV/manifold:/home/zhu11/RLKV/manifold \
  -v /home/zhu11/RLKV/manifold/slime:/root/slime \
  -v /home/zhu11/RLKV/RLKV_github:/home/zhu11/RLKV/RLKV_github \
  slimerl/slime:latest \
  sleep infinity
```

> **⚠️ 三个 -v 挂载全部必须有**，特别是 `-v manifold/slime:/root/slime` 会覆盖容器内原始 Slime，这是我们自定义 shadow mask 参数和 attention hooks 的关键。

### Step 5: 验证环境

```bash
# 检查 GPU 可用
docker exec slime_shadow nvidia-smi

# 检查 Slime 自定义修改是否生效
docker exec slime_shadow bash -c "grep -c 'shadow_mask' /root/slime/slime/rollout/shadow_mask_interceptor.py"
# 期望输出: 大于 0

# 检查关键 Python 包
docker exec slime_shadow bash -c "
  python -c 'import slime; print(\"Slime OK\")' &&
  python -c 'import megatron; print(\"Megatron OK\")' &&
  python -c 'import sglang; print(f\"SGLang {sglang.__version__}\")' &&
  echo 'All OK'
"

# 检查项目代码
docker exec slime_shadow bash -c "
  PYTHONPATH=/home/zhu11/RLKV/RLKV_github:/root/Megatron-LM:/root/slime \
  python -c '
from SMD.src.shadow_mask_interceptor import ShadowMaskConfig
from SMD.src.attention_extraction import register_attention_hooks
print(\"SMD imports OK\")
'
"
```

---

## ⚠️ 关键注意事项

### 1. Slime 不是公开包
Slime 是 veRL 的字节跳动内部 fork，**不是公开的 pip 包**。它只存在于 Docker 镜像 `slimerl/slime:latest` 中。我们的定制版在 `manifold/slime/` 目录，通过 `-v` 挂载覆盖容器内的 `/root/slime/`。

### 2. 定制版 Slime 的修改内容
我们对 Slime 做了以下修改（相对于 Docker 镜像内的原版）：

| 文件 | 修改 |
|------|------|
| `slime/rollout/shadow_mask_interceptor.py` | Shadow Mask 生成器（snapkv/r_kv/random/recent 4 种策略） |
| `slime/backends/megatron_utils/actor.py` | 注册 attention extraction hooks |
| `slime/backends/megatron_utils/shadow_distillation_loss.py` | 双轨 loss (Shadow PG + KL Distillation) |
| `slime/backends/megatron_utils/arguments.py` | 注册 shadow mask 命令行参数 |

### 3. 实验运行方式
所有实验都在 Docker 内执行：

```bash
docker exec slime_shadow bash -c "
  export PYTHONPATH=/root/Megatron-LM:/root/slime:/home/zhu11/RLKV/RLKV_github:\$PYTHONPATH
  python /root/slime/train.py \
    --custom-loss-function-path SMD.src.shadow_distillation_loss.shadow_distillation_loss_function \
    --loss-type custom_loss \
    --use-shadow-mask \
    --shadow-strategy snapkv \
    --shadow-retention-ratio 0.5 \
    ...
"
```

### 4. 模型路径
在 Docker 内引用模型时，用宿主机的绝对路径（因为 volume mount 保持路径不变）：
```
--hf-checkpoint /home/zhu11/RLKV/RLKV_github/shared_resources/models/Qwen3-1.7B
```

---

## 快速验证清单

- [ ] `docker exec slime_shadow nvidia-smi` 正常显示 GPU
- [ ] `grep shadow_mask /root/slime/slime/rollout/shadow_mask_interceptor.py` 有输出
- [ ] `python -c 'from SMD.src.shadow_mask_interceptor import ShadowMaskConfig'` 成功
- [ ] `ls /home/zhu11/RLKV/RLKV_github/shared_resources/models/` 有模型文件
- [ ] `ls /home/zhu11/RLKV/manifold/slime/train.py` 存在
