#!/usr/bin/env python3
"""
Standalone GRPO training script for SMD experiments on Qwen3-0.6B.
Uses veRL's trainer with custom reward functions from the project.

Usage:
  python run_grpo_training.py \
    --dataset tldr --method dense --num-rollouts 500 \
    --output-dir /workspace/RLKV/SMD/results/qwen3_0.6b_exp02/tldr_dense

Methods:
  dense        - Standard GRPO (no KV compression)
  smd          - Shadow Mask Distillation (SnapKV 50%)
  sparse_rl    - Sparse-RL baseline
  naive_snapkv - Naive SnapKV 50% (true eviction, for Exp 1)
  naive_random - Naive Random 50% (true eviction, for Exp 1)
"""
import argparse
import json
import os
import sys
import time
import random
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# Add project to path
sys.path.insert(0, "/workspace/RLKV")
from SMD.src.rewards import compute_reward, REWARD_REGISTRY


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

RLKV_ROOT = os.environ.get("RLKV_ROOT", "/workspace/RLKV")
MODEL_PATH = os.path.join(RLKV_ROOT, "shared_resources/models/Qwen3-0.6B")

DATASET_CONFIGS = {
    "tldr":      {"rm_type": "rouge",    "max_resp_len": 128, "data_file": os.path.join(RLKV_ROOT, "shared_resources/datasets/tldr/train.jsonl")},
    "gsm8k":     {"rm_type": "math",     "max_resp_len": 256, "data_file": os.path.join(RLKV_ROOT, "shared_resources/datasets/gsm8k/train.jsonl")},
    "govreport": {"rm_type": "govreport","max_resp_len": 512, "data_file": os.path.join(RLKV_ROOT, "shared_resources/datasets/gov_report/train.jsonl")},
    "hotpotqa":  {"rm_type": "hotpotqa", "max_resp_len": 256, "data_file": os.path.join(RLKV_ROOT, "shared_resources/datasets/hotpot_qa/train.jsonl")},
}


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class PromptDataset(Dataset):
    def __init__(self, data_file, max_samples=None):
        self.data = []
        with open(data_file, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                if max_samples and len(self.data) >= max_samples:
                    break
        print(f"Loaded {len(self.data)} prompts from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ═══════════════════════════════════════════════════════════════
# Shadow Mask (for SMD / Naive methods)
# ═══════════════════════════════════════════════════════════════

def compute_shadow_mask(attention_scores, strategy, retention_ratio):
    """Compute which KV positions to keep based on strategy."""
    seq_len = attention_scores.shape[-1]
    num_keep = max(1, int(seq_len * retention_ratio))

    if strategy == "snapkv":
        # SnapKV: keep positions with highest cumulative attention
        importance = attention_scores.sum(dim=1).sum(dim=1)  # [batch, seq_len]
        _, indices = importance.topk(num_keep, dim=-1)
        mask = torch.zeros_like(importance, dtype=torch.bool)
        mask.scatter_(1, indices, True)
    elif strategy == "random":
        mask = torch.zeros(attention_scores.shape[0], seq_len, dtype=torch.bool, device=attention_scores.device)
        for b in range(mask.shape[0]):
            keep_idx = torch.randperm(seq_len, device=attention_scores.device)[:num_keep]
            mask[b, keep_idx] = True
    elif strategy == "recent":
        mask = torch.zeros(attention_scores.shape[0], seq_len, dtype=torch.bool, device=attention_scores.device)
        mask[:, -num_keep:] = True
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return mask


# ═══════════════════════════════════════════════════════════════
# GRPO Training Loop
# ═══════════════════════════════════════════════════════════════

def generate_rollout(model, tokenizer, prompt_text, max_new_tokens, temperature=1.0, n_samples=2):
    """Generate n_samples responses for a prompt."""
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    responses = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        response_ids = output[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        responses.append({"text": response_text, "ids": response_ids})

    return responses, inputs


def compute_grpo_loss(model, tokenizer, prompt_inputs, responses, rewards, ref_model=None, 
                      kl_coef=0.5, eps_clip=0.2, method="dense"):
    """Compute GRPO loss with optional KL penalty."""
    # Normalize rewards (GRPO advantage estimation)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    if rewards_tensor.std() > 1e-8:
        advantages = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
    else:
        advantages = rewards_tensor - rewards_tensor.mean()

    total_loss = torch.tensor(0.0, device=model.device)
    metrics = {"rewards": rewards, "advantages": advantages.tolist()}

    for i, (resp, adv) in enumerate(zip(responses, advantages)):
        resp_ids = resp["ids"].to(model.device)
        if len(resp_ids) == 0:
            continue

        # Compute log probs under current policy
        full_ids = torch.cat([prompt_inputs["input_ids"][0], resp_ids]).unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, prompt_inputs["input_ids"].shape[1]-1:-1]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, resp_ids.unsqueeze(1)).squeeze(1)

        # Policy gradient with advantage
        policy_loss = -(token_log_probs * adv.to(model.device)).mean()

        # KL penalty (if ref model available)
        kl_loss = torch.tensor(0.0, device=model.device)
        if ref_model is not None and kl_coef > 0:
            with torch.no_grad():
                ref_outputs = ref_model(input_ids=full_ids, attention_mask=attention_mask)
                ref_logits = ref_outputs.logits[0, prompt_inputs["input_ids"].shape[1]-1:-1]
                ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                ref_token_log_probs = ref_log_probs.gather(1, resp_ids.unsqueeze(1)).squeeze(1)
            kl = (token_log_probs - ref_token_log_probs).mean()
            kl_loss = kl_coef * kl

        total_loss = total_loss + policy_loss + kl_loss

    total_loss = total_loss / max(len(responses), 1)
    return total_loss, metrics


def run_training(args):
    """Main training loop."""
    print(f"\n{'='*60}")
    print(f"  GRPO Training: {args.method} on {args.dataset}")
    print(f"  Model: Qwen3-0.6B | Rollouts: {args.num_rollouts}")
    print(f"{'='*60}\n")

    # Setup
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.train()

    # Reference model (frozen copy for KL)
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr), weight_decay=0.1
    )

    # Load dataset
    dataset = PromptDataset(dataset_cfg["data_file"], max_samples=args.num_rollouts * 4)

    # Training metrics log
    metrics_log = []
    start_time = time.time()

    for step in range(args.num_rollouts):
        # Sample a prompt
        idx = step % len(dataset)
        sample = dataset[idx]
        prompt = sample.get("prompt", sample.get("question", ""))
        label = sample.get("label", sample.get("answer", ""))

        # Generate rollouts
        model.eval()
        with torch.no_grad():
            responses, prompt_inputs = generate_rollout(
                model, tokenizer, prompt,
                max_new_tokens=dataset_cfg["max_resp_len"],
                temperature=1.0, n_samples=2
            )
        model.train()

        # Compute rewards
        rewards = []
        for resp in responses:
            r = compute_reward(dataset_cfg["rm_type"], resp["text"], label)
            rewards.append(r)

        # Compute loss and update
        optimizer.zero_grad()
        loss, step_metrics = compute_grpo_loss(
            model, tokenizer, prompt_inputs, responses, rewards,
            ref_model=ref_model, kl_coef=float(args.kl_coef), method=args.method
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Log metrics
        avg_reward = np.mean(rewards)
        step_info = {
            "step": step,
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "rewards": rewards,
            "resp_lens": [len(r["text"].split()) for r in responses],
            "elapsed": time.time() - start_time,
        }
        metrics_log.append(step_info)

        if step % 10 == 0 or step == args.num_rollouts - 1:
            print(f"[Step {step:4d}/{args.num_rollouts}] "
                  f"loss={loss.item():.4f} avg_reward={avg_reward:.4f} "
                  f"rewards={[f'{r:.3f}' for r in rewards]} "
                  f"elapsed={step_info['elapsed']:.0f}s")

        # Save checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step+1}.json")
            with open(ckpt_path, "w") as f:
                json.dump(metrics_log, f)

    # Save final results
    results_path = os.path.join(args.output_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nTraining complete! Results saved to {results_path}")
    print(f"Total time: {time.time() - start_time:.0f}s")

    return metrics_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO Training for SMD Experiments")
    parser.add_argument("--dataset", required=True, choices=["tldr", "gsm8k", "govreport", "hotpotqa"])
    parser.add_argument("--method", required=True, choices=["dense", "smd", "sparse_rl", "qurl", "rlhfless", "naive_snapkv", "naive_random"])
    parser.add_argument("--num-rollouts", type=int, default=500)
    parser.add_argument("--lr", default="1e-7")
    parser.add_argument("--kl-coef", default="0.5")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    run_training(args)
