#!/usr/bin/env python3
"""
Exp 10: Training Stability — Multi-seed Reproducibility Test

Purpose:
    Demonstrate that SMD training is stable and reproducible.
    Runs the same configuration with multiple random seeds,
    reports mean ± std across runs.

Configuration:
    Model: Qwen3-1.7B | Dataset: TL;DR
    Method: SMD (R-KV 50%, λ=0.1) and Dense baseline
    Seeds: 42, 123, 456, 789, 2026
    Rollouts: 500 per run

Output:
    results/<seed>/metrics.json per run
    stability_report.json with aggregated statistics

Usage:
    python run_stability.py --output-dir /workspace/results/exp10
    python run_stability.py --output-dir /workspace/results/exp10 --method dense
"""
import argparse
import json
import os
import sys
import time
import random as stdlib_random
import subprocess
from pathlib import Path

import numpy as np
import torch

# ── Configuration ────────────────────────────────────────────
RLKV_ROOT = os.environ.get("RLKV_ROOT", "/workspace/RLKV")
MODEL_PATH = os.path.join(RLKV_ROOT, "shared_resources/models/Qwen3-1.7B")
DATA_FILE = os.path.join(RLKV_ROOT, "shared_resources/datasets/tldr/train.jsonl")

SEEDS = [42, 123, 456, 789, 2026]
NUM_ROLLOUTS = 500
RM_TYPE = "rouge"
MAX_RESP_LEN = 128

sys.path.insert(0, RLKV_ROOT)


def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    stdlib_random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_seed(seed: int, method: str, output_dir: str) -> dict:
    """Run one training run with a given seed."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from SMD.src.rewards import compute_reward

    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Check if already completed
    metrics_path = os.path.join(seed_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print(f"  [Seed {seed}] Already completed, loading results...")
        with open(metrics_path) as f:
            return json.load(f)

    set_all_seeds(seed)
    print(f"  [Seed {seed}] Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-7, weight_decay=0.1
    )

    # Load dataset
    data = []
    with open(DATA_FILE) as f:
        for line in f:
            data.append(json.loads(line.strip()))
            if len(data) >= NUM_ROLLOUTS * 4:
                break

    metrics_log = []
    start_time = time.time()

    for step in range(NUM_ROLLOUTS):
        sample = data[step % len(data)]
        prompt = sample.get("prompt", "")
        label = sample.get("label", "")

        # Generate rollouts
        model.eval()
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt",
                           truncation=True, max_length=2048).to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        responses = []
        with torch.no_grad():
            for _ in range(2):
                output = model.generate(
                    **inputs, max_new_tokens=MAX_RESP_LEN,
                    temperature=1.0, do_sample=True, top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
                resp_ids = output[0][prompt_len:]
                resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                responses.append({"text": resp_text, "ids": resp_ids})

        model.train()

        # Compute rewards
        rewards = [compute_reward(RM_TYPE, r["text"], label) for r in responses]

        # GRPO loss
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8) \
            if rewards_t.std() > 1e-8 else rewards_t - rewards_t.mean()

        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=model.device)

        for i, (resp, adv) in enumerate(zip(responses, advantages)):
            resp_ids = resp["ids"].to(model.device)
            if len(resp_ids) == 0:
                continue
            full_ids = torch.cat([inputs["input_ids"][0], resp_ids]).unsqueeze(0)
            outputs = model(input_ids=full_ids,
                            attention_mask=torch.ones_like(full_ids))
            logits = outputs.logits[0, prompt_len - 1:-1]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_lp = log_probs.gather(1, resp_ids.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                ref_out = ref_model(input_ids=full_ids,
                                    attention_mask=torch.ones_like(full_ids))
                ref_logits = ref_out.logits[0, prompt_len - 1:-1]
                ref_lp = torch.log_softmax(ref_logits, dim=-1) \
                    .gather(1, resp_ids.unsqueeze(1)).squeeze(1)

            policy_loss = -(token_lp * adv.to(model.device)).mean()
            kl_loss = 0.5 * (token_lp - ref_lp).mean()
            total_loss = total_loss + policy_loss + kl_loss

        total_loss = total_loss / max(len(responses), 1)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        avg_reward = float(np.mean(rewards))
        step_info = {
            "step": step, "loss": total_loss.item(),
            "avg_reward": avg_reward, "seed": seed,
        }
        metrics_log.append(step_info)

        if step % 50 == 0 or step == NUM_ROLLOUTS - 1:
            print(f"  [Seed {seed} | Step {step:4d}/{NUM_ROLLOUTS}] "
                  f"loss={total_loss.item():.4f} reward={avg_reward:.4f}")

    # Save
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Cleanup GPU
    del model, ref_model, optimizer
    torch.cuda.empty_cache()

    return metrics_log


def generate_stability_report(output_dir: str, method: str):
    """Aggregate results across seeds and compute statistics."""
    all_final_rewards = []
    all_final_losses = []
    per_seed_curves = {}

    for seed in SEEDS:
        metrics_path = os.path.join(output_dir, f"seed_{seed}", "metrics.json")
        if not os.path.exists(metrics_path):
            print(f"  Warning: seed {seed} not found, skipping")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Last 50 steps average
        last50 = metrics[-50:]
        avg_reward = np.mean([m["avg_reward"] for m in last50])
        avg_loss = np.mean([m["loss"] for m in last50])
        all_final_rewards.append(avg_reward)
        all_final_losses.append(avg_loss)

        # Full reward curve (every 10 steps)
        per_seed_curves[seed] = [m["avg_reward"] for m in metrics[::10]]

    if not all_final_rewards:
        print("No completed runs found!")
        return

    report = {
        "method": method,
        "seeds": SEEDS,
        "num_rollouts": NUM_ROLLOUTS,
        "completed_seeds": len(all_final_rewards),
        "reward_mean": float(np.mean(all_final_rewards)),
        "reward_std": float(np.std(all_final_rewards)),
        "reward_min": float(np.min(all_final_rewards)),
        "reward_max": float(np.max(all_final_rewards)),
        "loss_mean": float(np.mean(all_final_losses)),
        "loss_std": float(np.std(all_final_losses)),
        "per_seed_final_reward": {
            str(SEEDS[i]): float(all_final_rewards[i])
            for i in range(len(all_final_rewards))
        },
    }

    report_path = os.path.join(output_dir, "stability_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Stability Report: {method}")
    print(f"{'='*60}")
    print(f"  Seeds completed: {len(all_final_rewards)}/{len(SEEDS)}")
    print(f"  Reward (last 50): {report['reward_mean']:.4f} ± {report['reward_std']:.4f}")
    print(f"  Loss   (last 50): {report['loss_mean']:.4f} ± {report['loss_std']:.4f}")
    print(f"  Range: [{report['reward_min']:.4f}, {report['reward_max']:.4f}]")
    print(f"  Saved to: {report_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 10: Training Stability")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", default="dense",
                        choices=["dense", "smd"],
                        help="Training method to test stability")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Override default seeds")
    parser.add_argument("--report-only", action="store_true",
                        help="Only generate report from existing results")
    args = parser.parse_args()

    if args.seeds:
        SEEDS = args.seeds

    method_dir = os.path.join(args.output_dir, args.method)
    os.makedirs(method_dir, exist_ok=True)

    if not args.report_only:
        print(f"\n{'='*60}")
        print(f"  Exp 10: Stability Test — {args.method}")
        print(f"  Model: Qwen3-1.7B | Dataset: TL;DR")
        print(f"  Seeds: {SEEDS} | Rollouts: {NUM_ROLLOUTS}")
        print(f"{'='*60}\n")

        for seed in SEEDS:
            print(f"\n--- Running seed {seed} ---")
            run_single_seed(seed, args.method, method_dir)

    generate_stability_report(method_dir, args.method)
