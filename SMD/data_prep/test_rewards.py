"""Sanity check: test all 4 dataset reward functions.

Run from repo root:
    python SMD/data_prep/test_rewards.py
"""
import json
import os
import sys

# Add project root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "SMD"))

from src.rewards import compute_reward

DATASETS_DIR = os.path.join(REPO_ROOT, "shared_resources", "datasets")

passed = 0
total = 0


def test_dataset(name, rm_type, data_dir, test_cases):
    """Test a reward function with synthetic examples."""
    global passed, total
    print(f"\n=== {name} (rm_type='{rm_type}') ===")

    # Check if data exists
    train_path = os.path.join(data_dir, "train.jsonl")
    if os.path.exists(train_path):
        with open(train_path) as f:
            sample = json.loads(f.readline())
        print(f"  Data: {train_path}")
        print(f"  Prompt: {len(sample['prompt'].split())} words")
        print(f"  Label:  {repr(sample['label'][:80])}...")
    else:
        print(f"  ⚠️  Data not found at {train_path} (run prep script first)")

    # Run test cases
    for desc, response, label, expected_check in test_cases:
        total += 1
        reward = compute_reward(rm_type, response, label)
        ok = expected_check(reward)
        status = "✅" if ok else "❌"
        print(f"  {status} {desc}: reward={reward:.4f}")
        if ok:
            passed += 1
        else:
            print(f"      FAILED: expected check failed for reward={reward}")


# ── TL;DR ─────────────────────────────────────────────────────────────
test_dataset(
    "TL;DR", "rouge",
    os.path.join(DATASETS_DIR, "tldr"),
    [
        ("Exact match", "This is a test summary", "This is a test summary", lambda r: r > 0.95),
        ("Partial match", "test summary", "This is a test summary about cats", lambda r: 0 < r < 1),
        ("Irrelevant", "Completely unrelated text about dogs", "A summary of economic policy", lambda r: r < 0.2),
        ("Empty", "", "Some label", lambda r: r == 0.0),
    ],
)

# ── GSM8K ─────────────────────────────────────────────────────────────
test_dataset(
    "GSM8K", "math",
    os.path.join(DATASETS_DIR, "gsm8k"),
    [
        ("Boxed correct", "So the answer is \\boxed{42}.", "42", lambda r: r == 1.0),
        ("Boxed wrong", "The answer is \\boxed{99}.", "42", lambda r: r == 0.0),
        ("Last number correct", "After calculation we get 42", "42", lambda r: r == 1.0),
        ("Empty", "", "42", lambda r: r == 0.0),
    ],
)

# ── GovReport ─────────────────────────────────────────────────────────
test_dataset(
    "GovReport", "govreport",
    os.path.join(DATASETS_DIR, "gov_report"),
    [
        ("Exact match", "The government report discusses policy", "The government report discusses policy", lambda r: r > 0.95),
        ("Irrelevant", "Cats are furry animals", "Government spending increased by 5%", lambda r: r < 0.2),
        ("Empty", "", "Some summary", lambda r: r == 0.0),
    ],
)

# ── HotpotQA ──────────────────────────────────────────────────────────
test_dataset(
    "HotpotQA", "hotpotqa",
    os.path.join(DATASETS_DIR, "hotpot_qa"),
    [
        ("Exact match", "Paris", "Paris", lambda r: r == 1.0),
        ("Embedded answer", "The answer is Paris.", "Paris", lambda r: r == 1.0),
        ("Case insensitive", "marie curie", "Marie Curie", lambda r: r == 1.0),
        ("Wrong answer", "London", "Paris", lambda r: r == 0.0),
        ("Empty", "", "Paris", lambda r: r == 0.0),
    ],
)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed}/{total} tests passed")
if passed == total:
    print("🎉 ALL TESTS PASSED!")
else:
    print("⚠️  Some tests failed!")
    sys.exit(1)
