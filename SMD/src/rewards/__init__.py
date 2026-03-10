"""Unified reward functions for all SMD datasets.

All reward functions are managed here and can be used both:
1. Standalone (for testing): `from src.rewards import compute_reward`
2. Integrated with Slime rm_hub (runtime registration in Docker)

Supported rm_types:
  - "rouge"     : ROUGE-L F1 with calibrated length penalty (TL;DR)
  - "govreport" : ROUGE-L F1, no length penalty (GovReport long summaries)
  - "math"      : Exact match on \\boxed{} extracted answer (GSM8K)
  - "hotpotqa"  : Fuzzy exact match with SQuAD-style normalization (HotpotQA)
"""
import re
import string


# ═══════════════════════════════════════════════════════════════════════
# TL;DR Reward (ROUGE-L + calibrated length penalty)
# ═══════════════════════════════════════════════════════════════════════

# Calibrated for Qwen3 models (longer outputs than Qwen2.5)
TLDR_LENGTH_PENALTY_LAMBDA = 0.1
TLDR_MAX_WORDS = 200


def compute_rouge_reward(response: str, label: str) -> float:
    """ROUGE-L F1 reward with mild length penalty for TL;DR.

    Returns a float roughly in [0, 1].
    """
    from rouge_score import rouge_scorer

    if not response or not response.strip():
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(label.strip(), response.strip())
    rouge_l_f1 = scores["rougeL"].fmeasure

    word_count = len(response.split())
    length_excess = max(0, word_count - TLDR_MAX_WORDS)
    length_penalty = TLDR_LENGTH_PENALTY_LAMBDA * length_excess / TLDR_MAX_WORDS

    return rouge_l_f1 - length_penalty


# ═══════════════════════════════════════════════════════════════════════
# GovReport Reward (ROUGE-L, no length penalty)
# ═══════════════════════════════════════════════════════════════════════

def compute_govreport_reward(response: str, label: str) -> float:
    """ROUGE-L F1 reward for government report summarization.

    No length penalty — GovReport summaries are naturally long (200-600 words).
    Returns a float in [0, 1].
    """
    from rouge_score import rouge_scorer

    if not response or not response.strip():
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(label.strip(), response.strip())
    return scores["rougeL"].fmeasure


# ═══════════════════════════════════════════════════════════════════════
# GSM8K Reward (Exact Match on boxed answer)
# ═══════════════════════════════════════════════════════════════════════

def _extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} format."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return ""


def _extract_last_number(text: str) -> str:
    """Fallback: extract the last number from text."""
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def compute_math_reward(response: str, label: str) -> float:
    """Sparse reward for GSM8K: 1.0 if answer matches, 0.0 otherwise.

    Tries \\boxed{} extraction first, then falls back to last-number extraction.
    """
    if not response or not response.strip():
        return 0.0

    # Try boxed format first
    predicted = _extract_boxed_answer(response)
    if not predicted:
        predicted = _extract_last_number(response)

    predicted = predicted.replace(",", "").strip()
    expected = label.replace(",", "").strip()

    if not predicted or not expected:
        return 0.0

    return 1.0 if predicted == expected else 0.0


# ═══════════════════════════════════════════════════════════════════════
# HotpotQA Reward (Fuzzy Exact Match)
# ═══════════════════════════════════════════════════════════════════════

def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison (SQuAD convention)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_hotpotqa_reward(response: str, label: str) -> float:
    """Sparse reward for HotpotQA: 1.0 if answer matches, 0.0 otherwise.

    Uses normalized exact match with fallback to substring containment,
    then token-level F1 >= 0.8 threshold.
    """
    if not response or not response.strip():
        return 0.0

    norm_response = _normalize_answer(response)
    norm_label = _normalize_answer(label)

    if not norm_label:
        return 0.0

    # Exact match
    if norm_response == norm_label:
        return 1.0

    # Substring containment
    if norm_label in norm_response:
        return 1.0

    # Token-level F1 tiebreaker
    pred_tokens = norm_response.split()
    gold_tokens = norm_label.split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    return 1.0 if f1 >= 0.8 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

REWARD_REGISTRY = {
    "rouge": compute_rouge_reward,        # TL;DR
    "govreport": compute_govreport_reward, # GovReport
    "math": compute_math_reward,           # GSM8K
    "hotpotqa": compute_hotpotqa_reward,   # HotpotQA
}


def compute_reward(rm_type: str, response: str, label: str) -> float:
    """Dispatch reward computation by type."""
    if rm_type not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown rm_type: '{rm_type}'. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[rm_type](response, label)
