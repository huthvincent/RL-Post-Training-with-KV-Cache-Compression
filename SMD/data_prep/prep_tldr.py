"""Download and format Reddit TL;DR dataset for RL training.

Source: CarperAI/openai_summarize_tldr (HuggingFace)
Output: JSONL with {prompt, label} format
- prompt: Reddit post text (SUBREDDIT + TITLE + POST)
- label: Ground truth TL;DR summary
"""
import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.environ.get(
    "SMD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "shared_resources", "datasets", "tldr"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading TL;DR dataset (CarperAI/openai_summarize_tldr)...")
ds = load_dataset("CarperAI/openai_summarize_tldr")

for split_name in ["train", "test"]:
    if split_name not in ds:
        continue
    split = ds[split_name]
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    count = 0
    with open(out_path, "w") as f:
        for row in split:
            prompt = row["prompt"].strip()
            label = row["label"].strip()
            if not prompt or not label:
                continue
            entry = {"prompt": prompt, "label": label}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    print(f"  {split_name}: {count} examples -> {out_path}")

print("\nTL;DR prep complete!")
