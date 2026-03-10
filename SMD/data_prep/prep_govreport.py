"""Download and format GovReport dataset for RL training.

Source: ccdv/govreport-summarization (HuggingFace)
Output: JSONL with {prompt, label} format
- prompt: "Summarize the following government report:\n\n{report}"
- label: ground truth summary
"""
import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.environ.get(
    "SMD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "shared_resources", "datasets", "gov_report"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading GovReport dataset...")
ds = load_dataset("ccdv/govreport-summarization")

for split_name in ["train", "validation", "test"]:
    if split_name not in ds:
        continue
    split = ds[split_name]
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    count = 0
    with open(out_path, "w") as f:
        for row in split:
            report = row["report"].strip()
            summary = row["summary"].strip()
            if not report or not summary:
                continue
            prompt = f"Summarize the following government report:\n\n{report}"
            entry = {"prompt": prompt, "label": summary}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    print(f"  {split_name}: {count} examples -> {out_path}")

# Print sample stats
train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
with open(train_path) as f:
    first = json.loads(f.readline())
    prompt_words = len(first["prompt"].split())
    label_words = len(first["label"].split())
    print(f"\nSample stats:")
    print(f"  Prompt words: {prompt_words}")
    print(f"  Label words:  {label_words}")

print("\nGovReport prep complete!")
