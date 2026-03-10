"""Download and format HotpotQA (distractor setting) for RL training.

Source: hotpot_qa (HuggingFace), 'distractor' config
Output: JSONL with {prompt, label} format
- prompt: "[Context 1]\n...\n[Context 10]\n\nQuestion: ...\nAnswer:"
- label: ground truth answer string
"""
import json
import os
from datasets import load_dataset

OUTPUT_DIR = os.environ.get(
    "SMD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "shared_resources", "datasets", "hotpot_qa"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading HotpotQA (distractor) dataset...")
ds = load_dataset("hotpot_qa", "distractor")

for split_name in ["train", "validation"]:
    if split_name not in ds:
        continue
    split = ds[split_name]
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    count = 0
    with open(out_path, "w") as f:
        for row in split:
            question = row["question"].strip()
            answer = row["answer"].strip()
            if not question or not answer:
                continue

            # Build context from all paragraphs (2 supporting + 8 distractors)
            contexts = []
            for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
                para_text = " ".join(sentences)
                contexts.append(f"[{title}]\n{para_text}")

            context_block = "\n\n".join(contexts)
            prompt = f"Read the following contexts and answer the question.\n\n{context_block}\n\nQuestion: {question}\nAnswer:"
            entry = {"prompt": prompt, "label": answer}
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

print("\nHotpotQA prep complete!")
