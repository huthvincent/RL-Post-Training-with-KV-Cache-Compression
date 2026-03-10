"""Download and format GSM8K dataset for RL training.

Source: openai/gsm8k (HuggingFace), with 6-shot prompt template
Output: JSONL with {prompt, label} format
- prompt: 6-shot few-shot examples + target problem (boxed answer format)
- label: numeric answer string (e.g., "42")
"""
import json
import os
import re
from datasets import load_dataset

OUTPUT_DIR = os.environ.get(
    "SMD_DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "shared_resources", "datasets", "gsm8k"),
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Downloading GSM8K dataset...")
ds = load_dataset("openai/gsm8k", "main")

# Extract numeric answer from GSM8K format: "... #### 42"
def extract_answer(solution: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+)", solution)
    if match:
        return match.group(1).replace(",", "")
    return ""


# Build 6-shot prompt template from first 6 training examples
def build_fewshot_prompt(examples: list, target_question: str) -> str:
    header = ("Solve the following math problem. Please reason step by step, "
              "and put your final answer within \\boxed{}.\n\n"
              "Here are some examples:\n")

    shots = []
    for i, ex in enumerate(examples):
        answer = extract_answer(ex["answer"])
        # Reformulate solution with \\boxed{}
        solution_text = ex["answer"].split("####")[0].strip()
        shots.append(
            f"Example {i+1}:\n"
            f"Problem: {ex['question']}\n"
            f"Solution: {solution_text}\n"
            f"The answer is \\boxed{{{answer}}}."
        )

    return header + "\n\n".join(shots) + f"\n\nNow solve this problem:\nProblem: {target_question}"


# Use first 6 examples as few-shot demonstrations
train_data = list(ds["train"])
fewshot_examples = train_data[:6]

for split_name, split_key in [("train", "train"), ("test", "test")]:
    if split_key not in ds:
        continue
    data = list(ds[split_key])
    out_path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
    count = 0
    with open(out_path, "w") as f:
        start_idx = 6 if split_key == "train" else 0  # Skip fewshot examples in train
        for row in data[start_idx:]:
            question = row["question"].strip()
            answer = extract_answer(row["answer"])
            if not question or not answer:
                continue
            prompt = build_fewshot_prompt(fewshot_examples, question)
            entry = {"prompt": prompt, "label": answer}
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    print(f"  {split_name}: {count} examples -> {out_path}")

print("\nGSM8K prep complete!")
