# Shared Resources

This directory holds large assets (datasets and models) that are NOT tracked in git. Only `.gitkeep` placeholder files are committed.

## Datasets

To download all datasets, run the prep scripts:

```bash
pip install datasets rouge-score
python SMD/data_prep/prep_tldr.py
python SMD/data_prep/prep_gsm8k.py
python SMD/data_prep/prep_govreport.py
python SMD/data_prep/prep_hotpotqa.py
```

### Directory Structure

```
shared_resources/datasets/
├── tldr/              # Reddit TL;DR summarization (~108k train, ~6k test)
│   ├── train.jsonl    #   {prompt: post, label: summary}
│   └── test.jsonl
├── gsm8k/             # Grade school math (~7k train, ~1.3k test)
│   ├── train.jsonl    #   {prompt: 6-shot + problem, label: number}
│   └── test.jsonl
├── gov_report/        # Government report summarization (~17.5k train)
│   ├── train.jsonl    #   {prompt: report, label: summary}
│   ├── validation.jsonl
│   └── test.jsonl
└── hotpot_qa/         # Multi-hop QA with distractors (~90k train)
    ├── train.jsonl    #   {prompt: contexts + question, label: answer}
    └── validation.jsonl
```

## Models

Models are downloaded via `huggingface-cli download` and converted to `torch_dist` format using Slime's conversion tool. See `SMD/README.md` for setup instructions.

```
shared_resources/models/
├── Qwen3-0.6B/
├── Qwen3-0.6B_torch_dist/
├── Qwen3-1.7B/
├── Qwen3-1.7B_torch_dist/
├── Qwen3-4B-Instruct-2507/
└── Qwen3-4B-Instruct-2507_torch_dist/
```
