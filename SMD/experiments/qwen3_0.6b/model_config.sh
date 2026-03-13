#!/bin/bash
# Model configuration for Qwen3-0.6B
# Architecture from config.json: 28 layers, hidden 1024, heads 16, kv_heads 8, ffn 3072

MODEL_ARGS=(
  --num-layers 28
  --hidden-size 1024
  --num-attention-heads 16
  --group-query-attention
  --num-query-groups 8
  --ffn-hidden-size 3072
  --max-position-embeddings 40960
  --seq-length 4096
  --tokenizer-type HuggingFaceTokenizer
  --tokenizer-model /workspace/RLKV/shared_resources/models/Qwen3-0.6B
  --bf16
  --swiglu
  --disable-bias-linear
  --use-rotary-position-embeddings
  --rotary-base 1000000
  --normalization RMSNorm
  --no-position-embedding
  --untie-embeddings-and-output-weights
)
