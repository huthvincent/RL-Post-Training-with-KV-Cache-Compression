#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Exp 10: Training Stability (Multi-Seed Reproducibility)
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

OUTPUT_DIR="${1:-/workspace/results/exp10_stability}"

echo "Running stability test: Dense (5 seeds)..."
python "$(dirname "$0")/run_stability.py" --output-dir "$OUTPUT_DIR" --method dense

echo "Running stability test: SMD (5 seeds)..."
python "$(dirname "$0")/run_stability.py" --output-dir "$OUTPUT_DIR" --method smd

echo "Done! Reports in $OUTPUT_DIR/"
