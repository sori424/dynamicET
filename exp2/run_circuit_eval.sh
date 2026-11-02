#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/circuit_eval.py" \
  --models llama8b \
  --results-dir results/path_patching/llama8b \
  --prompt_format auto \
  --output-json results/path_patching/llama8b/circuit_eval_metrics_pruning_heldout3.json \
  --random-circuit-iters 10 \
  --run-minimality \
  --minimality-samples 50 \
  --minimality-threshold 0.010 \
  --minimality-metric mean_label_logit \
  --minimality-criterion minimality-py \
  --run-heldout \
  --heldout-prompt-id-start 300 \
  --heldout-samples 300
