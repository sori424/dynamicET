#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FOLDER="${MODEL_FOLDER:-llama8b}"
MODEL_NAME="${MODEL_NAME:-$MODEL_FOLDER}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/path_patching/$MODEL_FOLDER}"
COMPONENTS_JSON="${COMPONENTS_JSON:-$RESULTS_DIR/circuit_eval_metrics_pruning_heldout3.json}"

"$PYTHON_BIN" "$SCRIPT_DIR/circuit_eval.py" \
  --models "$MODEL_FOLDER" \
  --model-name "$MODEL_NAME" \
  --results-dir "$RESULTS_DIR" \
  --prompt_format "${PROMPT_FORMAT:-auto}" \
  --output-json "$COMPONENTS_JSON" \
  --random-circuit-iters 10 \
  --run-minimality \
  --minimality-samples 50 \
  --minimality-threshold 0.010 \
  --minimality-metric mean_label_logit \
  --minimality-criterion minimality-py \
  --run-heldout \
  --heldout-prompt-id-start 300 \
  --heldout-samples 300 \
  "$@"
