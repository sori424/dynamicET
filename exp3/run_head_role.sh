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

"$PYTHON_BIN" "$SCRIPT_DIR/head_role.py" \
  --model_name "$MODEL_NAME" \
  --prompt_format "${PROMPT_FORMAT:-auto}" \
  --results_dir "$RESULTS_DIR" \
  --swap_pair 1,0 \
  --components_json "$COMPONENTS_JSON" \
  --component_set minimality_kept \
  --output_json "$RESULTS_DIR/out_all_v2.json" \
  --role-prompt-scores \
  "$@"
