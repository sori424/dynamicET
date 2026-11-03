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

"$PYTHON_BIN" "$SCRIPT_DIR/bid_intervention.py" \
  --model_name "$MODEL_NAME" \
  --components_json "$COMPONENTS_JSON" \
  --prompt_format "${PROMPT_FORMAT:-auto}" \
  --swap-pair 1,0 \
  --alpha 1 \
  --num-samples 300 \
  --source-position-kinds box,object \
  --conditions no_intervention,random_query_shift,random_source_swap,random_both,query_shift,source_swap,both_restore \
  --output-json "$RESULTS_DIR/bid/binding_id_qk_intervention_group_b_head_from_logit_drop.json" \
  --no-per-sample \
  "$@"
