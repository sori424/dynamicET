#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/bid_intervention.py" \
  --model_name gemma12b \
  --components_json ./results/path_patching/gemma12b/circuit_eval_metrics_pruning_heldout3.json \
  --prompt_format auto \
  --swap-pair 1,0 \
  --alpha 1 \
  --num-samples 300 \
  --source-position-kinds box,object \
  --conditions no_intervention,random_query_shift,random_source_swap,random_both,query_shift,source_swap,both_restore \
  --output-json ./results/path_patching/gemma12b/bid/binding_id_qk_intervention_group_b_head_from_logit_drop.json \
  --no-per-sample
