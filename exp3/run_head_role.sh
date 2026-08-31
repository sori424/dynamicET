#!/usr/bin/env bash


python "$SCRIPT_DIR/head_role.py" \
  --model_name llama8b \
  --prompt_format auto \
  --results_dir ./results/path_patching/llama8b \
  --swap_pair 1,0 \
  --components_json ./results/path_patching/llama8b/circuit_eval_metrics_pruning_heldout3.json \
  --component_set minimality_kept \
  --output_json ./results/path_patching/llama8b/out_all_v2.json \
  --role-prompt-scores
