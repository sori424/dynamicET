#!/usr/bin/env bash

cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/tracing.py" \
  context \
  gemma9b \
  --vocab_tag BOXES \
  --num_samples 300 \
  --device_id 0 \
  --num_entities 3 \
  --query_name 0 \
  --swap_box_a 1 \
  --swap_box_b 0 \
  --names_trace_mode swap_only
