#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

python "$SCRIPT_DIR/tracing.py" \
  context \
  gemma9b \
  --vocab_tag BOXES \
  --num_samples 100 \
  --device_id 0 \
  --num_entities 3 \
  --query_name 0 \
  --swap_box_a 1 \
  --swap_box_b 0 \
  --names_trace_mode swap_only
