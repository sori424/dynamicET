#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts=(
  "$ROOT_DIR/exp1/run_tracing.sh"
  "$ROOT_DIR/exp2/run_circuit.sh"
  "$ROOT_DIR/exp3/run_head_role.sh"
  "$ROOT_DIR/exp4/run_bid.sh"
)

for script in "${scripts[@]}"; do
  echo "Running: $script"
  bash "$script"
done
