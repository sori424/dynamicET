#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FOLDER="${MODEL_FOLDER:-llama8b}"
MODEL_NAME="${MODEL_NAME:-$MODEL_FOLDER}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-20}"
NUM_BOXES="${NUM_BOXES:-3}"

N_VALUE_FETCHER="${N_VALUE_FETCHER:-50}"
N_POS_TRANS="${N_POS_TRANS:-50}"
N_POS_DETECT="${N_POS_DETECT:-25}"
N_STRUCT_READ="${N_STRUCT_READ:-10}"
N_CONTEXT_READER="${N_CONTEXT_READER:-10}"

PROMPT_FORMAT="${PROMPT_FORMAT:-auto}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_ROOT/results/path_patching/$MODEL_FOLDER}"
CIRCUIT_JSON="${CIRCUIT_JSON:-}"
DATAFILE="${DATAFILE:-}"

# An alternate producer must support the same CLI, including --select-circuit.
PATH_PATCHING_ENTRYPOINT="${PATH_PATCHING_ENTRYPOINT:-$SCRIPT_DIR/run_path_patching.py}"

if [[ ! -f "$PATH_PATCHING_ENTRYPOINT" ]]; then
  printf 'Path-patching entry point not found: %s\n' "$PATH_PATCHING_ENTRYPOINT" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"

pp_args=(
  --model_name "$MODEL_NAME"
  --num_samples "$NUM_SAMPLES"
  --num_boxes "$NUM_BOXES"
  --batch_size "$BATCH_SIZE"
  --seed "$SEED"
  --n_value_fetcher "$N_VALUE_FETCHER"
  --n_pos_trans "$N_POS_TRANS"
  --n_pos_detect "$N_POS_DETECT"
  --n_struct_read "$N_STRUCT_READ"
  --n_context_reader "$N_CONTEXT_READER"
  --prompt_format "$PROMPT_FORMAT"
  --output_path "$RESULTS_DIR/"
  --select-circuit
)

if [[ -n "$DATAFILE" ]]; then
  pp_args+=(--datafile "$DATAFILE")
fi

if [[ -n "$CIRCUIT_JSON" ]]; then
  pp_args+=(--circuit-json "$CIRCUIT_JSON")
fi

exec "$PYTHON_BIN" "$PATH_PATCHING_ENTRYPOINT" "${pp_args[@]}" "$@"
