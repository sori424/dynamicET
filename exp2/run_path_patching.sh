#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_FOLDER="${MODEL_FOLDER:-llama8b}"
MODEL_NAME="${MODEL_NAME:-llama-8b-hf}"
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
CIRCUIT_JSON="${CIRCUIT_JSON:-$RESULTS_DIR/circuit.json}"
DATAFILE="${DATAFILE:-}"

# The producer lives beside this wrapper. PATH_PATCHING_ENTRYPOINT remains
# overridable for backwards compatibility with older external checkouts.
PATH_PATCHING_ENTRYPOINT="${PATH_PATCHING_ENTRYPOINT:-$SCRIPT_DIR/run_path_patching.py}"

if [[ ! -f "$PATH_PATCHING_ENTRYPOINT" ]]; then
  printf 'Path-patching entry point not found: %s\n' "$PATH_PATCHING_ENTRYPOINT" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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
)

if [[ -n "$DATAFILE" ]]; then
  pp_args+=(--datafile "$DATAFILE")
fi

printf 'Running path patching for %s...\n' "$MODEL_NAME"
"$PYTHON_BIN" "$PATH_PATCHING_ENTRYPOINT" "${pp_args[@]}" "$@"

# Older producers save the structural-reader tensor under this legacy name.
if [[ -f "$RESULTS_DIR/struct_reader.pt" && ! -f "$RESULTS_DIR/struct_reader_swap_target.pt" ]]; then
  cp "$RESULTS_DIR/struct_reader.pt" "$RESULTS_DIR/struct_reader_swap_target.pt"
fi

required_tensors=(
  value_fetcher.pt
  pos_transmitter.pt
  pos_detector.pt
  struct_reader_swap_target.pt
)

for tensor in "${required_tensors[@]}"; do
  if [[ ! -f "$RESULTS_DIR/$tensor" ]]; then
    printf 'Missing expected path-patching tensor: %s\n' "$RESULTS_DIR/$tensor" >&2
    exit 1
  fi
done

if [[ ! -f "$RESULTS_DIR/struct_reader_context_target_box_to_target_box.pt" ]]; then
  printf '%s\n' \
    'Note: no context-target structural-reader tensor was produced; that optional stage will be omitted.' >&2
fi

printf 'Selecting circuit heads...\n'
cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/circuit_eval.py" \
  --models "$MODEL_FOLDER" \
  --model-name "$MODEL_NAME" \
  --results-dir "$RESULTS_DIR" \
  --output-json "$CIRCUIT_JSON" \
  --prompt-format "$PROMPT_FORMAT" \
  --num-entities "$NUM_BOXES" \
  --n-value-fetcher "$N_VALUE_FETCHER" \
  --n-pos-trans "$N_POS_TRANS" \
  --n-pos-detect "$N_POS_DETECT" \
  --n-struct-read "$N_STRUCT_READ" \
  --n-context-reader "$N_CONTEXT_READER" \
  --no-model-eval

printf 'Circuit written to %s\n' "$CIRCUIT_JSON"
