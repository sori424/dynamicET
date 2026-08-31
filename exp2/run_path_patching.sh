#!/usr/bin/env bash

python "$SCRIPT_DIR/circuit_eval.py" \
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
  --n-context-reader "$N_CONTEXT_READER" 
