#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# bitsandbytes can call torch.compile while TransformerLens imports. Disable
# Dynamo early to avoid a Torch Inductor import mismatch in this environment.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def disable_optional_vision_backends() -> None:
    try:
        from transformers.utils import import_utils
    except Exception:
        return
    import_utils._torchvision_available = False
    import_utils._torchvision_version = "N/A"


disable_optional_vision_backends()

MODEL_NAME_BY_FOLDER = {
    "gemma9b": "google/gemma-2-9b-it",
    "gemma2-9b-it": "google/gemma-2-9b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "gemma3-12b-it": "google/gemma-3-12b-it",
    "google__gemma-2-9b-it": "google/gemma-2-9b-it",
    "google__gemma-3-12b-it": "google/gemma-3-12b-it",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama__Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama__Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "llama-8b-hf",
    "llama-8b-hf": "llama-8b-hf",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen8b": "Qwen/Qwen3-8B",
}

PROMPT_FORMATS = (
    "raw",
    "auto",
    "chat-template",
    "chat-prefill",
    "gemma-it",
    "gemma-it-prefill",
    "llama-instruct",
    "llama-instruct-prefill",
    "qwen-instruct",
    "qwen-instruct-prefill",
)

METRIC_CHOICES = (
    "accuracy",
    "full_vocab_accuracy",
    "candidate_accuracy",
    "mean_label_prob",
    "mean_label_logit",
    "mean_candidate_margin",
)

MINIMALITY_RAW_DROP_METRICS = frozenset(
    {
        "mean_label_logit",
        "mean_candidate_margin",
    }
)

SYSTEM_PROMPT = "Respond with a single word only. No punctuation, no explanation."
ANSWER_PREFILL_RE = re.compile(r"\s*Answer:\s*$", re.IGNORECASE)

STAGE_FILES = {
    "value_fetcher": "value_fetcher.pt",
    "pos_transmitter": "pos_transmitter.pt",
    "pos_detector": "pos_detector.pt",
    "struct_reader_swap_target": "struct_reader_swap_target.pt",
    "struct_reader_context_target_box_to_target_box": (
        "struct_reader_context_target_box_to_target_box.pt"
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp2.eval_core import (
    SimpleTLUtils,
    CircuitComponent,
    DictTensorDataset,
    find_binding_root,
    import_binding_modules,
    set_seed,
    normalize_device,
    torch_load_cpu,
    tensor_topk_components,
    all_filtered_components,
    hook_component_name,
    hook_name,
    layer_from_hook_name,
    span_starts,
    last_token_indices,
    last_token_indices_from_attention_mask,
    infer_prompt_format,
    resolve_prompt_format,
    _apply_chat_template,
    _split_assistant_prefill,
    _tokenizer_has_token,
    _format_llama_prefill_fallback,
    _format_llama_prompt_fallback,
    _apply_chat_prefill_template,
    format_prompt_for_model,
    decode_prompts_from_tokens,
    encode_prompt_texts,
    first_answer_token_id,
    answer_token_ids,
    padded_token_id_tensor,
    answer_words_from_tokens,
    position_fields_from_tokens,
)

from exp2.eval_data import (
    quiet_build_example_refactored,
    paired_contexts,
    normal_context_index,
    path_patching_contexts,
    swap_statement,
    swap_target_box_from_context,
    assign_swap_position_fields,
    build_box_dataset,
)

from exp2.eval_metrics import (
    move_batch_to_device,
    position_tensor,
    new_metrics,
    _valid_unique_token_ids,
    update_metrics,
    finalize_metrics,
    compute_mean_activations,
    mean_ablate_activation,
    evaluate_metrics,
    metrics_mean,
    metrics_std,
)

from exp2.circuit_components import (
    selected_stage_heads,
    add_components,
    append_attention_pattern_context_reader_components,
    dedupe_components,
    circuit_by_position_from_components,
    component_count_by_position,
    slice_dict_tensor_dataset,
    ranges_overlap,
    compute_context_reader_attention_scores,
    update_output_circuit_fields,
    tensor_topk_from_candidates,
    load_circuit_components,
    random_components_like,
    metric_value,
    remove_one_component,
    component_key,
    remove_component_set,
    full_model_components_for_positions,
    sample_component_subset,
)

from exp2.completeness import (
    completeness_summary,
    plot_completeness_records,
    resolve_completeness_figure_path,
    evaluate_completeness_subset,
    run_completeness_evaluation,
    run_minimality_pruning,
)

from exp2.eval_runner import (
    resolve_model_results_dir,
    model_name_for_folder,
    write_json,
    evaluate_one_model,
)

from exp2.eval_cli import (
    parse_swap_pair,
    build_argparser,
    main,
)


if __name__ == "__main__":
    main()
