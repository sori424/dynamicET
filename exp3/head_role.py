#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# bitsandbytes calls torch.compile at import time through TransformerLens.
# This environment has a mismatched Torch Inductor install, so disable Dynamo
# before importing anything that can pull in transformer_lens.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


def _disable_optional_vision_backends() -> None:
    try:
        from transformers.utils import import_utils
    except Exception:
        return
    import_utils._torchvision_available = False
    import_utils._torchvision_version = "N/A"


_disable_optional_vision_backends()

import numpy as np
import torch
import transformer_lens.utils as utils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dynamicet_data import datascience
from dynamicet_data.api import load_vocab
from dynamicet_data.box import Statement
from exp2.pp_utils import compute_topk_components, get_model_and_tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "path_patching"

SITE_CHOICES = (
    "answer_colon",
    "question_query_box",
    "swap_query_box",
    "swap_target_box",
    "context_target_box",
)

STAGE_CHOICES = (
    "value_fetcher",
    "pos_transmitter",
    "pos_detector",
    "struct_reader_swap_target",
    "struct_reader_context_target_box_to_target_box",
)
PAYLOAD_STAGES = {"value_fetcher"}
POINTER_STAGES = set(STAGE_CHOICES) - PAYLOAD_STAGES
ROLE_KIND_BY_STAGE = {
    **{stage: "payload" for stage in PAYLOAD_STAGES},
    **{stage: "pointer" for stage in POINTER_STAGES},
}
PATCH_POINT_INTERPRETATION = {
    "attention_pattern_or_qk": "where the head looks; pointer/address role",
    "o_proj_result": "post-W_O per-head contribution; payload/value role",
    "attention_pattern": "attention distribution; pointer/address role",
}
ROLE_PROMPT_SCORE_MODES = {
    "payload_prompt": "value_fetcher_objects",
    "pointer_prompt": "new_swap_objects",
}
ROLE_DEFAULT_SCORE_SITES = {
    "value_fetcher": ("answer_colon",),
    "pos_transmitter": ("answer_colon",),
    "pos_detector": ("question_query_box",),
    "struct_reader_swap_target": ("swap_target_box",),
    "struct_reader_context_target_box_to_target_box": ("context_target_box",),
}

DEFAULT_SCORE_SITES = (
    "answer_colon",
    "question_query_box",
    "swap_query_box",
    "swap_target_box",
)

POSITION_SITE_ALIASES = {
    "last_token": "answer_colon",
}

MODEL_NAME_ALIASES = {
    "gemma9b": "google/gemma-2-9b-it",
    "gemma2-9b-it": "google/gemma-2-9b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "gemma3-12b-it": "google/gemma-3-12b-it",
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
    "llama-8b": "llama-8b-hf",
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

SYSTEM_PROMPT = "Respond with a single word only. No punctuation, no explanation."
ANSWER_PREFILL_RE = re.compile(r"\s*Answer:\s*$", re.IGNORECASE)

C_TO_A_MILK_BASE_PROMPT = (
    "Box X milk. Box P cup. Box T apple. Swap X and P.\n"
    "Question: Box X Answer:"
)
C_TO_A_MILK_SOURCE_PROMPT = (
    "Box P cup. Box X milk. Box T apple. Swap P and X.\n"
    "Question: Box X Answer:"
)
C_TO_A_MILK_ANSWER = "cup"
C_TO_A_MILK_MEASURED_OBJECT = "milk"
C_TO_A_MILK_CANDIDATES = ("cup", "milk", "apple")

from exp3.head_config import (
    DictTensorDataset,
    parse_swap_pair,
    normalize_device,
    set_seed,
    torch_load_cpu,
    resolve_score_path,
    safe_path_stem,
    default_output_path_for_score,
    select_heads,
    normalize_component_site,
    load_component_dicts,
    filter_components_by_stage,
    role_aligned_hook_name,
    component_hook_name,
    component_count_by_stage,
    component_heads_by_site_hook,
    select_site_heads,
    score_heads_by_site_hook,
    all_hook_names,
    component_count_by_site,
    unique_layer_heads,
    normalize_hook_component,
    hook_name,
    layer_from_hook_name,
)

from exp3.head_data import (
    quiet_build_example_refactored,
    last_token_indices,
    resolve_model_name,
    infer_prompt_format,
    resolve_prompt_format,
    _apply_chat_template,
    _split_assistant_prefill,
    _tokenizer_has_token,
    _format_llama_prefill_fallback,
    _format_llama_prompt_fallback,
    _apply_chat_prefill_template,
    format_prompt_for_model,
    encode_prompt_texts,
    first_answer_token_id,
    answer_words_from_tokens,
    decode_prompts_from_tokens,
    span_starts,
    paired_contexts,
    normal_context_index,
    swap_target_box_from_context,
    swap_ordinal_box_char,
    site_char,
    compute_formatted_site_positions,
    object_words_for_attrs,
    build_paired_dataset,
    c_to_a_milk_site_char,
    compute_c_to_a_milk_site_positions,
    build_c_to_a_milk_dataset,
)

from exp3.head_metrics import (
    move_batch_to_device,
    new_metrics,
    token_stats,
    token_delta,
    logit_margin,
    scalar_delta,
    format_optional_float,
    optional_metric_delta,
    update_metrics,
    finalize_metrics,
)

from exp3.head_interventions import (
    positions_for_sites,
    patch_offsets_for_site,
    patch_selected_attention_patterns,
    patch_selected_heads,
    heads_by_hook_for_sites,
    evaluate_interchange,
)

from exp3.role_scores import (
    ensure_attn_result_enabled,
    evaluation_plan_for_sites,
    evaluate_role_prompt_scores,
    decode_prompt,
    preview_dataset,
)

from exp3.head_cli import (
    build_argparser,
)

from exp3.head_runner import (
    main,
)


if __name__ == "__main__":
    main()
