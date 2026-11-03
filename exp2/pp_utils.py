import csv
import json
import math
import os
import random
import re
import string
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import transformer_lens.utils as utils
from datasets import Dataset
# from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

curr_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(curr_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from coref.models import fetch_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_ALIASES = {
    "llama": "llama-7b-hf",
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
    "google/gemma-2-9b-it": "gemma2-9b-it",
}

CUSTOM_LLAMA_FAMILY_MODELS = {
    "vicuna": "AlekseyKorshuk/vicuna-7b",
    "float": "nikhil07prakash/float-7b",
}

ASSIGNMENT_RE = re.compile(
    r"Box\s+([A-Z])\s+contains\s+the\s+([A-Za-z][\w-]*)\.",
    re.IGNORECASE,
)
SWAP_BOXES_RE = re.compile(
    r"Swap\s+the\s+items\s+of\s+Box\s+([A-Z])\s+and\s+Box\s+([A-Z])\.",
    re.IGNORECASE,
)
QUERY_SENTENCE_RE = re.compile(
    r"(?:Answer:\s*)?Box\s+([A-Z])\s+contains\s+the"
    r"(?=\s*(?:$|<\|im_end\|>|<end_of_turn>|<\|eot_id\|>|assistant\b|model\b))",
    re.IGNORECASE,
)
QUESTION_SENTENCE_RE = re.compile(
    r"Question:\s+Which\s+item\s+does\s+Box\s+([A-Z])\s+contain\?",
    re.IGNORECASE,
)

BOX_NAME_POOL = tuple(string.ascii_uppercase)
_DEFAULT_OBJECT_POOL: Optional[Tuple[str, ...]] = None

PROMPT_POSITION_MODES = (
    "question_query_box",
    "answer_query_box",
    "swap_query_box",
    "swap_target_box",
    "context_query_box",
    "context_target_box",
    "context_source_box",
    "source_object",
    "context_source_object",
    "context_query_object",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import exp2.pp_prompts as _pp_prompts
import exp2.pp_corruptions as _pp_corruptions
import exp2.pp_models as _pp_models
import exp2.pp_patching as _pp_patching

from exp2.pp_prompts import (
    _model_device,
    _model_num_layers,
    _model_num_heads,
    _ensure_pad_token,
    _z_hook_name,
    _o_proj_hook_name,
    _attn_hook_name,
    _receiver_key,
    _layer_from_hook_name,
    _move_batch_to_device,
    _read_jsonl_rows,
    _first_present,
    _clean_prompt,
    _corrupt_prompt,
    _clean_answer,
    _encode_prompts,
    _encode_single_token_answer,
    _as_long_tensor,
    _slice_tensor_rows,
    _compute_last_token_indices_from_tokens,
    _squeeze_label_tensor,
    _decode_prompt_from_tokens,
    _parse_swap_prompt,
    _query_source_box,
    _query_source_object,
    _query_box_object,
    _char_to_token_index,
    _tokenize_ids,
    _find_subsequence_starts,
    _query_box_token_ids,
    _matched_slice_with_boundary,
    _resolve_sentence_match,
    _find_query_box_token_position,
    _question_query_box_char,
    _answer_query_box_char,
    _swap_query_box_char,
    _swap_target_box_char,
    _context_query_box_char,
    _context_source_box_char,
    _source_object_char,
    _trim_trailing_pad_tokens,
    _find_context_object_token_position,
    _compute_prompt_position_fields,
)

from exp2.pp_corruptions import (
    _replace_assignment_object,
    _default_object_pool,
    _rerandomize_swap_prompt,
    _select_donor_source_object,
    _generate_corrupt_prompts_from_anchor_prompts,
    _normalize_pretokenized_mapping,
    _build_dataset_from_rows,
    _build_dataset_from_pretokenized,
    _build_dataset_from_clean_only_pretokenized,
    _coerce_in_memory_dataset,
)

from exp2.pp_models import (
    _load_custom_llama_family_model,
    _load_goat_model,
    compute_topk_components,
    compute_prev_query_box_pos,
    get_model_and_tokenizer,
)

from exp2.pp_patching import (
    load_dataloader,
    get_caches,
    patching_sender_heads,
    patching_receiver_heads,
    score_path_patching_heads,
    get_receiver_layers,
    load_eval_data,
    load_ablation_data,
    get_mean_activations,
    mean_ablate,
    eval_circuit_performance,
    get_circuit,
    get_random_circuit,
    compute_pair_drop_values,
    get_head_significance_score,
    get_final_circuit,
)

def set_device(value):
    """Set the shared execution device used by model-loading helpers."""
    global device
    device = value
    for module in (_pp_prompts, _pp_corruptions, _pp_models, _pp_patching):
        module.device = value
    return value
