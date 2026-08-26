import json
import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# bitsandbytes calls torch.compile at import time through TransformerLens.
# This environment has a mismatched Torch Inductor install, so disable Dynamo
# before importing anything that can pull in transformer_lens.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


@contextlib.contextmanager
def _hide_torchvision_imports():
    original_find_spec = importlib.util.find_spec

    def find_spec_without_torchvision(name, package=None):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = find_spec_without_torchvision
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.common import lookup_answer_logits


TRACE_CATEGORY_ALIASES = {
    "character": "names",
    "characters": "names",
    "context": "names",
    "name": "names",
    "names": "names",
    "object": "countries",
    "objects": "countries",
    "country": "countries",
    "countries": "countries",
    "query": "question_query_box",
    "question": "question_query_box",
    "question_query_box": "question_query_box",
    "swap": "swap_boxes",
    "swap_box": "swap_boxes",
    "swap_boxes": "swap_boxes",
    "swap_sentence": "swap_boxes",
    "state": "swap_boxes",
}


VALID_NAMES_TRACE_MODES = {
    "context_only",
    "context_target_box",
    "context_target_box_and_swap_query_box",
    "context_and_swap",
    "context_and_swap_target_box",
    "swap_only",
    "swap_target_box",
    "swap_target_box_to_remaining_context_box",
    "context_target_object_and_swap_target_box_to_remaining_context_box",
}
VALID_OBJECT_TRACE_MODES = {
    "target_box",
    "query_box",
    "context_target_object_and_swap_query_box",
    "swap_target_object_with_remaining_context_object",
}
QUERY_TRACE_MODE_ALIASES = {
    "default": "same_query",
    "same": "same_query",
    "same_query": "same_query",
    "swap_partner": "swap_partner",
    "partner": "swap_partner",
    "swap_partner_box": "swap_partner",
}
VALID_QUERY_TRACE_MODES = set(QUERY_TRACE_MODE_ALIASES.values())

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp1.trace_contexts import (
    _get_act_name,
    _default_binding_selector_extractor,
    _stack_tokens,
    _model_device,
    _move_batch_to_device,
    load_model,
    normalize_trace_category,
    resolve_trace_modes,
    resolve_query_trace_mode,
    _validate_entity_index,
    resolve_swap_partner,
    resolve_source_swap_pair,
    resolve_swap_partner_from_pair,
    resolve_remaining_context_box,
    resolve_prompt_query_names,
    default_template_context,
    _replace_swap_box_mention,
    build_trace_contexts,
)

from exp1.trace_data import (
    _decode_prompt,
    _selector_positions,
    summarize_selector_positions,
    _build_prompt_batch,
    TracePromptDataset,
    build_tracing_dataset,
)

from exp1.trace_runner import (
    _candidate_prediction_summary,
    find_correct_samples,
    _patch_resid_pre_at_token,
    _init_nested_scores,
    _build_result_payload,
    save_tracing_results,
    run_tracing_experiment,
)
