#!/nethome/soyoung/miniconda3/envs/py39/bin/python

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


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

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
os.chdir(SCRIPT_DIR)

import exp2.pp_utils  # noqa: E402
from datasets import datascience  # noqa: E402
from datasets.api import load_vocab  # noqa: E402
from datasets.box import Statement  # noqa: E402
from exp2.pp_utils import get_model_and_tokenizer  # noqa: E402
from exp3.head_role import (  # noqa: E402
    PROMPT_FORMATS,
    DictTensorDataset,
    decode_prompts_from_tokens,
    encode_prompt_texts,
    first_answer_token_id,
    format_prompt_for_model,
    last_token_indices,
    move_batch_to_device,
    resolve_model_name,
    resolve_prompt_format,
    set_seed,
)


NUM_ENTITIES = 3
BASE_OBJECT_IDS_BY_ENTITY = {0: 0, 1: 1, 2: 2}
DEFAULT_COMPONENTS_JSON = (
    SCRIPT_DIR
    / "results/path_patching/gemma9b/circuit_eval_metrics_pruning_heldout2.json"
)
DEFAULT_CONDITIONS = (
    "no_intervention",
    "query_shift",
    "source_swap",
    "both_restore",
    "random_both",
    "wrong_head_both",
    "wrong_position_both",
    "question_query_box_source_shift",
    "q_select_question_query_box_more",
    "q_select_question_query_box_less",
    "k_question_query_box_more",
    "k_question_query_box_less",
    "k_competing_carriers_more",
    "k_competing_carriers_less",
    "k_question_query_box_more_competing_less",
    "k_question_query_box_less_competing_more",
)
CONDITION_CHOICES = (
    "no_intervention",
    "query_shift",
    "source_swap",
    "both_restore",
    "random_query_shift",
    "random_source_swap",
    "random_both",
    "wrong_head_query_shift",
    "wrong_head_source_swap",
    "wrong_head_both",
    "wrong_position_query_shift",
    "wrong_position_source_shift",
    "wrong_position_both",
    "question_query_box_source_shift",
    "q_select_question_query_box_more",
    "q_select_question_query_box_less",
    "k_question_query_box_more",
    "k_question_query_box_less",
    "k_competing_carriers_more",
    "k_competing_carriers_less",
    "k_question_query_box_more_competing_less",
    "k_question_query_box_less_competing_more",
)
CONTROL_CONDITIONS = {
    "random_query_shift",
    "random_source_swap",
    "random_both",
    "wrong_head_query_shift",
    "wrong_head_source_swap",
    "wrong_head_both",
    "wrong_position_query_shift",
    "wrong_position_source_shift",
    "wrong_position_both",
    "question_query_box_source_shift",
}
POSITION_MODE_ALIASES = {"last_token": "answer_colon"}
SOURCE_POSITION_KIND_CHOICES = ("box", "object")

Head = Tuple[int, int]
TensorByHead = Dict[Head, torch.Tensor]

from exp4.bid_data import (
    quiet_build_example_refactored,
    parse_int_tuple,
    parse_csv,
    parse_source_position_kinds,
    parse_conditions,
    normalize_device,
    q_hook_name,
    k_hook_name,
    pattern_hook_name,
    layer_from_hook_name,
    normalize_position_mode,
    group_heads_by_layer,
    key_value_head_for_attention_head,
    aggregate_deltas_by_kv_head,
    context_from_bindings,
    source_slot_for_query,
    query_entity_for_source_slot,
    base_object_words,
    base_box_names,
    object_token_ids_for_prompts,
    entity_context_token_positions_from_formatted,
    query_box_positions_from_formatted,
    build_binding_id_dataset,
    load_group_b_components,
    sanity_table,
    print_sanity_table,
    select_wrong_heads,
)

from exp4.bid_directions import (
    hook_names_for_heads,
    cache_forward,
    batched_position_gather,
    estimate_binding_id_directions,
    delta_between,
    random_matched_deltas,
    matched_norm_directions,
    negate_deltas,
    make_q_patch_hook,
    make_k_patch_hook,
    make_intervention_hooks,
)

from exp4.bid_attention import (
    attention_masses_for_heads,
    carrier_labels,
    carrier_masses_for_heads,
    init_attention_bucket,
    update_attention_bucket,
    finalize_attention_bucket,
    init_attention_stats,
    update_attention_stats,
    finalize_attention_stats,
    init_carrier_bucket,
    update_carrier_bucket,
    finalize_carrier_bucket,
    init_carrier_stats,
    update_carrier_stats,
    finalize_carrier_stats,
    init_logit_stats,
    update_logit_stats,
    finalize_logit_stats,
    mean_mass_by_entity,
    mean_carrier_mass,
    tensor_row_to_floats,
    tensor_carrier_row_to_floats,
)

from exp4.bid_evaluation import (
    evaluate_conditions,
    build_condition_specs,
)

from exp4.bid_cli import (
    serialize_heads,
    serialize_delta_norms,
    serialize_kv_head_mapping,
    parse_args,
)

from exp4.bid_runner import (
    main,
)


if __name__ == "__main__":
    main()
