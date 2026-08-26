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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
os.chdir(SCRIPT_DIR)

import exp2.pp_utils  # noqa: E402
from exp2 import pp_utils  # noqa: E402
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

def quiet_build_example_refactored(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return datascience.build_example_refactored(**kwargs)


def parse_int_tuple(value: str, *, expected_len: Optional[int] = None) -> Tuple[int, ...]:
    pieces = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if expected_len is not None and len(pieces) != expected_len:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_len} comma-separated integers, got {value!r}."
        )
    return pieces


def parse_csv(value: str) -> Tuple[str, ...]:
    return tuple(piece.strip() for piece in value.split(",") if piece.strip())


def parse_source_position_kinds(value: str) -> Tuple[str, ...]:
    kinds = parse_csv(value)
    invalid = [kind for kind in kinds if kind not in SOURCE_POSITION_KIND_CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid source position kind(s): {invalid}. "
            f"Expected any of {SOURCE_POSITION_KIND_CHOICES}."
        )
    if not kinds:
        raise argparse.ArgumentTypeError("At least one source position kind is required.")
    return tuple(dict.fromkeys(kinds))


def parse_conditions(value: str) -> Tuple[str, ...]:
    conditions = parse_csv(value)
    invalid = [condition for condition in conditions if condition not in CONDITION_CHOICES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid condition(s): {invalid}. Expected any of {CONDITION_CHOICES}."
        )
    if not conditions:
        raise argparse.ArgumentTypeError("At least one condition is required.")
    return tuple(dict.fromkeys(conditions))


def normalize_device(device: Optional[str]) -> torch.device:
    if device is None:
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif str(device).isdigit():
        selected = torch.device(f"cuda:{device}")
    else:
        selected = torch.device(device)
    if selected.type == "cuda" and selected.index is not None:
        torch.cuda.set_device(selected)
    pp_utils.set_device(selected)
    return selected


def q_hook_name(layer: int) -> str:
    return f"blocks.{int(layer)}.attn.hook_q"


def k_hook_name(layer: int) -> str:
    return f"blocks.{int(layer)}.attn.hook_k"


def pattern_hook_name(layer: int) -> str:
    return f"blocks.{int(layer)}.attn.hook_pattern"


def layer_from_hook_name(name: str) -> int:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "blocks" and idx + 1 < len(parts):
            return int(parts[idx + 1])
    raise ValueError(f"Could not infer layer index from hook name: {name}")


def normalize_position_mode(position_mode: str) -> str:
    return POSITION_MODE_ALIASES.get(position_mode, position_mode)


def group_heads_by_layer(heads: Sequence[Head]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    return {layer: sorted(set(layer_heads)) for layer, layer_heads in grouped.items()}


def key_value_head_for_attention_head(
    attention_head: int,
    *,
    logical_n_heads: int,
    activation_n_heads: int,
) -> int:
    """Map a query/pattern head index onto the live K/V hook head index.

    Gemma-style grouped-query attention exposes fewer heads at ``hook_k`` than
    at ``hook_q`` / ``hook_pattern``. In that case, each contiguous group of
    query heads shares one K/V head. If the activation already has the logical
    number of heads, this is just the identity map.
    """

    attention_head = int(attention_head)
    logical_n_heads = int(logical_n_heads)
    activation_n_heads = int(activation_n_heads)
    if activation_n_heads <= 0:
        raise ValueError("activation_n_heads must be positive.")
    if logical_n_heads <= activation_n_heads:
        return min(activation_n_heads - 1, attention_head)
    group_size = max(1, logical_n_heads // activation_n_heads)
    return min(activation_n_heads - 1, attention_head // group_size)


def aggregate_deltas_by_kv_head(
    *,
    layer_heads: Sequence[int],
    deltas: Mapping[Head, torch.Tensor],
    layer: int,
    logical_n_heads: int,
    activation_n_heads: int,
) -> Dict[int, torch.Tensor]:
    by_kv: Dict[int, List[torch.Tensor]] = defaultdict(list)
    for head in layer_heads:
        kv_head = key_value_head_for_attention_head(
            int(head),
            logical_n_heads=logical_n_heads,
            activation_n_heads=activation_n_heads,
        )
        by_kv[kv_head].append(deltas[(int(layer), int(head))])
    return {
        kv_head: torch.stack([delta.float() for delta in kv_deltas], dim=0).mean(dim=0)
        for kv_head, kv_deltas in by_kv.items()
    }


def context_from_bindings(
    bindings_by_entity: Mapping[int, int],
    *,
    include_swap: bool,
    swap_pair: Tuple[int, int],
) -> List[Statement]:
    if sorted(bindings_by_entity) != list(range(NUM_ENTITIES)):
        raise ValueError(
            f"Expected bindings for entities 0..{NUM_ENTITIES - 1}, "
            f"got {sorted(bindings_by_entity)}."
        )
    context = [
        Statement(entity, int(bindings_by_entity[entity]), "normal")
        for entity in range(NUM_ENTITIES)
    ]
    if include_swap:
        context.append(Statement(int(swap_pair[0]), int(swap_pair[1]), "swap"))
    return context


def source_slot_for_query(
    query_entity: int,
    *,
    include_swap: bool,
    swap_pair: Tuple[int, int],
) -> int:
    query_entity = int(query_entity)
    if include_swap and query_entity == int(swap_pair[0]):
        return int(swap_pair[1])
    if include_swap and query_entity == int(swap_pair[1]):
        return int(swap_pair[0])
    return query_entity


def query_entity_for_source_slot(
    source_slot: int,
    *,
    include_swap: bool,
    swap_pair: Tuple[int, int],
) -> int:
    source_slot = int(source_slot)
    if include_swap and source_slot == int(swap_pair[0]):
        return int(swap_pair[1])
    if include_swap and source_slot == int(swap_pair[1]):
        return int(swap_pair[0])
    return source_slot


def base_object_words(vocab, prompt_id: int) -> List[str]:
    return [
        vocab.fetch_shuffled_attr(BASE_OBJECT_IDS_BY_ENTITY[entity], prompt_id)[1]
        for entity in range(NUM_ENTITIES)
    ]


def base_box_names(vocab, prompt_id: int) -> List[str]:
    names_perm, _ = vocab.get_shuffled_labels(prompt_id)
    return [
        vocab.filtered_names[names_perm[entity]]
        for entity in range(NUM_ENTITIES)
    ]


def object_token_ids_for_prompts(
    *,
    tokenizer,
    formatted_prompts: Sequence[str],
    vocab,
    prompt_id_start: int,
) -> torch.Tensor:
    rows = []
    for row_idx, prompt in enumerate(formatted_prompts):
        prompt_id = prompt_id_start + row_idx
        rows.append(
            [
                first_answer_token_id(tokenizer, prompt, object_word)
                for object_word in base_object_words(vocab, prompt_id)
            ]
        )
    return torch.tensor(rows, dtype=torch.long)


def entity_context_token_positions_from_formatted(
    *,
    tokenizer,
    raw_prompts: Sequence[str],
    formatted_prompts: Sequence[str],
    tokens: torch.Tensor,
    vocab,
    prompt_id_start: int,
    token_kind: str,
) -> torch.Tensor:
    if token_kind not in SOURCE_POSITION_KIND_CHOICES:
        raise ValueError(f"Unknown token kind: {token_kind!r}.")
    rows = []
    for row_idx, (raw_prompt, formatted_prompt, token_row) in enumerate(
        zip(raw_prompts, formatted_prompts, tokens)
    ):
        row = []
        prompt_id = prompt_id_start + row_idx
        for box_name, object_word in zip(
            base_box_names(vocab, prompt_id),
            base_object_words(vocab, prompt_id),
        ):
            statement = f"Box {box_name} contains the {object_word}"
            statement_start = formatted_prompt.find(statement)
            if statement_start < 0:
                raise ValueError(
                    f"Could not find context statement {statement!r} in the "
                    f"formatted prompt. Raw prompt was:\n{raw_prompt}\n\n"
                    f"Formatted prompt was:\n{formatted_prompt}"
                )
            if token_kind == "box":
                token_start = statement_start + len("Box ")
            else:
                token_start = statement_start + len(f"Box {box_name} contains the ")
            row.append(
                pp_utils._char_to_token_index(
                    tokenizer,
                    formatted_prompt,
                    token_start,
                    token_ids=token_row.tolist(),
                )
            )
        rows.append(row)
    return torch.tensor(rows, dtype=torch.long)


def query_box_positions_from_formatted(
    *,
    tokenizer,
    raw_prompts: Sequence[str],
    formatted_prompts: Sequence[str],
    tokens: torch.Tensor,
) -> torch.Tensor:
    values = []
    for raw_prompt, formatted_prompt, token_row in zip(
        raw_prompts,
        formatted_prompts,
        tokens,
    ):
        try:
            char_idx = pp_utils._question_query_box_char(formatted_prompt)
        except Exception as exc:
            raise ValueError(
                "Could not find the question/query box in the formatted prompt. "
                f"Raw prompt was:\n{raw_prompt}\n\nFormatted prompt was:\n{formatted_prompt}"
            ) from exc
        values.append(
            pp_utils._char_to_token_index(
                tokenizer,
                formatted_prompt,
                char_idx,
                token_ids=token_row.tolist(),
            )
        )
    return torch.tensor(values, dtype=torch.long)


def build_binding_id_dataset(
    *,
    tokenizer,
    model_name: str,
    vocab,
    query_entity: int,
    num_samples: int,
    prompt_id_start: int,
    prompt_format: str,
    include_swap: bool,
    swap_pair: Tuple[int, int],
    template_type: str,
) -> DictTensorDataset:
    base_context = context_from_bindings(
        BASE_OBJECT_IDS_BY_ENTITY,
        include_swap=include_swap,
        swap_pair=swap_pair,
    )
    template_context = {"query_name": int(query_entity), "raw_query_name": None}

    base_tokens, _, _, _ = quiet_build_example_refactored(
        batch_size=num_samples,
        vocab=vocab,
        num_entities=NUM_ENTITIES,
        context=base_context,
        prompt_id_start=prompt_id_start,
        template_context=template_context,
        template_type=template_type,
    )

    resolved_prompt_format = resolve_prompt_format(model_name, prompt_format)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    base_prompts = decode_prompts_from_tokens(tokenizer, base_tokens)

    if resolved_prompt_format == "raw":
        formatted_base_prompts = base_prompts
        formatted_base_tokens = base_tokens.to(torch.long)
    else:
        formatted_base_prompts = [
            format_prompt_for_model(tokenizer, prompt, resolved_prompt_format)
            for prompt in base_prompts
        ]
        formatted_base_tokens = encode_prompt_texts(tokenizer, formatted_base_prompts)
    base_last = last_token_indices(formatted_base_tokens, pad_token_id)
    source_entity = source_slot_for_query(
        query_entity,
        include_swap=include_swap,
        swap_pair=swap_pair,
    )

    data = {
        "base_tokens": formatted_base_tokens.to(torch.long),
        "object_token_ids": object_token_ids_for_prompts(
            tokenizer=tokenizer,
            formatted_prompts=formatted_base_prompts,
            vocab=vocab,
            prompt_id_start=prompt_id_start,
        ),
        "box_token_positions": entity_context_token_positions_from_formatted(
            tokenizer=tokenizer,
            raw_prompts=base_prompts,
            formatted_prompts=formatted_base_prompts,
            tokens=formatted_base_tokens,
            vocab=vocab,
            prompt_id_start=prompt_id_start,
            token_kind="box",
        ),
        "object_token_positions": entity_context_token_positions_from_formatted(
            tokenizer=tokenizer,
            raw_prompts=base_prompts,
            formatted_prompts=formatted_base_prompts,
            tokens=formatted_base_tokens,
            vocab=vocab,
            prompt_id_start=prompt_id_start,
            token_kind="object",
        ),
        "query_box_positions": query_box_positions_from_formatted(
            tokenizer=tokenizer,
            raw_prompts=base_prompts,
            formatted_prompts=formatted_base_prompts,
            tokens=formatted_base_tokens,
        ),
        "prompt_ids": torch.arange(
            prompt_id_start,
            prompt_id_start + num_samples,
            dtype=torch.long,
        ),
        "query_entities": torch.full((num_samples,), int(query_entity), dtype=torch.long),
        "source_entities": torch.full((num_samples,), int(source_entity), dtype=torch.long),
        "base_last_token_indices": base_last,
        "base_answer_colon_positions": base_last,
    }
    return DictTensorDataset(data)


def load_group_b_components(
    path: Path,
    *,
    expected_stage: str,
    expected_position_modes: Sequence[str],
) -> Tuple[
    List[Dict[str, object]],
    List[Head],
    Dict[str, object],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kept_components = payload.get("minimality_pruning", {}).get("kept_components")
    if kept_components is None:
        raise ValueError(f"{path} has no minimality_pruning.kept_components.")
    if not isinstance(kept_components, list):
        raise ValueError(
            f"Expected minimality_pruning.kept_components to be a list, "
            f"got {type(kept_components)!r}."
        )

    expected_positions = {
        normalize_position_mode(position_mode)
        for position_mode in expected_position_modes
    }
    selected = []
    nonmatching = []
    for component in kept_components:
        component_dict = dict(component)
        stage = str(component_dict.get("stage"))
        position = normalize_position_mode(str(component_dict.get("position_mode")))
        if stage == expected_stage and position in expected_positions:
            selected.append(component_dict)
        else:
            nonmatching.append(component_dict)

    if not selected:
        stage_counts: Dict[str, int] = defaultdict(int)
        for component in kept_components:
            stage_counts[str(component.get("stage"))] += 1
        raise ValueError(
            "No group-B components matched "
            f"stage={expected_stage!r}, positions={sorted(expected_positions)}. "
            f"Kept component stage counts: {dict(stage_counts)}"
        )

    seen = set()
    heads: List[Head] = []
    for component in selected:
        head = (int(component["layer"]), int(component["head"]))
        if head in seen:
            continue
        seen.add(head)
        heads.append(head)
    heads.sort()
    return selected, heads, payload, [dict(c) for c in kept_components], nonmatching


def sanity_table(components: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "stage": str(component.get("stage")),
            "position_mode": str(component.get("position_mode")),
            "hook_name": str(component.get("hook_name")),
            "layer": int(component["layer"]),
            "head": int(component["head"]),
        }
        for component in components
    ]


def print_sanity_table(rows: Sequence[Mapping[str, object]]) -> None:
    print(
        "Selected group-B sanity table "
        "(stage, position_mode, recorded_hook_name, layer, head):",
        file=sys.stderr,
    )
    for row in rows:
        print(
            f"  {row['stage']}\t{row['position_mode']}\t{row['hook_name']}"
            f"\tL{row['layer']}H{row['head']}",
            file=sys.stderr,
        )


def select_wrong_heads(model, b_heads: Sequence[Head]) -> List[Head]:
    b_set = set((int(layer), int(head)) for layer, head in b_heads)
    logical_n_heads = int(getattr(model.cfg, "n_heads", 0))
    if logical_n_heads <= 0:
        logical_n_heads = max(int(head) for _, head in b_heads) + 1
    kv_n_heads = int(
        getattr(
            model.cfg,
            "n_key_value_heads",
            getattr(model.cfg, "n_kv_heads", logical_n_heads),
        )
    )
    b_kv_set = {
        (
            int(layer),
            key_value_head_for_attention_head(
                int(head),
                logical_n_heads=logical_n_heads,
                activation_n_heads=kv_n_heads,
            ),
        )
        for layer, head in b_heads
    }

    def is_good_wrong_candidate(candidate: Head) -> bool:
        if candidate in b_set or candidate in wrong:
            return False
        layer, head = candidate
        candidate_kv = key_value_head_for_attention_head(
            int(head),
            logical_n_heads=logical_n_heads,
            activation_n_heads=kv_n_heads,
        )
        return (int(layer), candidate_kv) not in b_kv_set

    wrong: List[Head] = []
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    for layer, head in b_heads:
        for offset in range(1, n_heads + 1):
            candidate = (int(layer), (int(head) + offset) % n_heads)
            if is_good_wrong_candidate(candidate):
                wrong.append(candidate)
                break
    if len(wrong) < len(b_heads):
        for layer in range(n_layers):
            for head in range(n_heads):
                candidate = (layer, head)
                if not is_good_wrong_candidate(candidate):
                    continue
                wrong.append(candidate)
                if len(wrong) == len(b_heads):
                    return wrong
    return wrong
