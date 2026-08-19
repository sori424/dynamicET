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
    exp2.pp_utils.device = selected
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


def hook_names_for_heads(heads: Sequence[Head], hook_fn) -> List[str]:
    return sorted({hook_fn(layer) for layer, _ in heads})


def cache_forward(
    model,
    tokens: torch.Tensor,
    *,
    cache_hook_names: Sequence[str],
    fwd_hooks: Optional[Sequence[Tuple[str, object]]] = None,
):
    hook_name_set = set(cache_hook_names)
    cache, cache_hooks, _ = model.get_caching_hooks(
        names_filter=lambda name: name in hook_name_set,
        incl_bwd=False,
        device=None,
        remove_batch_dim=False,
    )
    hooks = list(fwd_hooks or []) + list(cache_hooks)
    with model.hooks(fwd_hooks=hooks):
        logits = model(tokens, return_type="logits")
    return logits, cache


def batched_position_gather(
    activation: torch.Tensor,
    positions: torch.Tensor,
    head: int,
) -> torch.Tensor:
    batch_indices = torch.arange(activation.shape[0], device=activation.device)
    return activation[
        batch_indices,
        positions.to(activation.device).to(torch.long),
        int(head),
        :,
    ]


def estimate_binding_id_directions(
    *,
    model,
    dataloaders_by_source_slot: Mapping[int, DataLoader],
    heads: Sequence[Head],
    source_position_kinds: Sequence[str],
    device: torch.device,
    label: str,
    k_position_fields: Sequence[str] = ("query_box_positions",),
) -> Dict[str, object]:
    heads_by_layer = group_heads_by_layer(heads)
    logical_n_heads = int(getattr(model.cfg, "n_heads", 0)) or max(
        int(head) for _, head in heads
    ) + 1
    cache_hook_names = sorted(
        set(hook_names_for_heads(heads, q_hook_name))
        | set(hook_names_for_heads(heads, k_hook_name))
    )
    q_means: Dict[int, TensorByHead] = {}
    k_means: Dict[int, TensorByHead] = {}
    q_counts: Dict[int, Dict[str, int]] = {}
    k_counts: Dict[int, Dict[str, int]] = {}
    k_position_means: Dict[str, Dict[int, TensorByHead]] = {
        field: {} for field in k_position_fields
    }
    k_position_counts: Dict[str, Dict[int, Dict[str, int]]] = {
        field: {} for field in k_position_fields
    }

    model.eval()
    with torch.no_grad():
        for source_slot, dataloader in dataloaders_by_source_slot.items():
            q_sums: TensorByHead = {}
            k_sums: TensorByHead = {}
            q_count_by_head: Dict[Head, int] = defaultdict(int)
            k_count_by_head: Dict[Head, int] = defaultdict(int)
            k_position_sums: Dict[str, TensorByHead] = {
                field: {} for field in k_position_fields
            }
            k_position_count_by_head: Dict[str, Dict[Head, int]] = {
                field: defaultdict(int) for field in k_position_fields
            }
            for batch in tqdm(
                dataloader,
                desc=f"Estimate {label} Q/K ID slot={source_slot}",
            ):
                batch = move_batch_to_device(batch, device)
                _, cache = cache_forward(
                    model,
                    batch["base_tokens"],
                    cache_hook_names=cache_hook_names,
                )
                for layer, layer_heads in heads_by_layer.items():
                    q = cache[q_hook_name(layer)]
                    k = cache[k_hook_name(layer)]
                    answer_positions = batch["base_answer_colon_positions"]
                    for head in layer_heads:
                        key = (layer, head)
                        q_values = batched_position_gather(q, answer_positions, head)
                        q_sums[key] = q_sums.get(
                            key,
                            torch.zeros(q_values.shape[-1], dtype=torch.float32),
                        ) + q_values.detach().float().sum(dim=0).cpu()
                        q_count_by_head[key] += int(q_values.shape[0])

                        k_head = key_value_head_for_attention_head(
                            head,
                            logical_n_heads=logical_n_heads,
                            activation_n_heads=int(k.shape[2]),
                        )
                        for kind in source_position_kinds:
                            positions = batch[f"{kind}_token_positions"][
                                :,
                                int(source_slot),
                            ]
                            k_values = batched_position_gather(k, positions, k_head)
                            k_sums[key] = k_sums.get(
                                key,
                                torch.zeros(k_values.shape[-1], dtype=torch.float32),
                            ) + k_values.detach().float().sum(dim=0).cpu()
                            k_count_by_head[key] += int(k_values.shape[0])
                        for position_field in k_position_fields:
                            positions = batch[position_field]
                            k_values = batched_position_gather(k, positions, k_head)
                            k_position_sums[position_field][key] = (
                                k_position_sums[position_field].get(
                                    key,
                                    torch.zeros(
                                        k_values.shape[-1],
                                        dtype=torch.float32,
                                    ),
                                )
                                + k_values.detach().float().sum(dim=0).cpu()
                            )
                            k_position_count_by_head[position_field][key] += int(
                                k_values.shape[0]
                            )
                del cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            q_means[int(source_slot)] = {
                key: q_sums[key] / max(q_count_by_head[key], 1)
                for key in heads
            }
            k_means[int(source_slot)] = {
                key: k_sums[key] / max(k_count_by_head[key], 1)
                for key in heads
            }
            q_counts[int(source_slot)] = {
                f"L{layer}H{head}": int(q_count_by_head[(layer, head)])
                for layer, head in heads
            }
            k_counts[int(source_slot)] = {
                f"L{layer}H{head}": int(k_count_by_head[(layer, head)])
                for layer, head in heads
            }
            for position_field in k_position_fields:
                k_position_means[position_field][int(source_slot)] = {
                    key: (
                        k_position_sums[position_field][key]
                        / max(k_position_count_by_head[position_field][key], 1)
                    )
                    for key in heads
                }
                k_position_counts[position_field][int(source_slot)] = {
                    f"L{layer}H{head}": int(
                        k_position_count_by_head[position_field][(layer, head)]
                    )
                    for layer, head in heads
                }

    return {
        "q_means": q_means,
        "k_means": k_means,
        "q_counts": q_counts,
        "k_counts": k_counts,
        "k_position_means": k_position_means,
        "k_position_counts": k_position_counts,
    }


def delta_between(means: Mapping[int, TensorByHead], from_slot: int, to_slot: int) -> TensorByHead:
    return {
        head: means[int(to_slot)][head] - means[int(from_slot)][head]
        for head in means[int(from_slot)]
    }


def random_matched_deltas(
    reference: Mapping[Head, torch.Tensor],
    *,
    seed: int,
) -> TensorByHead:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_deltas: TensorByHead = {}
    for head, ref in sorted(reference.items()):
        ref_float = ref.detach().float().cpu()
        random_vec = torch.randn(
            ref_float.shape,
            generator=generator,
            dtype=torch.float32,
        )
        ref_norm = torch.linalg.vector_norm(ref_float)
        random_norm = torch.linalg.vector_norm(random_vec).clamp_min(1e-12)
        random_deltas[head] = random_vec * (ref_norm / random_norm)
    return random_deltas


def matched_norm_directions(
    directions: Mapping[Head, torch.Tensor],
    *,
    reference: Mapping[Head, torch.Tensor],
) -> TensorByHead:
    matched: TensorByHead = {}
    for head, direction in sorted(directions.items()):
        direction_float = direction.detach().float().cpu()
        ref_float = reference[head].detach().float().cpu()
        direction_norm = torch.linalg.vector_norm(direction_float).clamp_min(1e-12)
        ref_norm = torch.linalg.vector_norm(ref_float)
        matched[head] = direction_float * (ref_norm / direction_norm)
    return matched


def negate_deltas(deltas: Mapping[Head, torch.Tensor]) -> TensorByHead:
    return {head: -delta.detach().float().cpu() for head, delta in deltas.items()}


def make_q_patch_hook(
    *,
    heads_by_layer: Mapping[int, Sequence[int]],
    deltas: Mapping[Head, torch.Tensor],
    alpha: float,
    position_field: str,
    batch: Mapping[str, torch.Tensor],
):
    def patch_q(q: torch.Tensor, hook):
        layer = layer_from_hook_name(hook.name)
        layer_heads = heads_by_layer.get(layer, ())
        if not layer_heads:
            return q
        positions = batch[position_field].to(q.device).to(torch.long)
        batch_indices = torch.arange(q.shape[0], device=q.device)
        for head in layer_heads:
            delta = deltas[(layer, int(head))].to(q.device, dtype=q.dtype)
            q[batch_indices, positions, int(head), :] += float(alpha) * delta
        return q

    return patch_q


def make_k_patch_hook(
    *,
    heads_by_layer: Mapping[int, Sequence[int]],
    entity_deltas: Optional[Mapping[int, Mapping[Head, torch.Tensor]]],
    position_field_deltas: Optional[Mapping[str, Mapping[Head, torch.Tensor]]],
    alpha: float,
    source_position_kinds: Sequence[str],
    batch: Mapping[str, torch.Tensor],
    logical_n_heads: int,
):
    def patch_k(k: torch.Tensor, hook):
        layer = layer_from_hook_name(hook.name)
        layer_heads = heads_by_layer.get(layer, ())
        if not layer_heads:
            return k
        batch_indices = torch.arange(k.shape[0], device=k.device)

        def add_delta_at_positions(
            positions: torch.Tensor,
            *,
            kv_head: int,
            delta: torch.Tensor,
        ) -> None:
            positions = positions.to(k.device).to(torch.long)
            delta = delta.to(k.device, dtype=k.dtype)
            if positions.ndim == 1:
                k[batch_indices, positions, int(kv_head), :] += float(alpha) * delta
                return
            expanded_batch_indices = batch_indices.view(
                (k.shape[0],) + (1,) * (positions.ndim - 1)
            ).expand_as(positions)
            k[
                expanded_batch_indices.reshape(-1),
                positions.reshape(-1),
                int(kv_head),
                :,
            ] += float(alpha) * delta

        for entity, deltas in (entity_deltas or {}).items():
            for kind in source_position_kinds:
                positions = batch[f"{kind}_token_positions"][
                    :,
                    int(entity),
                ]
                kv_deltas = aggregate_deltas_by_kv_head(
                    layer_heads=layer_heads,
                    deltas=deltas,
                    layer=layer,
                    logical_n_heads=logical_n_heads,
                    activation_n_heads=int(k.shape[2]),
                )
                for kv_head, delta in kv_deltas.items():
                    add_delta_at_positions(positions, kv_head=kv_head, delta=delta)
        for position_field, deltas in (position_field_deltas or {}).items():
            positions = batch[position_field]
            kv_deltas = aggregate_deltas_by_kv_head(
                layer_heads=layer_heads,
                deltas=deltas,
                layer=layer,
                logical_n_heads=logical_n_heads,
                activation_n_heads=int(k.shape[2]),
            )
            for kv_head, delta in kv_deltas.items():
                add_delta_at_positions(positions, kv_head=kv_head, delta=delta)
        return k

    return patch_k


def make_intervention_hooks(
    *,
    heads: Sequence[Head],
    q_deltas: Optional[Mapping[Head, torch.Tensor]],
    q_position_field: Optional[str],
    k_entity_deltas: Optional[Mapping[int, Mapping[Head, torch.Tensor]]],
    k_position_field_deltas: Optional[Mapping[str, Mapping[Head, torch.Tensor]]],
    source_position_kinds: Sequence[str],
    alpha: float,
    batch: Mapping[str, torch.Tensor],
    logical_n_heads: int,
) -> List[Tuple[str, object]]:
    hooks: List[Tuple[str, object]] = []
    heads_by_layer = group_heads_by_layer(heads)
    if q_deltas is not None:
        if q_position_field is None:
            raise ValueError("q_position_field is required when q_deltas are provided.")
        q_hook = make_q_patch_hook(
            heads_by_layer=heads_by_layer,
            deltas=q_deltas,
            alpha=alpha,
            position_field=q_position_field,
            batch=batch,
        )
        hooks.extend((q_hook_name(layer), q_hook) for layer in sorted(heads_by_layer))
    if k_entity_deltas is not None or k_position_field_deltas is not None:
        k_hook = make_k_patch_hook(
            heads_by_layer=heads_by_layer,
            entity_deltas=k_entity_deltas,
            position_field_deltas=k_position_field_deltas,
            alpha=alpha,
            source_position_kinds=source_position_kinds,
            batch=batch,
            logical_n_heads=logical_n_heads,
        )
        hooks.extend((k_hook_name(layer), k_hook) for layer in sorted(heads_by_layer))
    return hooks


def attention_masses_for_heads(
    *,
    cache: Mapping[str, torch.Tensor],
    heads: Sequence[Head],
    batch: Mapping[str, torch.Tensor],
    source_position_kinds: Sequence[str],
) -> Dict[Head, torch.Tensor]:
    masses_by_head: Dict[Head, torch.Tensor] = {}
    for layer, head in heads:
        pattern = cache[pattern_hook_name(layer)]
        batch_indices = torch.arange(pattern.shape[0], device=pattern.device)
        dest_positions = batch["base_answer_colon_positions"].to(pattern.device).to(torch.long)
        entity_masses = []
        for entity in range(NUM_ENTITIES):
            mass = torch.zeros(pattern.shape[0], device=pattern.device, dtype=torch.float32)
            for kind in source_position_kinds:
                src_positions = batch[f"{kind}_token_positions"][
                    :,
                    entity,
                ].to(pattern.device).to(torch.long)
                mass = mass + pattern[
                    batch_indices,
                    int(head),
                    dest_positions,
                    src_positions,
                ].detach().float()
            entity_masses.append(mass)
        masses_by_head[(int(layer), int(head))] = torch.stack(entity_masses, dim=1)
    return masses_by_head


def carrier_labels() -> List[str]:
    return ["question_query_box"] + [
        f"context_box_{entity}" for entity in range(NUM_ENTITIES)
    ]


def carrier_masses_for_heads(
    *,
    cache: Mapping[str, torch.Tensor],
    heads: Sequence[Head],
    batch: Mapping[str, torch.Tensor],
) -> Dict[Head, torch.Tensor]:
    masses_by_head: Dict[Head, torch.Tensor] = {}
    for layer, head in heads:
        pattern = cache[pattern_hook_name(layer)]
        batch_indices = torch.arange(pattern.shape[0], device=pattern.device)
        dest_positions = batch["base_answer_colon_positions"].to(pattern.device).to(torch.long)
        query_box_positions = batch["query_box_positions"].to(pattern.device).to(torch.long)
        query_box_mass = pattern[
            batch_indices,
            int(head),
            dest_positions,
            query_box_positions,
        ].detach().float()
        carrier_masses = [query_box_mass]
        for entity in range(NUM_ENTITIES):
            context_positions = batch["box_token_positions"][
                :,
                entity,
            ].to(pattern.device).to(torch.long)
            carrier_masses.append(
                pattern[
                    batch_indices,
                    int(head),
                    dest_positions,
                    context_positions,
                ].detach().float()
            )
        masses_by_head[(int(layer), int(head))] = torch.stack(carrier_masses, dim=1)
    return masses_by_head


def init_attention_bucket() -> Dict[str, object]:
    return {
        "before_mass_sum": [0.0 for _ in range(NUM_ENTITIES)],
        "after_mass_sum": [0.0 for _ in range(NUM_ENTITIES)],
        "count": 0,
        "log_ratio_before_sum": 0.0,
        "log_ratio_after_sum": 0.0,
        "switch_original_to_target": 0,
        "top_after_target": 0,
    }


def update_attention_bucket(
    bucket: Dict[str, object],
    *,
    before: torch.Tensor,
    after: torch.Tensor,
    original_entity: int,
    target_entity: int,
    eps: float,
) -> None:
    before_cpu = before.detach().float().cpu()
    after_cpu = after.detach().float().cpu()
    for entity in range(NUM_ENTITIES):
        bucket["before_mass_sum"][entity] += float(before_cpu[:, entity].sum().item())
        bucket["after_mass_sum"][entity] += float(after_cpu[:, entity].sum().item())
    bucket["count"] += int(before_cpu.shape[0])
    before_log_ratio = torch.log(before_cpu[:, target_entity] + eps) - torch.log(
        before_cpu[:, original_entity] + eps
    )
    after_log_ratio = torch.log(after_cpu[:, target_entity] + eps) - torch.log(
        after_cpu[:, original_entity] + eps
    )
    bucket["log_ratio_before_sum"] += float(before_log_ratio.sum().item())
    bucket["log_ratio_after_sum"] += float(after_log_ratio.sum().item())
    before_top = torch.argmax(before_cpu, dim=1)
    after_top = torch.argmax(after_cpu, dim=1)
    bucket["switch_original_to_target"] += int(
        ((before_top == original_entity) & (after_top == target_entity)).sum().item()
    )
    bucket["top_after_target"] += int((after_top == target_entity).sum().item())


def finalize_attention_bucket(
    bucket: Mapping[str, object],
    *,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    count = int(bucket["count"])
    if count == 0:
        return {
            "count": 0,
            "mean_before_mass_by_entity": None,
            "mean_after_mass_by_entity": None,
        }
    before = [value / count for value in bucket["before_mass_sum"]]
    after = [value / count for value in bucket["after_mass_sum"]]
    before_tmo = before[target_entity] - before[original_entity]
    after_tmo = after[target_entity] - after[original_entity]
    before_lr = float(bucket["log_ratio_before_sum"]) / count
    after_lr = float(bucket["log_ratio_after_sum"]) / count
    return {
        "count": count,
        "mean_before_mass_by_entity": {
            str(entity): before[entity] for entity in range(NUM_ENTITIES)
        },
        "mean_after_mass_by_entity": {
            str(entity): after[entity] for entity in range(NUM_ENTITIES)
        },
        "target_minus_original_attention_mass": {
            "before": before_tmo,
            "after": after_tmo,
            "delta_after_minus_before": after_tmo - before_tmo,
        },
        "log_ratio_target_over_original": {
            "before": before_lr,
            "after": after_lr,
            "delta_after_minus_before": after_lr - before_lr,
        },
        "switch_original_to_target_fraction": (
            int(bucket["switch_original_to_target"]) / count
        ),
        "top_after_target_fraction": int(bucket["top_after_target"]) / count,
    }


def init_attention_stats(heads: Sequence[Head]) -> Dict[str, object]:
    return {
        "overall": init_attention_bucket(),
        "by_head": {f"L{layer}H{head}": init_attention_bucket() for layer, head in heads},
    }


def update_attention_stats(
    stats: Dict[str, object],
    *,
    before_masses: Mapping[Head, torch.Tensor],
    after_masses: Mapping[Head, torch.Tensor],
    original_entity: int,
    target_entity: int,
    eps: float,
) -> None:
    for head, before in before_masses.items():
        after = after_masses[head]
        key = f"L{head[0]}H{head[1]}"
        update_attention_bucket(
            stats["by_head"][key],
            before=before,
            after=after,
            original_entity=original_entity,
            target_entity=target_entity,
            eps=eps,
        )
        update_attention_bucket(
            stats["overall"],
            before=before,
            after=after,
            original_entity=original_entity,
            target_entity=target_entity,
            eps=eps,
        )


def finalize_attention_stats(
    stats: Mapping[str, object],
    *,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    return {
        "overall": finalize_attention_bucket(
            stats["overall"],
            original_entity=original_entity,
            target_entity=target_entity,
        ),
        "by_head": {
            head_key: finalize_attention_bucket(
                bucket,
                original_entity=original_entity,
                target_entity=target_entity,
            )
            for head_key, bucket in stats["by_head"].items()
        },
    }


def init_carrier_bucket() -> Dict[str, object]:
    return {
        "before_mass_sum": [0.0 for _ in carrier_labels()],
        "after_mass_sum": [0.0 for _ in carrier_labels()],
        "count": 0,
        "log_ratio_before_sum": 0.0,
        "log_ratio_after_sum": 0.0,
        "top_before_question": 0,
        "top_after_question": 0,
        "switch_to_question": 0,
    }


def update_carrier_bucket(
    bucket: Dict[str, object],
    *,
    before: torch.Tensor,
    after: torch.Tensor,
    eps: float,
) -> None:
    before_cpu = before.detach().float().cpu()
    after_cpu = after.detach().float().cpu()
    labels = carrier_labels()
    for idx in range(len(labels)):
        bucket["before_mass_sum"][idx] += float(before_cpu[:, idx].sum().item())
        bucket["after_mass_sum"][idx] += float(after_cpu[:, idx].sum().item())
    bucket["count"] += int(before_cpu.shape[0])

    before_context = before_cpu[:, 1:].sum(dim=1)
    after_context = after_cpu[:, 1:].sum(dim=1)
    before_log_ratio = torch.log(before_cpu[:, 0] + eps) - torch.log(
        before_context + eps
    )
    after_log_ratio = torch.log(after_cpu[:, 0] + eps) - torch.log(
        after_context + eps
    )
    bucket["log_ratio_before_sum"] += float(before_log_ratio.sum().item())
    bucket["log_ratio_after_sum"] += float(after_log_ratio.sum().item())

    before_top = torch.argmax(before_cpu, dim=1)
    after_top = torch.argmax(after_cpu, dim=1)
    bucket["top_before_question"] += int((before_top == 0).sum().item())
    bucket["top_after_question"] += int((after_top == 0).sum().item())
    bucket["switch_to_question"] += int(
        ((before_top != 0) & (after_top == 0)).sum().item()
    )


def finalize_carrier_bucket(bucket: Mapping[str, object]) -> Dict[str, object]:
    count = int(bucket["count"])
    labels = carrier_labels()
    if count == 0:
        return {
            "count": 0,
            "mean_before_mass_by_carrier": None,
            "mean_after_mass_by_carrier": None,
        }
    before = [value / count for value in bucket["before_mass_sum"]]
    after = [value / count for value in bucket["after_mass_sum"]]
    before_context = sum(before[1:])
    after_context = sum(after[1:])
    before_qmc = before[0] - before_context
    after_qmc = after[0] - after_context
    before_lr = float(bucket["log_ratio_before_sum"]) / count
    after_lr = float(bucket["log_ratio_after_sum"]) / count
    return {
        "count": count,
        "carrier_order": labels,
        "mean_before_mass_by_carrier": {
            labels[idx]: before[idx] for idx in range(len(labels))
        },
        "mean_after_mass_by_carrier": {
            labels[idx]: after[idx] for idx in range(len(labels))
        },
        "question_minus_context_boxes_attention_mass": {
            "before": before_qmc,
            "after": after_qmc,
            "delta_after_minus_before": after_qmc - before_qmc,
        },
        "log_ratio_question_over_context_boxes": {
            "before": before_lr,
            "after": after_lr,
            "delta_after_minus_before": after_lr - before_lr,
        },
        "top_before_question_fraction": int(bucket["top_before_question"]) / count,
        "top_after_question_fraction": int(bucket["top_after_question"]) / count,
        "switch_to_question_fraction": int(bucket["switch_to_question"]) / count,
    }


def init_carrier_stats(heads: Sequence[Head]) -> Dict[str, object]:
    return {
        "overall": init_carrier_bucket(),
        "by_head": {f"L{layer}H{head}": init_carrier_bucket() for layer, head in heads},
    }


def update_carrier_stats(
    stats: Dict[str, object],
    *,
    before_masses: Mapping[Head, torch.Tensor],
    after_masses: Mapping[Head, torch.Tensor],
    eps: float,
) -> None:
    for head, before in before_masses.items():
        after = after_masses[head]
        key = f"L{head[0]}H{head[1]}"
        update_carrier_bucket(
            stats["by_head"][key],
            before=before,
            after=after,
            eps=eps,
        )
        update_carrier_bucket(
            stats["overall"],
            before=before,
            after=after,
            eps=eps,
        )


def finalize_carrier_stats(stats: Mapping[str, object]) -> Dict[str, object]:
    return {
        "overall": finalize_carrier_bucket(stats["overall"]),
        "by_head": {
            head_key: finalize_carrier_bucket(bucket)
            for head_key, bucket in stats["by_head"].items()
        },
    }


def init_logit_stats() -> Dict[str, object]:
    return {
        "before_logit_sum_by_entity": [0.0 for _ in range(NUM_ENTITIES)],
        "after_logit_sum_by_entity": [0.0 for _ in range(NUM_ENTITIES)],
        "delta_logit_sum_by_entity": [0.0 for _ in range(NUM_ENTITIES)],
        "pairwise_before_sum": 0.0,
        "pairwise_after_sum": 0.0,
        "pairwise_delta_sum": 0.0,
        "count": 0,
    }


def update_logit_stats(
    stats: Dict[str, object],
    *,
    before_logits_at_answer: torch.Tensor,
    after_logits_at_answer: torch.Tensor,
    object_token_ids: torch.Tensor,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    before_values = []
    after_values = []
    delta_values = []
    for entity in range(NUM_ENTITIES):
        token_id = int(object_token_ids[entity].item())
        before_logit = float(before_logits_at_answer[token_id].item())
        after_logit = float(after_logits_at_answer[token_id].item())
        delta = after_logit - before_logit
        stats["before_logit_sum_by_entity"][entity] += before_logit
        stats["after_logit_sum_by_entity"][entity] += after_logit
        stats["delta_logit_sum_by_entity"][entity] += delta
        before_values.append(before_logit)
        after_values.append(after_logit)
        delta_values.append(delta)

    before_pairwise = before_values[target_entity] - before_values[original_entity]
    after_pairwise = after_values[target_entity] - after_values[original_entity]
    pairwise_delta = after_pairwise - before_pairwise
    stats["pairwise_before_sum"] += before_pairwise
    stats["pairwise_after_sum"] += after_pairwise
    stats["pairwise_delta_sum"] += pairwise_delta
    stats["count"] += 1
    return {
        "before_logit_by_entity": {
            str(entity): before_values[entity] for entity in range(NUM_ENTITIES)
        },
        "after_logit_by_entity": {
            str(entity): after_values[entity] for entity in range(NUM_ENTITIES)
        },
        "delta_logit_by_entity": {
            str(entity): delta_values[entity] for entity in range(NUM_ENTITIES)
        },
        "pairwise_target_minus_original": {
            "before": before_pairwise,
            "after": after_pairwise,
            "delta_after_minus_before": pairwise_delta,
        },
    }


def finalize_logit_stats(
    stats: Mapping[str, object],
    *,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    count = int(stats["count"])
    if count == 0:
        return {"count": 0}
    before = [value / count for value in stats["before_logit_sum_by_entity"]]
    after = [value / count for value in stats["after_logit_sum_by_entity"]]
    delta = [value / count for value in stats["delta_logit_sum_by_entity"]]
    other_entities = [
        entity
        for entity in range(NUM_ENTITIES)
        if entity not in (int(original_entity), int(target_entity))
    ]
    other_delta = (
        sum(delta[entity] for entity in other_entities) / len(other_entities)
        if other_entities
        else None
    )
    return {
        "count": count,
        "mean_before_logit_by_entity": {
            str(entity): before[entity] for entity in range(NUM_ENTITIES)
        },
        "mean_after_logit_by_entity": {
            str(entity): after[entity] for entity in range(NUM_ENTITIES)
        },
        "mean_delta_logit_by_entity": {
            str(entity): delta[entity] for entity in range(NUM_ENTITIES)
        },
        "raw_delta_roles": {
            "original_answer": delta[original_entity],
            "target_answer": delta[target_entity],
            "other_objects_mean": other_delta,
            "other_object_entities": other_entities,
        },
        "pairwise_delta_target_minus_original": {
            "before": float(stats["pairwise_before_sum"]) / count,
            "after": float(stats["pairwise_after_sum"]) / count,
            "delta_after_minus_before": float(stats["pairwise_delta_sum"]) / count,
        },
    }


def mean_mass_by_entity(masses_by_head: Mapping[Head, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(masses_by_head.values()), dim=0)
    return stacked.mean(dim=0)


def mean_carrier_mass(masses_by_head: Mapping[Head, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(masses_by_head.values()), dim=0)
    return stacked.mean(dim=0)


def tensor_row_to_floats(row: torch.Tensor) -> Dict[str, float]:
    return {str(idx): float(value) for idx, value in enumerate(row.detach().cpu().tolist())}


def tensor_carrier_row_to_floats(row: torch.Tensor) -> Dict[str, float]:
    labels = carrier_labels()
    values = row.detach().cpu().tolist()
    return {labels[idx]: float(values[idx]) for idx in range(len(labels))}


def evaluate_conditions(
    *,
    model,
    tokenizer,
    dataloader: DataLoader,
    condition_specs: Sequence[Mapping[str, object]],
    b_heads: Sequence[Head],
    source_position_kinds: Sequence[str],
    original_entity: int,
    target_entity: int,
    device: torch.device,
    include_per_sample: bool,
    attention_eps: float,
) -> Dict[str, Dict[str, object]]:
    pattern_hook_names = hook_names_for_heads(b_heads, pattern_hook_name)
    logical_n_heads = int(getattr(model.cfg, "n_heads", 0)) or max(
        int(head) for _, head in b_heads
    ) + 1
    results: Dict[str, Dict[str, object]] = {}
    working_stats = {
        str(spec["name"]): {
            "attention": init_attention_stats(b_heads),
            "carrier_attention": init_carrier_stats(b_heads),
            "logits": init_logit_stats(),
            "per_sample": [],
        }
        for spec in condition_specs
    }

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate binding-ID Q/K interventions"):
            batch = move_batch_to_device(batch, device)
            before_logits, before_cache = cache_forward(
                model,
                batch["base_tokens"],
                cache_hook_names=pattern_hook_names,
            )
            before_masses = attention_masses_for_heads(
                cache=before_cache,
                heads=b_heads,
                batch=batch,
                source_position_kinds=source_position_kinds,
            )
            before_carrier_masses = carrier_masses_for_heads(
                cache=before_cache,
                heads=b_heads,
                batch=batch,
            )

            for spec in condition_specs:
                name = str(spec["name"])
                q_deltas = spec.get("q_deltas")
                k_entity_deltas = spec.get("k_entity_deltas")
                k_position_field_deltas = spec.get("k_position_field_deltas")
                if (
                    q_deltas is None
                    and k_entity_deltas is None
                    and k_position_field_deltas is None
                ):
                    after_logits = before_logits
                    after_cache = before_cache
                else:
                    hooks = make_intervention_hooks(
                        heads=spec["intervention_heads"],
                        q_deltas=q_deltas,
                        q_position_field=spec.get("q_position_field"),
                        k_entity_deltas=k_entity_deltas,
                        k_position_field_deltas=k_position_field_deltas,
                        source_position_kinds=source_position_kinds,
                        alpha=float(spec["alpha"]),
                        batch=batch,
                        logical_n_heads=logical_n_heads,
                    )
                    after_logits, after_cache = cache_forward(
                        model,
                        batch["base_tokens"],
                        cache_hook_names=pattern_hook_names,
                        fwd_hooks=hooks,
                    )

                after_masses = attention_masses_for_heads(
                    cache=after_cache,
                    heads=b_heads,
                    batch=batch,
                    source_position_kinds=source_position_kinds,
                )
                after_carrier_masses = carrier_masses_for_heads(
                    cache=after_cache,
                    heads=b_heads,
                    batch=batch,
                )
                update_attention_stats(
                    working_stats[name]["attention"],
                    before_masses=before_masses,
                    after_masses=after_masses,
                    original_entity=original_entity,
                    target_entity=target_entity,
                    eps=attention_eps,
                )
                update_carrier_stats(
                    working_stats[name]["carrier_attention"],
                    before_masses=before_carrier_masses,
                    after_masses=after_carrier_masses,
                    eps=attention_eps,
                )

                batch_size = int(batch["base_tokens"].shape[0])
                before_mean_mass = mean_mass_by_entity(before_masses)
                after_mean_mass = mean_mass_by_entity(after_masses)
                before_mean_carrier_mass = mean_carrier_mass(before_carrier_masses)
                after_mean_carrier_mass = mean_carrier_mass(after_carrier_masses)
                for batch_idx in range(batch_size):
                    answer_pos = int(batch["base_last_token_indices"][batch_idx].item())
                    object_token_ids = batch["object_token_ids"][batch_idx].to(torch.long)
                    sample_logit_payload = update_logit_stats(
                        working_stats[name]["logits"],
                        before_logits_at_answer=before_logits[batch_idx, answer_pos],
                        after_logits_at_answer=after_logits[batch_idx, answer_pos],
                        object_token_ids=object_token_ids,
                        original_entity=original_entity,
                        target_entity=target_entity,
                    )
                    if include_per_sample:
                        before_mass_row = before_mean_mass[batch_idx]
                        after_mass_row = after_mean_mass[batch_idx]
                        before_carrier_row = before_mean_carrier_mass[batch_idx]
                        after_carrier_row = after_mean_carrier_mass[batch_idx]
                        working_stats[name]["per_sample"].append(
                            {
                                "prompt_id": int(batch["prompt_ids"][batch_idx].item()),
                                "query_entity": int(
                                    batch["query_entities"][batch_idx].item()
                                ),
                                "original_source_entity": int(original_entity),
                                "target_source_entity": int(target_entity),
                                "object_tokens": {
                                    str(entity): {
                                        "token_id": int(object_token_ids[entity].item()),
                                        "token": tokenizer.decode(
                                            [int(object_token_ids[entity].item())],
                                            clean_up_tokenization_spaces=False,
                                        ),
                                    }
                                    for entity in range(NUM_ENTITIES)
                                },
                                "mean_b_attention_mass_by_entity": {
                                    "before": tensor_row_to_floats(before_mass_row),
                                    "after": tensor_row_to_floats(after_mass_row),
                                },
                                "mean_b_carrier_attention_mass": {
                                    "before": tensor_carrier_row_to_floats(
                                        before_carrier_row
                                    ),
                                    "after": tensor_carrier_row_to_floats(
                                        after_carrier_row
                                    ),
                                },
                                "mean_b_top_entity": {
                                    "before": int(torch.argmax(before_mass_row).item()),
                                    "after": int(torch.argmax(after_mass_row).item()),
                                },
                                "mean_b_top_carrier": {
                                    "before": carrier_labels()[
                                        int(torch.argmax(before_carrier_row).item())
                                    ],
                                    "after": carrier_labels()[
                                        int(torch.argmax(after_carrier_row).item())
                                    ],
                                },
                                "logits": sample_logit_payload,
                            }
                        )

                if after_cache is not before_cache:
                    del after_cache, after_logits
            del before_cache, before_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for spec in condition_specs:
        name = str(spec["name"])
        stats = working_stats[name]
        public_spec = {
            key: value
            for key, value in spec.items()
            if key not in {"q_deltas", "k_entity_deltas", "k_position_field_deltas"}
        }
        intervention_heads = list(public_spec.get("intervention_heads", []))
        q_intervention_hooks = (
            hook_names_for_heads(intervention_heads, q_hook_name)
            if spec.get("q_deltas") is not None
            else []
        )
        k_intervention_hooks = (
            hook_names_for_heads(intervention_heads, k_hook_name)
            if (
                spec.get("k_entity_deltas") is not None
                or spec.get("k_position_field_deltas") is not None
            )
            else []
        )
        public_spec["intervention_heads"] = [
            {"layer": int(layer), "head": int(head)}
            for layer, head in intervention_heads
        ]
        public_spec["intervention_hook_names"] = {
            "q": q_intervention_hooks,
            "k": k_intervention_hooks,
            "pattern_readout": hook_names_for_heads(b_heads, pattern_hook_name),
        }
        results[name] = {
            "condition": public_spec,
            "attention": finalize_attention_stats(
                stats["attention"],
                original_entity=original_entity,
                target_entity=target_entity,
            ),
            "carrier_attention": finalize_carrier_stats(stats["carrier_attention"]),
            "final_logits": finalize_logit_stats(
                stats["logits"],
                original_entity=original_entity,
                target_entity=target_entity,
            ),
            "per_sample": stats["per_sample"] if include_per_sample else None,
        }
    return results


def build_condition_specs(
    *,
    requested_conditions: Sequence[str],
    b_heads: Sequence[Head],
    wrong_heads: Sequence[Head],
    original_entity: int,
    target_entity: int,
    distractor_entity: int,
    alpha: float,
    b_q_original_to_target: TensorByHead,
    b_k_target_to_original: TensorByHead,
    b_k_original_to_target: TensorByHead,
    q_select_question_query_box_more: TensorByHead,
    q_select_question_query_box_less: TensorByHead,
    k_select_question_query_box_more: TensorByHead,
    k_select_question_query_box_less: TensorByHead,
    random_q_original_to_target: TensorByHead,
    random_k_target_to_original: TensorByHead,
    random_k_original_to_target: TensorByHead,
    wrong_q_original_to_target: Optional[TensorByHead],
    wrong_k_target_to_original: Optional[TensorByHead],
    wrong_k_original_to_target: Optional[TensorByHead],
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []

    def add(
        name: str,
        *,
        category: str,
        description: str,
        intervention_heads: Sequence[Head],
        q_deltas: Optional[TensorByHead] = None,
        q_position_field: Optional[str] = None,
        k_entity_deltas: Optional[Mapping[int, TensorByHead]] = None,
        k_position_field_deltas: Optional[Mapping[str, TensorByHead]] = None,
    ) -> None:
        specs.append(
            {
                "name": name,
                "category": category,
                "description": description,
                "alpha": float(alpha),
                "intervention_heads": list(intervention_heads),
                "q_deltas": q_deltas,
                "q_position_field": q_position_field,
                "k_entity_deltas": dict(k_entity_deltas) if k_entity_deltas else None,
                "k_position_field_deltas": (
                    dict(k_position_field_deltas) if k_position_field_deltas else None
                ),
                "k_delta_entities": (
                    {str(entity): "custom_delta" for entity in k_entity_deltas}
                    if k_entity_deltas
                    else {}
                ),
                "k_delta_position_fields": (
                    {str(field): "custom_delta" for field in k_position_field_deltas}
                    if k_position_field_deltas
                    else {}
                ),
            }
        )

    for condition in requested_conditions:
        if condition == "no_intervention":
            add(
                condition,
                category="baseline",
                description="No Q/K intervention; after metrics equal before metrics.",
                intervention_heads=[],
            )
        elif condition == "query_shift":
            add(
                condition,
                category="main",
                description=(
                    "Add Q delta original_source->target_source to B heads at "
                    "answer_colon."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "source_swap":
            add(
                condition,
                category="main",
                description=(
                    "Swap K-side IDs at original and target context source positions "
                    "for B heads."
                ),
                intervention_heads=b_heads,
                k_entity_deltas={
                    int(target_entity): b_k_target_to_original,
                    int(original_entity): b_k_original_to_target,
                },
            )
        elif condition == "both_restore":
            add(
                condition,
                category="main",
                description=(
                    "Apply Q original->target and the K swap together; the shifted "
                    "query should again match the original source slot."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="base_answer_colon_positions",
                k_entity_deltas={
                    int(target_entity): b_k_target_to_original,
                    int(original_entity): b_k_original_to_target,
                },
            )
        elif condition == "random_query_shift":
            add(
                condition,
                category="control",
                description="Matched-norm random Q direction on B heads at answer_colon.",
                intervention_heads=b_heads,
                q_deltas=random_q_original_to_target,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "random_source_swap":
            add(
                condition,
                category="control",
                description="Matched-norm random K directions on B heads at source slots.",
                intervention_heads=b_heads,
                k_entity_deltas={
                    int(target_entity): random_k_target_to_original,
                    int(original_entity): random_k_original_to_target,
                },
            )
        elif condition == "random_both":
            add(
                condition,
                category="control",
                description="Matched-norm random Q and K directions on B heads.",
                intervention_heads=b_heads,
                q_deltas=random_q_original_to_target,
                q_position_field="base_answer_colon_positions",
                k_entity_deltas={
                    int(target_entity): random_k_target_to_original,
                    int(original_entity): random_k_original_to_target,
                },
            )
        elif condition == "wrong_head_query_shift":
            if wrong_q_original_to_target is None:
                raise ValueError("wrong_head_query_shift requested without wrong-head Q deltas.")
            add(
                condition,
                category="control",
                description="Q original->target shift applied to non-B heads.",
                intervention_heads=wrong_heads,
                q_deltas=wrong_q_original_to_target,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "wrong_head_source_swap":
            if wrong_k_target_to_original is None or wrong_k_original_to_target is None:
                raise ValueError("wrong_head_source_swap requested without wrong-head K deltas.")
            add(
                condition,
                category="control",
                description="K source-slot swap applied to non-B heads.",
                intervention_heads=wrong_heads,
                k_entity_deltas={
                    int(target_entity): wrong_k_target_to_original,
                    int(original_entity): wrong_k_original_to_target,
                },
            )
        elif condition == "wrong_head_both":
            if (
                wrong_q_original_to_target is None
                or wrong_k_target_to_original is None
                or wrong_k_original_to_target is None
            ):
                raise ValueError("wrong_head_both requested without wrong-head deltas.")
            add(
                condition,
                category="control",
                description="Q and K binding-ID shifts applied to matched non-B heads.",
                intervention_heads=wrong_heads,
                q_deltas=wrong_q_original_to_target,
                q_position_field="base_answer_colon_positions",
                k_entity_deltas={
                    int(target_entity): wrong_k_target_to_original,
                    int(original_entity): wrong_k_original_to_target,
                },
            )
        elif condition == "wrong_position_query_shift":
            add(
                condition,
                category="control",
                description=(
                    "B-head Q original->target shift applied at the question box token "
                    "instead of answer_colon."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="query_box_positions",
            )
        elif condition == "wrong_position_source_shift":
            add(
                condition,
                category="control",
                description=(
                    "B-head K target->original shift applied to a distractor source "
                    "slot rather than original/target slots."
                ),
                intervention_heads=b_heads,
                k_entity_deltas={int(distractor_entity): b_k_target_to_original},
            )
        elif condition == "wrong_position_both":
            add(
                condition,
                category="control",
                description=(
                    "B-head Q shift at the question box plus K shift at a distractor "
                    "source slot."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="query_box_positions",
                k_entity_deltas={int(distractor_entity): b_k_target_to_original},
            )
        elif condition == "question_query_box_source_shift":
            add(
                condition,
                category="control",
                description=(
                    "B-head K target->original binding-ID delta applied at the "
                    "question_query_box token as the source position."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": b_k_target_to_original,
                },
            )
        elif condition == "q_select_question_query_box_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Add a matched-norm mean K(question_query_box) direction to "
                    "B-head Q at answer_colon, increasing compatibility with the "
                    "question_query_box carrier."
                ),
                intervention_heads=b_heads,
                q_deltas=q_select_question_query_box_more,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "q_select_question_query_box_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Subtract the matched-norm mean K(question_query_box) direction "
                    "from B-head Q at answer_colon, decreasing compatibility with "
                    "the question_query_box carrier."
                ),
                intervention_heads=b_heads,
                q_deltas=q_select_question_query_box_less,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "k_question_query_box_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Add a matched-norm mean Q(answer_colon) direction to B-head K "
                    "at the question_query_box token."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_more,
                },
            )
        elif condition == "k_question_query_box_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Subtract the matched-norm mean Q(answer_colon) direction from "
                    "B-head K at the question_query_box token."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_less,
                },
            )
        elif condition == "k_competing_carriers_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Add a matched-norm mean Q(answer_colon) direction to B-head K "
                    "at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "box_token_positions": k_select_question_query_box_more,
                },
            )
        elif condition == "k_competing_carriers_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Subtract the matched-norm mean Q(answer_colon) direction from "
                    "B-head K at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "box_token_positions": k_select_question_query_box_less,
                },
            )
        elif condition == "k_question_query_box_more_competing_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Increase B-head K compatibility at question_query_box while "
                    "decreasing compatibility at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_more,
                    "box_token_positions": k_select_question_query_box_less,
                },
            )
        elif condition == "k_question_query_box_less_competing_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Decrease B-head K compatibility at question_query_box while "
                    "increasing compatibility at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_less,
                    "box_token_positions": k_select_question_query_box_more,
                },
            )
        else:
            raise ValueError(f"Unhandled condition: {condition}")
    return specs


def serialize_heads(heads: Sequence[Head]) -> List[Dict[str, int]]:
    return [{"layer": int(layer), "head": int(head)} for layer, head in heads]


def serialize_delta_norms(deltas: Mapping[Head, torch.Tensor]) -> Dict[str, float]:
    return {
        f"L{layer}H{head}": float(torch.linalg.vector_norm(delta.float()).item())
        for (layer, head), delta in sorted(deltas.items())
    }


def serialize_kv_head_mapping(
    heads: Sequence[Head],
    *,
    logical_n_heads: int,
    kv_n_heads: int,
) -> Dict[str, int]:
    return {
        f"L{layer}H{head}": key_value_head_for_attention_head(
            int(head),
            logical_n_heads=logical_n_heads,
            activation_n_heads=kv_n_heads,
        )
        for layer, head in sorted(heads)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run genuine binding-ID Q/K interventions on the kept group-B "
            "attention heads from minimality_pruning.kept_components."
        )
    )
    parser.add_argument("--model-name", "--model_name", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--components-json",
        "--components_json",
        type=Path,
        default=DEFAULT_COMPONENTS_JSON,
        help="Circuit JSON containing minimality_pruning.kept_components.",
    )
    parser.add_argument(
        "--group-b-stage",
        "--group_b_stage",
        default="pos_transmitter",
        help="Expected kept-component stage for group B.",
    )
    parser.add_argument(
        "--group-b-position-modes",
        "--group_b_position_modes",
        default="answer_colon,last_token",
        help="Comma-separated accepted position modes for group B.",
    )
    parser.add_argument("--vocab-tag", "--vocab_tag", default="BOXES")
    parser.add_argument("--vocab-split", "--vocab_split", default="train")
    parser.add_argument(
        "--prompt-format",
        "--prompt_format",
        choices=PROMPT_FORMATS,
        default="auto",
    )
    parser.add_argument("--template-type", "--template_type", default="normal")
    parser.add_argument("--prompt-id-start", "--prompt_id_start", type=int, default=0)
    parser.add_argument("--num-samples", "--num_samples", type=int, default=100)
    parser.add_argument(
        "--direction-prompt-id-start",
        "--direction_prompt_id_start",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--direction-num-samples",
        "--direction_num_samples",
        type=int,
        default=100,
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--source-position-kinds",
        "--source_position_kinds",
        type=parse_source_position_kinds,
        default=parse_source_position_kinds("box,object"),
        help="Comma-separated K source/readout token kinds: box, object, or box,object.",
    )
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=DEFAULT_CONDITIONS,
        help=f"Comma-separated conditions. Choices: {', '.join(CONDITION_CHOICES)}.",
    )
    parser.add_argument("--query-entity", "--query_entity", type=int, default=0)
    parser.add_argument(
        "--target-source-entity",
        "--target_source_entity",
        type=int,
        default=0,
        help="Source slot/entity to shift attention toward.",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--include-swap",
        "--include_swap",
        action="store_true",
        default=True,
        help="Include the BOXES swap sentence. Enabled by default.",
    )
    parser.add_argument(
        "--no-include-swap",
        "--no_include_swap",
        dest="include_swap",
        action="store_false",
        help="Disable the BOXES swap sentence.",
    )
    parser.add_argument(
        "--swap-pair",
        "--swap_pair",
        default="0,1",
        help="Swap sentence pair. Default gives Query 0 original source slot 1.",
    )
    parser.add_argument(
        "--attention-eps",
        "--attention_eps",
        type=float,
        default=1e-9,
    )
    parser.add_argument("--output-json", "--output_json", type=Path, default=None)
    parser.add_argument(
        "--no-per-sample",
        "--no_per_sample",
        action="store_true",
        help="Omit optional per-sample results from the JSON payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    components_path = args.components_json.expanduser().resolve()
    if not components_path.exists():
        raise FileNotFoundError(components_path)

    swap_pair = parse_int_tuple(args.swap_pair, expected_len=2)
    if len(set(swap_pair)) != 2:
        raise ValueError(f"--swap-pair entries must be different, got {swap_pair}.")
    if not (0 <= args.query_entity < NUM_ENTITIES):
        raise ValueError(f"--query-entity must be in [0, {NUM_ENTITIES}), got {args.query_entity}.")
    if not (0 <= args.target_source_entity < NUM_ENTITIES):
        raise ValueError(
            f"--target-source-entity must be in [0, {NUM_ENTITIES}), "
            f"got {args.target_source_entity}."
        )

    resolved_model_name = resolve_model_name(args.model_name)
    resolved_prompt_format = resolve_prompt_format(
        resolved_model_name,
        args.prompt_format,
    )
    selected_device = normalize_device(args.device)
    set_seed(args.seed)

    group_b_components, b_heads, component_payload, kept_components, nonmatching = (
        load_group_b_components(
            components_path,
            expected_stage=args.group_b_stage,
            expected_position_modes=parse_csv(args.group_b_position_modes),
        )
    )
    sanity_rows = sanity_table(group_b_components)
    print_sanity_table(sanity_rows)

    original_source_entity = source_slot_for_query(
        args.query_entity,
        include_swap=bool(args.include_swap),
        swap_pair=swap_pair,
    )
    target_source_entity = int(args.target_source_entity)
    if original_source_entity == target_source_entity:
        raise ValueError(
            "Target source entity equals the original source entity. Choose a "
            "different --target-source-entity for an intervention."
        )
    distractor_entity = next(
        entity
        for entity in range(NUM_ENTITIES)
        if entity not in (original_source_entity, target_source_entity)
    )

    print(f"Model: {resolved_model_name}", file=sys.stderr)
    print(f"Prompt format: {resolved_prompt_format}", file=sys.stderr)
    print(f"Components JSON: {components_path}", file=sys.stderr)
    print(
        "Loaded minimality_pruning.kept_components: "
        f"{len(kept_components)}; selected group-B entries: {len(group_b_components)}",
        file=sys.stderr,
    )
    if nonmatching:
        print(
            "Kept components outside the configured group-B stage/site were "
            f"loaded for provenance but not used as group B: {len(nonmatching)}",
            file=sys.stderr,
        )
    print(f"Selected B heads: {serialize_heads(b_heads)}", file=sys.stderr)
    print(
        f"Query entity={args.query_entity}; original source={original_source_entity}; "
        f"target source={target_source_entity}; distractor={distractor_entity}",
        file=sys.stderr,
    )
    print(f"Source position kinds: {args.source_position_kinds}", file=sys.stderr)
    print(f"Conditions: {args.conditions}", file=sys.stderr)
    print(f"Device: {selected_device}", file=sys.stderr)

    model, tokenizer = get_model_and_tokenizer(resolved_model_name)
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
    vocab = load_vocab(
        args.vocab_tag,
        resolved_model_name,
        split=args.vocab_split,
        tokenizer=tokenizer,
    )

    direction_dataloaders = {
        source_slot: DataLoader(
            build_binding_id_dataset(
                tokenizer=tokenizer,
                model_name=resolved_model_name,
                vocab=vocab,
                query_entity=query_entity_for_source_slot(
                    source_slot,
                    include_swap=bool(args.include_swap),
                    swap_pair=swap_pair,
                ),
                num_samples=args.direction_num_samples,
                prompt_id_start=args.direction_prompt_id_start,
                prompt_format=resolved_prompt_format,
                include_swap=bool(args.include_swap),
                swap_pair=swap_pair,
                template_type=args.template_type,
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        for source_slot in range(NUM_ENTITIES)
    }

    b_direction_payload = estimate_binding_id_directions(
        model=model,
        dataloaders_by_source_slot=direction_dataloaders,
        heads=b_heads,
        source_position_kinds=args.source_position_kinds,
        device=selected_device,
        label="B heads",
    )
    b_q_original_to_target = delta_between(
        b_direction_payload["q_means"],
        original_source_entity,
        target_source_entity,
    )
    b_k_target_to_original = delta_between(
        b_direction_payload["k_means"],
        target_source_entity,
        original_source_entity,
    )
    b_k_original_to_target = delta_between(
        b_direction_payload["k_means"],
        original_source_entity,
        target_source_entity,
    )
    b_q_current = b_direction_payload["q_means"][int(original_source_entity)]
    b_k_question_query_box_current = b_direction_payload["k_position_means"][
        "query_box_positions"
    ][int(original_source_entity)]
    q_select_question_query_box_more = matched_norm_directions(
        b_k_question_query_box_current,
        reference=b_q_original_to_target,
    )
    q_select_question_query_box_less = negate_deltas(q_select_question_query_box_more)
    k_select_question_query_box_more = matched_norm_directions(
        b_q_current,
        reference=b_k_original_to_target,
    )
    k_select_question_query_box_less = negate_deltas(k_select_question_query_box_more)

    random_q_original_to_target = random_matched_deltas(
        b_q_original_to_target,
        seed=args.seed + 1001,
    )
    random_k_target_to_original = random_matched_deltas(
        b_k_target_to_original,
        seed=args.seed + 1002,
    )
    random_k_original_to_target = random_matched_deltas(
        b_k_original_to_target,
        seed=args.seed + 1003,
    )

    wrong_heads: List[Head] = []
    wrong_q_original_to_target = None
    wrong_k_target_to_original = None
    wrong_k_original_to_target = None
    if any(condition.startswith("wrong_head") for condition in args.conditions):
        wrong_heads = select_wrong_heads(model, b_heads)
        wrong_direction_payload = estimate_binding_id_directions(
            model=model,
            dataloaders_by_source_slot=direction_dataloaders,
            heads=wrong_heads,
            source_position_kinds=args.source_position_kinds,
            device=selected_device,
            label="wrong heads",
        )
        wrong_q_original_to_target = delta_between(
            wrong_direction_payload["q_means"],
            original_source_entity,
            target_source_entity,
        )
        wrong_k_target_to_original = delta_between(
            wrong_direction_payload["k_means"],
            target_source_entity,
            original_source_entity,
        )
        wrong_k_original_to_target = delta_between(
            wrong_direction_payload["k_means"],
            original_source_entity,
            target_source_entity,
        )
    else:
        wrong_direction_payload = None

    eval_dataloader = DataLoader(
        build_binding_id_dataset(
            tokenizer=tokenizer,
            model_name=resolved_model_name,
            vocab=vocab,
            query_entity=args.query_entity,
            num_samples=args.num_samples,
            prompt_id_start=args.prompt_id_start,
            prompt_format=resolved_prompt_format,
            include_swap=bool(args.include_swap),
            swap_pair=swap_pair,
            template_type=args.template_type,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    condition_specs = build_condition_specs(
        requested_conditions=args.conditions,
        b_heads=b_heads,
        wrong_heads=wrong_heads,
        original_entity=original_source_entity,
        target_entity=target_source_entity,
        distractor_entity=distractor_entity,
        alpha=float(args.alpha),
        b_q_original_to_target=b_q_original_to_target,
        b_k_target_to_original=b_k_target_to_original,
        b_k_original_to_target=b_k_original_to_target,
        q_select_question_query_box_more=q_select_question_query_box_more,
        q_select_question_query_box_less=q_select_question_query_box_less,
        k_select_question_query_box_more=k_select_question_query_box_more,
        k_select_question_query_box_less=k_select_question_query_box_less,
        random_q_original_to_target=random_q_original_to_target,
        random_k_target_to_original=random_k_target_to_original,
        random_k_original_to_target=random_k_original_to_target,
        wrong_q_original_to_target=wrong_q_original_to_target,
        wrong_k_target_to_original=wrong_k_target_to_original,
        wrong_k_original_to_target=wrong_k_original_to_target,
    )

    condition_results = evaluate_conditions(
        model=model,
        tokenizer=tokenizer,
        dataloader=eval_dataloader,
        condition_specs=condition_specs,
        b_heads=b_heads,
        source_position_kinds=args.source_position_kinds,
        original_entity=original_source_entity,
        target_entity=target_source_entity,
        device=selected_device,
        include_per_sample=not args.no_per_sample,
        attention_eps=float(args.attention_eps),
    )

    control_results = {
        name: result
        for name, result in condition_results.items()
        if name in CONTROL_CONDITIONS
    }
    main_results = {
        name: result
        for name, result in condition_results.items()
        if name not in CONTROL_CONDITIONS
    }

    payload = {
        "model_name": resolved_model_name,
        "requested_model_name": args.model_name,
        "prompt_format": resolved_prompt_format,
        "requested_prompt_format": args.prompt_format,
        "components_json": str(components_path),
        "source_circuit_model": component_payload.get("model_name"),
        "loaded_minimality_pruning_kept_components": kept_components,
        "selected_b_components": sanity_rows,
        "selected_b_heads": serialize_heads(b_heads),
        "non_group_b_kept_components": sanity_table(nonmatching),
        "group_b_selection": {
            "component_set": "minimality_pruning.kept_components",
            "expected_stage": args.group_b_stage,
            "expected_position_modes": list(parse_csv(args.group_b_position_modes)),
            "normalized_expected_position_modes": sorted(
                {
                    normalize_position_mode(position_mode)
                    for position_mode in parse_csv(args.group_b_position_modes)
                }
            ),
            "recorded_hooks_are_discovery_hooks_only": True,
            "intervention_hooks": {
                "q": hook_names_for_heads(b_heads, q_hook_name),
                "k": hook_names_for_heads(b_heads, k_hook_name),
                "pattern_readout": hook_names_for_heads(b_heads, pattern_hook_name),
            },
            "gqa_head_mapping": {
                "logical_attention_heads": int(logical_n_heads),
                "key_value_heads": int(kv_n_heads),
                "selected_b_attention_head_to_kv_head": serialize_kv_head_mapping(
                    b_heads,
                    logical_n_heads=logical_n_heads,
                    kv_n_heads=kv_n_heads,
                ),
                "wrong_attention_head_to_kv_head": serialize_kv_head_mapping(
                    wrong_heads,
                    logical_n_heads=logical_n_heads,
                    kv_n_heads=kv_n_heads,
                )
                if wrong_heads
                else {},
            },
        },
        "direction_estimation": {
            "spaces": "B-local hook_q and hook_k head dimensions",
            "prompt_id_start": args.direction_prompt_id_start,
            "num_samples": args.direction_num_samples,
            "batch_size": args.batch_size,
            "source_position_kinds_for_k": list(args.source_position_kinds),
            "carrier_position_fields_for_k": ["query_box_positions"],
            "source_slot_to_query_entity": {
                str(source_slot): query_entity_for_source_slot(
                    source_slot,
                    include_swap=bool(args.include_swap),
                    swap_pair=swap_pair,
                )
                for source_slot in range(NUM_ENTITIES)
            },
            "b_q_counts": b_direction_payload["q_counts"],
            "b_k_counts": b_direction_payload["k_counts"],
            "b_k_carrier_counts": b_direction_payload["k_position_counts"],
            "delta_norms": {
                "b_q_original_to_target": serialize_delta_norms(
                    b_q_original_to_target
                ),
                "b_k_target_to_original": serialize_delta_norms(
                    b_k_target_to_original
                ),
                "b_k_original_to_target": serialize_delta_norms(
                    b_k_original_to_target
                ),
                "q_select_question_query_box_more": serialize_delta_norms(
                    q_select_question_query_box_more
                ),
                "q_select_question_query_box_less": serialize_delta_norms(
                    q_select_question_query_box_less
                ),
                "k_select_question_query_box_more": serialize_delta_norms(
                    k_select_question_query_box_more
                ),
                "k_select_question_query_box_less": serialize_delta_norms(
                    k_select_question_query_box_less
                ),
                "random_q_original_to_target": serialize_delta_norms(
                    random_q_original_to_target
                ),
                "random_k_target_to_original": serialize_delta_norms(
                    random_k_target_to_original
                ),
                "random_k_original_to_target": serialize_delta_norms(
                    random_k_original_to_target
                ),
            },
            "wrong_head_counts": (
                {
                    "q": wrong_direction_payload["q_counts"],
                    "k": wrong_direction_payload["k_counts"],
                    "k_carriers": wrong_direction_payload["k_position_counts"],
                }
                if wrong_direction_payload is not None
                else None
            ),
        },
        "intervention_setup": {
            "alpha": float(args.alpha),
            "query_entity": int(args.query_entity),
            "include_swap": bool(args.include_swap),
            "swap_pair": list(swap_pair) if args.include_swap else None,
            "original_source_entity": int(original_source_entity),
            "target_source_entity": int(target_source_entity),
            "distractor_entity_for_wrong_position_control": int(distractor_entity),
            "q_intervention_target": "base_answer_colon_positions",
            "q_wrong_position_target": "query_box_positions",
            "k_intervention_source_positions": list(args.source_position_kinds),
            "source_side_k_rule": {
                "target_source_entity": "delta_k_target_to_original",
                "original_source_entity": "delta_k_original_to_target",
            },
            "carrier_selection_rule": {
                "question_query_box_position_field": "query_box_positions",
                "competing_context_carrier_position_field": "box_token_positions",
                "q_side": (
                    "Q(answer_colon) receives +/- matched-norm mean "
                    "K(question_query_box)."
                ),
                "k_side": (
                    "K(question_query_box) and/or context box carriers receive "
                    "+/- matched-norm mean Q(answer_colon)."
                ),
            },
            "both_side_rule": (
                "Q is shifted original->target while K swaps original and target, "
                "so the shifted query should again match the original source slot."
            ),
        },
        "num_entities": NUM_ENTITIES,
        "base_object_ids_by_entity": BASE_OBJECT_IDS_BY_ENTITY,
        "vocab_tag": args.vocab_tag,
        "vocab_split": args.vocab_split,
        "prompt_id_start": args.prompt_id_start,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "conditions_requested": list(args.conditions),
        "wrong_head_control_heads": serialize_heads(wrong_heads),
        "results": main_results,
        "control_results": control_results,
    }

    rendered = json.dumps(payload, indent=2, allow_nan=True)
    if args.output_json is not None:
        output_path = args.output_json.expanduser()
        if not output_path.is_absolute():
            output_path = SCRIPT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output_path}", file=sys.stderr)
    print(rendered)


if __name__ == "__main__":
    main()
