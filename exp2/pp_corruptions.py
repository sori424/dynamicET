from __future__ import annotations

from exp2.pp_prompts import (
    ASSIGNMENT_RE,
    BOX_NAME_POOL,
    QUERY_SENTENCE_RE,
    QUESTION_SENTENCE_RE,
    SWAP_BOXES_RE,
    _DEFAULT_OBJECT_POOL,
    _as_long_tensor,
    _clean_answer,
    _clean_prompt,
    _compute_last_token_indices_from_tokens,
    _compute_prompt_position_fields,
    _corrupt_prompt,
    _decode_prompt_from_tokens,
    _encode_prompts,
    _encode_single_token_answer,
    _first_present,
    _parse_swap_prompt,
    _query_source_object,
    _slice_tensor_rows,
    _squeeze_label_tensor,
    curr_dir,
)

from collections import defaultdict
from datasets import Dataset
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import csv
import random
import re
import torch

def _replace_assignment_object(prompt: str, target_box: str, new_object: str) -> str:
    for match in ASSIGNMENT_RE.finditer(prompt):
        if match.group(1).upper() != target_box.upper():
            continue
        obj_start, obj_end = match.span(2)
        return prompt[:obj_start] + new_object + prompt[obj_end:]
    raise ValueError(f"Could not find assignment for Box {target_box} in prompt:\n{prompt}")


def _default_object_pool() -> Tuple[str, ...]:
    global _DEFAULT_OBJECT_POOL

    if _DEFAULT_OBJECT_POOL is None:
        objects_path = Path(curr_dir) / "coref" / "datasets" / "raw" / "objects.csv"
        with objects_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _DEFAULT_OBJECT_POOL = tuple(
                row["objects"] for row in reader if row.get("objects")
            )

    return _DEFAULT_OBJECT_POOL


def _rerandomize_swap_prompt(
    prompt: str,
    meta: Mapping[str, str],
    object_pool: Sequence[str],
) -> str:
    anchor_boxes = set(meta.get("all_boxes", (meta["box1"], meta["box2"])))
    available_boxes = [box for box in BOX_NAME_POOL if box not in anchor_boxes]
    if len(available_boxes) < 2:
        raise ValueError(
            "Could not sample a fully different counterfactual prompt because "
            "there are not enough alternative box labels."
        )
    new_box1, new_box2 = random.sample(available_boxes, 2)

    anchor_objects = set(meta.get("all_objects", (meta["obj1"], meta["obj2"])))
    available_objects = [obj for obj in object_pool if obj not in anchor_objects]
    if len(available_objects) < 2:
        raise ValueError(
            "Could not sample a fully different counterfactual prompt because "
            "there are not enough alternative object words."
        )
    new_obj1, new_obj2 = random.sample(available_objects, 2)

    query_box = new_box1 if meta["query_box"] == meta["box1"] else new_box2
    assignment_replacements = {
        meta["box1"]: (new_box1, new_obj1),
        meta["box2"]: (new_box2, new_obj2),
    }

    def replace_assignment(match: re.Match) -> str:
        old_box = match.group(1).upper()
        replacement = assignment_replacements.get(old_box)
        if replacement is None:
            return match.group(0)
        return "Box {} contains the {}.".format(*replacement)

    rerandomized_prompt = ASSIGNMENT_RE.sub(replace_assignment, prompt)
    rerandomized_prompt = SWAP_BOXES_RE.sub(
        f"Swap the items of Box {new_box1} and Box {new_box2}.",
        rerandomized_prompt,
        count=1,
    )
    rerandomized_prompt = QUESTION_SENTENCE_RE.sub(
        f"Question: Which item does Box {query_box} contain?",
        rerandomized_prompt,
        count=1,
    )

    def replace_query_stem(match: re.Match) -> str:
        prefix = (
            "Answer: "
            if re.match(r"\s*Answer:", match.group(0), re.IGNORECASE)
            else ""
        )
        return f"{prefix}Box {query_box} contains the"

    rerandomized_prompt = QUERY_SENTENCE_RE.sub(
        replace_query_stem,
        rerandomized_prompt,
        count=1,
    )
    return rerandomized_prompt


def _select_donor_source_object(
    anchor_meta: Mapping[str, str],
    all_metas: Sequence[Mapping[str, str]],
    anchor_index: int,
) -> str:
    fallback_object: Optional[str] = None
    anchor_objects = {anchor_meta["obj1"], anchor_meta["obj2"]}
    preferred_query_box = anchor_meta["box1"] if anchor_meta["query_box"] == anchor_meta["box2"] else anchor_meta["box2"]

    for pass_query_box in (preferred_query_box, None):
        for offset in range(1, len(all_metas)):
            donor_meta = all_metas[(anchor_index + offset) % len(all_metas)]
            donor_source_object = _query_source_object(donor_meta)
            if donor_source_object in anchor_objects:
                continue
            if pass_query_box is not None and donor_meta["query_box"] != pass_query_box:
                if fallback_object is None:
                    fallback_object = donor_source_object
                continue
            return donor_source_object

    if fallback_object is not None:
        return fallback_object

    pooled_objects = sorted(
        {
            meta["obj1"]
            for meta in all_metas
        }.union({meta["obj2"] for meta in all_metas})
        - anchor_objects
    )
    if pooled_objects:
        return pooled_objects[0]

    raise ValueError(
        "Could not construct counterfactual prompts from the provided training data. "
        "Need at least two distinct object values across the anchor samples."
    )


def _generate_corrupt_prompts_from_anchor_prompts(
    clean_prompts: Sequence[str],
) -> List[str]:
    if not clean_prompts:
        raise ValueError("Received no clean prompts to build counterfactual prompts from.")

    metas = [_parse_swap_prompt(prompt) for prompt in clean_prompts]
    object_pool = list(
        dict.fromkeys(
            [meta["obj1"] for meta in metas]
            + [meta["obj2"] for meta in metas]
            + list(_default_object_pool())
        )
    )
    corrupt_prompts: List[str] = []
    for prompt, meta in zip(clean_prompts, metas):
        corrupt_prompts.append(_rerandomize_swap_prompt(prompt, meta, object_pool))
    return corrupt_prompts


def _normalize_pretokenized_mapping(token_data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    aliases = {
        "base_tokens": ("base_tokens", "clean_tokens", "input_tokens", "input_ids", "tokens"),
        "source_tokens": (
            "source_tokens",
            "corrupt_tokens",
            "counterfactual_tokens",
            "source_input_ids",
        ),
        "labels": ("labels", "answer_tokens", "answer_ids", "output_ids"),
        "base_last_token_indices": ("base_last_token_indices", "last_token_indices"),
        "source_last_token_indices": (
            "source_last_token_indices",
            "corrupt_last_token_indices",
            "source_last_indices",
        ),
    }

    normalized = {}
    for canonical_key, keys in aliases.items():
        value = _first_present(token_data, keys)
        if value is not None:
            normalized[canonical_key] = value

    if "base_tokens" in normalized and "labels" in normalized:
        return normalized
    return None


def _build_dataset_from_rows(
    tokenizer: PreTrainedTokenizerBase,
    rows: Sequence[Dict[str, Any]],
) -> Dataset:
    clean_prompts = [_clean_prompt(row) for row in rows]
    if all(
        _first_present(
            row,
            ("counterfactual_input", "source_input", "corrupted_input", "alt_input"),
        )
        is not None
        for row in rows
    ):
        corrupt_prompts = [_corrupt_prompt(row) for row in rows]
    else:
        corrupt_prompts = _generate_corrupt_prompts_from_anchor_prompts(clean_prompts)
    labels = [_encode_single_token_answer(tokenizer, _clean_answer(row)) for row in rows]

    base_tokens, base_last_token_indices = _encode_prompts(tokenizer, clean_prompts)
    source_tokens, source_last_token_indices = _encode_prompts(tokenizer, corrupt_prompts)
    base_position_fields = _compute_prompt_position_fields(
        tokenizer,
        clean_prompts,
        prefix="base_",
        token_batches=base_tokens.tolist(),
    )
    source_position_fields = _compute_prompt_position_fields(
        tokenizer,
        corrupt_prompts,
        prefix="source_",
        token_batches=source_tokens.tolist(),
    )

    dataset_dict = {
        "base_tokens": base_tokens,
        "base_last_token_indices": base_last_token_indices,
        "source_tokens": source_tokens,
        "source_last_token_indices": source_last_token_indices,
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    dataset_dict.update(base_position_fields)
    dataset_dict.update(source_position_fields)

    return Dataset.from_dict(dataset_dict).with_format("torch")


def _build_dataset_from_pretokenized(
    tokenizer: PreTrainedTokenizerBase,
    token_data: Mapping[str, Any],
    num_samples: int,
) -> Dataset:
    required_keys = {"base_tokens", "source_tokens", "labels"}
    missing = required_keys.difference(token_data.keys())
    if missing:
        raise KeyError(
            f"Pretokenized path patching data is missing required keys: {sorted(missing)}"
        )

    base_tokens = _slice_tensor_rows(_as_long_tensor(token_data["base_tokens"]), num_samples)
    source_tokens = _slice_tensor_rows(_as_long_tensor(token_data["source_tokens"]), num_samples)
    labels = _slice_tensor_rows(_as_long_tensor(token_data["labels"]), num_samples)
    labels = _squeeze_label_tensor(labels)

    if base_tokens.ndim == 1:
        base_tokens = base_tokens.unsqueeze(0)
    if source_tokens.ndim == 1:
        source_tokens = source_tokens.unsqueeze(0)

    base_last_token_indices = token_data.get("base_last_token_indices")
    if base_last_token_indices is None:
        base_last_token_indices = _compute_last_token_indices_from_tokens(
            base_tokens,
            tokenizer.pad_token_id,
        )
    else:
        base_last_token_indices = _slice_tensor_rows(
            _as_long_tensor(base_last_token_indices),
            num_samples,
        )

    source_last_token_indices = token_data.get("source_last_token_indices")
    if source_last_token_indices is None:
        source_last_token_indices = _compute_last_token_indices_from_tokens(
            source_tokens,
            tokenizer.pad_token_id,
        )
    else:
        source_last_token_indices = _slice_tensor_rows(
            _as_long_tensor(source_last_token_indices),
            num_samples,
        )

    clean_prompts = [
        _decode_prompt_from_tokens(tokenizer, tokens, int(last_idx.item()))
        for tokens, last_idx in zip(base_tokens, base_last_token_indices)
    ]
    corrupt_prompts = [
        _decode_prompt_from_tokens(tokenizer, tokens, int(last_idx.item()))
        for tokens, last_idx in zip(source_tokens, source_last_token_indices)
    ]
    base_position_fields = _compute_prompt_position_fields(
        tokenizer,
        clean_prompts,
        prefix="base_",
        token_batches=base_tokens.tolist(),
    )
    source_position_fields = _compute_prompt_position_fields(
        tokenizer,
        corrupt_prompts,
        prefix="source_",
        token_batches=source_tokens.tolist(),
    )

    dataset_dict = {
        "base_tokens": base_tokens,
        "base_last_token_indices": base_last_token_indices,
        "source_tokens": source_tokens,
        "source_last_token_indices": source_last_token_indices,
        "labels": labels,
    }
    dataset_dict.update(base_position_fields)
    dataset_dict.update(source_position_fields)

    return Dataset.from_dict(dataset_dict).with_format("torch")


def _build_dataset_from_clean_only_pretokenized(
    tokenizer: PreTrainedTokenizerBase,
    token_data: Mapping[str, Any],
    num_samples: int,
) -> Dataset:
    required_keys = {"base_tokens", "labels"}
    missing = required_keys.difference(token_data.keys())
    if missing:
        raise KeyError(
            f"Clean-only path patching data is missing required keys: {sorted(missing)}"
        )

    base_tokens = _slice_tensor_rows(_as_long_tensor(token_data["base_tokens"]), num_samples)
    labels = _slice_tensor_rows(_as_long_tensor(token_data["labels"]), num_samples)
    labels = _squeeze_label_tensor(labels)

    if base_tokens.ndim == 1:
        base_tokens = base_tokens.unsqueeze(0)

    base_last_token_indices = token_data.get("base_last_token_indices")
    if base_last_token_indices is None:
        base_last_token_indices = _compute_last_token_indices_from_tokens(
            base_tokens,
            tokenizer.pad_token_id,
        )
    else:
        base_last_token_indices = _slice_tensor_rows(
            _as_long_tensor(base_last_token_indices),
            num_samples,
        )

    clean_prompts = [
        _decode_prompt_from_tokens(tokenizer, tokens, int(last_idx.item()))
        for tokens, last_idx in zip(base_tokens, base_last_token_indices)
    ]
    corrupt_prompts = _generate_corrupt_prompts_from_anchor_prompts(clean_prompts)
    source_tokens, source_last_token_indices = _encode_prompts(tokenizer, corrupt_prompts)
    base_position_fields = _compute_prompt_position_fields(
        tokenizer,
        clean_prompts,
        prefix="base_",
        token_batches=base_tokens.tolist(),
    )
    source_position_fields = _compute_prompt_position_fields(
        tokenizer,
        corrupt_prompts,
        prefix="source_",
        token_batches=source_tokens.tolist(),
    )

    dataset_dict = {
        "base_tokens": base_tokens,
        "base_last_token_indices": base_last_token_indices,
        "source_tokens": source_tokens,
        "source_last_token_indices": source_last_token_indices,
        "labels": labels,
    }
    dataset_dict.update(base_position_fields)
    dataset_dict.update(source_position_fields)

    return Dataset.from_dict(dataset_dict).with_format("torch")


def _coerce_in_memory_dataset(
    tokenizer: PreTrainedTokenizerBase,
    data_rows: Any,
    num_samples: int,
):
    prompt_row_keys = {
        "original_input",
        "input",
        "clean_input",
        "base_input",
        "counterfactual_input",
        "source_input",
        "corrupted_input",
        "alt_input",
    }

    if isinstance(data_rows, Dataset):
        sliced = (
            data_rows.select(range(min(num_samples, len(data_rows))))
            if num_samples
            else data_rows
        )
        return _coerce_in_memory_dataset(tokenizer, sliced[:], 0)

    if isinstance(data_rows, Mapping):
        normalized = _normalize_pretokenized_mapping(data_rows)
        if normalized is not None:
            if "source_tokens" in normalized:
                return _build_dataset_from_pretokenized(tokenizer, normalized, num_samples)
            return _build_dataset_from_clean_only_pretokenized(
                tokenizer,
                normalized,
                num_samples,
            )
        if prompt_row_keys.intersection(data_rows.keys()):
            return _build_dataset_from_rows(tokenizer, [dict(data_rows)])

    if isinstance(data_rows, Iterable) and not isinstance(data_rows, (str, bytes)):
        rows = list(data_rows)
        if num_samples:
            rows = rows[:num_samples]
        if not rows:
            raise ValueError("Received empty in-memory training data.")
        first = rows[0]
        if isinstance(first, Mapping):
            normalized_rows = [_normalize_pretokenized_mapping(row) for row in rows]
            if normalized_rows[0] is not None and all(row is not None for row in normalized_rows):
                collated = defaultdict(list)
                for row in normalized_rows:
                    for key, value in row.items():
                        collated[key].append(value)
                collated = dict(collated)
                if "source_tokens" in collated:
                    return _build_dataset_from_pretokenized(tokenizer, collated, 0)
                return _build_dataset_from_clean_only_pretokenized(
                    tokenizer,
                    collated,
                    0,
                )
            return _build_dataset_from_rows(tokenizer, rows)

    raise TypeError(
        "Unsupported in-memory data format. Provide a DataLoader, a HuggingFace Dataset, "
        "a list of row dicts with original prompts, or a dict/list with pretokenized "
        "base fields (and optionally explicit source fields)."
    )
