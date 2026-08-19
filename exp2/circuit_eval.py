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

class SimpleTLUtils:
    @staticmethod
    def get_act_name(component: str, layer_idx: int) -> str:
        component = hook_component_name(component)
        return f"blocks.{layer_idx}.attn.hook_{component}"


@dataclass(frozen=True)
class CircuitComponent:
    stage: str
    position_mode: str
    hook_name: str
    layer: int
    head: int
    score: float


class DictTensorDataset:
    def __init__(self, data: Mapping[str, Any]):
        lengths = {
            key: int(value.shape[0])
            for key, value in data.items()
            if hasattr(value, "shape")
        }
        if not lengths:
            raise ValueError("DictTensorDataset requires at least one tensor field.")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Tensor fields have mismatched lengths: {lengths}")
        self.data = dict(data)
        self.length = next(iter(lengths.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            key: value[idx] if hasattr(value, "__getitem__") else value
            for key, value in self.data.items()
        }


def find_binding_root(start: Optional[Path] = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for base in (start, *start.parents):
        if (base / "coref").is_dir() and (base / "run_path_patching.py").exists():
            return base
        nested = base / "binding-iclr"
        if (nested / "coref").is_dir() and (nested / "run_path_patching.py").exists():
            return nested
    raise FileNotFoundError(
        "Could not find binding-iclr from the current working directory."
    )


def import_binding_modules(binding_root: Path):
    if str(binding_root) not in sys.path:
        sys.path.insert(0, str(binding_root))

    import numpy as np
    import torch
    import transformer_lens.utils as utils
    from torch.utils.data import DataLoader

    import pp_utils
    from datasets import datascience
    from datasets.api import load_vocab
    from datasets.box import Statement

    return {
        "np": np,
        "torch": torch,
        "utils": utils,
        "DataLoader": DataLoader,
        "pp_utils": pp_utils,
        "datascience": datascience,
        "load_vocab": load_vocab,
        "Statement": Statement,
    }


def set_seed(seed: int, torch_module=None, np_module=None) -> None:
    random.seed(seed)
    if np_module is not None:
        np_module.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed_all(seed)


def normalize_device(device: Optional[str], torch_module, pp_utils) -> Any:
    if device is None:
        selected = torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
    elif str(device).isdigit():
        selected = torch_module.device(f"cuda:{device}")
    else:
        selected = torch_module.device(device)
    if selected.type == "cuda" and selected.index is not None:
        torch_module.cuda.set_device(selected)
    pp_utils.device = selected
    return selected


def torch_load_cpu(path: Path):
    import torch

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not hasattr(obj, "detach"):
        raise TypeError(f"Expected a tensor in {path}, found {type(obj).__name__}.")
    return obj.detach().cpu()


def tensor_topk_components(tensor, k: int, *, largest: bool) -> Tuple[List[List[int]], List[float]]:
    k = min(int(k), int(tensor.numel()))
    if k <= 0:
        return [], []
    values, flat_indices = tensor.flatten().topk(k, largest=largest)
    n_heads = int(tensor.shape[1])
    heads = [
        [int(flat_idx // n_heads), int(flat_idx % n_heads)]
        for flat_idx in flat_indices.tolist()
    ]
    return heads, [float(value) for value in values.tolist()]


def all_filtered_components(
    tensor,
    *,
    largest: bool,
    zero_tolerance: float,
) -> Tuple[List[List[int]], List[float]]:
    flat = tensor.flatten()
    if largest:
        mask = flat > zero_tolerance
        order = flat[mask].argsort(descending=True)
    else:
        mask = flat < -zero_tolerance
        order = flat[mask].argsort(descending=False)
    selected_indices = mask.nonzero(as_tuple=False).flatten()[order]
    n_heads = int(tensor.shape[1])
    heads = [
        [int(flat_idx // n_heads), int(flat_idx % n_heads)]
        for flat_idx in selected_indices.tolist()
    ]
    values = [float(flat[int(flat_idx)].item()) for flat_idx in selected_indices.tolist()]
    return heads, values


def hook_component_name(component: str) -> str:
    if component == "o_proj":
        return "z"
    return component


def hook_name(utils, component: str, layer_idx: int) -> str:
    return utils.get_act_name(hook_component_name(component), layer_idx)


def layer_from_hook_name(name: str) -> int:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "blocks" and idx + 1 < len(parts):
            return int(parts[idx + 1])
    raise ValueError(f"Could not infer layer from hook name: {name}")


def span_starts(token_map_entry) -> Any:
    if token_map_entry.ndim != 2 or token_map_entry.shape[1] < 1:
        raise ValueError(
            f"Expected token-map spans [batch, 2], got {tuple(token_map_entry.shape)}."
        )
    return token_map_entry[:, 0].to(dtype=token_map_entry.dtype)


def last_token_indices(tokens, pad_token_id: Optional[int], torch_module):
    if tokens.ndim != 2:
        raise ValueError(f"Expected token tensor [batch, seq], got {tuple(tokens.shape)}.")
    if pad_token_id is None:
        return torch_module.full(
            (tokens.shape[0],), tokens.shape[1] - 1, dtype=torch_module.long
        )
    non_pad = tokens.ne(int(pad_token_id))
    if not bool(non_pad.any(dim=1).all().item()):
        raise ValueError("Found an all-padding prompt.")
    positions_from_end = torch_module.flip(non_pad, dims=[1]).to(torch_module.long).argmax(dim=1)
    return (
        tokens.shape[1] - 1 - positions_from_end
    ).to(torch_module.long)


def last_token_indices_from_attention_mask(attention_mask, torch_module):
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected attention mask [batch, seq], got {tuple(attention_mask.shape)}."
        )
    attended = attention_mask.to(dtype=torch_module.bool)
    if not bool(attended.any(dim=1).all().item()):
        raise ValueError("Found an all-padding prompt.")
    positions_from_end = torch_module.flip(attended, dims=[1]).to(torch_module.long).argmax(dim=1)
    return (
        attention_mask.shape[1] - 1 - positions_from_end
    ).to(torch_module.long)


def infer_prompt_format(model_name: str) -> str:
    resolved_name = MODEL_NAME_BY_FOLDER.get(model_name, model_name)
    model_name_lower = resolved_name.lower()
    if "qwen" in model_name_lower:
        return "qwen-instruct-prefill"
    if "gemma" in model_name_lower and (
        "it" in model_name_lower or "instruct" in model_name_lower
    ):
        return "gemma-it-prefill"
    if "llama" in model_name_lower and (
        "instruct" in model_name_lower
        or "chat" in model_name_lower
        or resolved_name == "llama-8b-hf"
    ):
        return "llama-instruct-prefill"
    return "raw"


def resolve_prompt_format(model_name: str, prompt_format: str) -> str:
    if prompt_format == "auto":
        return infer_prompt_format(model_name)
    return prompt_format


def _apply_chat_template(tokenizer, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not expose apply_chat_template.")
    messages = [{"role": "user", "content": prompt}]
    if SYSTEM_PROMPT.strip():
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _split_assistant_prefill(prompt: str) -> Tuple[str, str]:
    match = ANSWER_PREFILL_RE.search(prompt)
    if match is None:
        return prompt.rstrip(), "Answer:"
    return prompt[: match.start()].rstrip(), prompt[match.start() : match.end()].strip()


def _tokenizer_has_token(tokenizer, token: str) -> bool:
    try:
        return token in tokenizer.get_vocab()
    except Exception:
        return token in getattr(tokenizer, "all_special_tokens", [])


def _format_llama_prefill_fallback(
    tokenizer,
    user_prompt: str,
    assistant_prefill: str,
) -> str:
    if _tokenizer_has_token(tokenizer, "<|start_header_id|>"):
        bos = tokenizer.bos_token or "<|begin_of_text|>"
        return (
            f"{bos}<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_prefill}"
        )
    bos = tokenizer.bos_token or "<s>"
    return (
        f"{bos}[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_prompt} [/INST] {assistant_prefill}"
    )


def _format_llama_prompt_fallback(tokenizer, prompt: str) -> str:
    if _tokenizer_has_token(tokenizer, "<|start_header_id|>"):
        bos = tokenizer.bos_token or "<|begin_of_text|>"
        return (
            f"{bos}<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    bos = tokenizer.bos_token or "<s>"
    return f"{bos}[INST] {prompt} [/INST] "


def _apply_chat_prefill_template(tokenizer, prompt: str) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Tokenizer does not expose apply_chat_template.")
    user_prompt, assistant_prefill = _split_assistant_prefill(prompt)
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_prefill},
    ]
    if SYSTEM_PROMPT.strip():
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        continue_final_message=True,
    )


def format_prompt_for_model(tokenizer, prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt

    if prompt_format == "chat-template":
        return _apply_chat_template(tokenizer, prompt)

    if prompt_format == "chat-prefill":
        return _apply_chat_prefill_template(tokenizer, prompt)

    if prompt_format.endswith("-prefill"):
        try:
            return _apply_chat_prefill_template(tokenizer, prompt)
        except Exception:
            pass
        user_prompt, assistant_prefill = _split_assistant_prefill(prompt)
        if prompt_format == "qwen-instruct-prefill":
            return (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_prefill}"
            )
        if prompt_format == "gemma-it-prefill":
            bos = tokenizer.bos_token or ""
            return (
                f"{bos}<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{user_prompt}"
                f"<end_of_turn>\n<start_of_turn>model\n{assistant_prefill}"
            )
        if prompt_format == "llama-instruct-prefill":
            return _format_llama_prefill_fallback(
                tokenizer,
                user_prompt,
                assistant_prefill,
            )

    try:
        return _apply_chat_template(tokenizer, prompt)
    except Exception:
        pass

    if prompt_format == "qwen-instruct":
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if prompt_format == "gemma-it":
        bos = tokenizer.bos_token or ""
        return f"{bos}<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    if prompt_format == "llama-instruct":
        return _format_llama_prompt_fallback(tokenizer, prompt)

    raise ValueError(f"Unknown prompt format: {prompt_format}")


def decode_prompts_from_tokens(tokenizer, tokens, torch_module) -> List[str]:
    last_indices = last_token_indices(
        tokens,
        getattr(tokenizer, "pad_token_id", None),
        torch_module,
    )
    prompts = []
    for row, last_idx in zip(tokens, last_indices):
        prompts.append(
            tokenizer.decode(
                row[: int(last_idx.item()) + 1].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        )
    return prompts


def encode_prompt_texts(pp_utils, tokenizer, prompts: Sequence[str], torch_module):
    if hasattr(pp_utils, "_ensure_pad_token"):
        pp_utils._ensure_pad_token(tokenizer)
    encoded = tokenizer(
        list(prompts),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=True,
    )
    return (
        encoded["input_ids"].to(torch_module.long),
        encoded["attention_mask"].to(torch_module.long),
    )


def first_answer_token_id(tokenizer, formatted_prompt: str, answer_word: str) -> int:
    return answer_token_ids(tokenizer, formatted_prompt, answer_word)[0]


def answer_token_ids(tokenizer, formatted_prompt: str, answer_word: str) -> List[int]:
    answer_word = answer_word.strip()
    if not answer_word:
        raise ValueError("Cannot encode an empty answer word.")

    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
    suffixes = (
        (answer_word, f" {answer_word}")
        if formatted_prompt[-1:].isspace()
        else (f" {answer_word}", answer_word)
    )
    token_ids_by_priority: List[int] = []
    for suffix in suffixes:
        full_ids = tokenizer(
            formatted_prompt + suffix,
            add_special_tokens=False,
        )["input_ids"]
        if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
            token_ids_by_priority.append(int(full_ids[len(prompt_ids)]))

    for suffix in suffixes:
        fallback_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        if fallback_ids:
            token_ids_by_priority.append(int(fallback_ids[0]))

    deduped: List[int] = []
    seen = set()
    for token_id in token_ids_by_priority:
        if token_id in seen:
            continue
        deduped.append(token_id)
        seen.add(token_id)
    if deduped:
        return deduped

    raise ValueError(f"Could not encode answer {answer_word!r}.")


def padded_token_id_tensor(token_ids_by_row: Sequence[Sequence[int]], torch_module):
    width = max((len(row) for row in token_ids_by_row), default=0)
    if width == 0:
        raise ValueError("Cannot build a padded tensor from empty token-id rows.")
    rows = [
        [int(token_id) for token_id in row] + [-1] * (width - len(row))
        for row in token_ids_by_row
    ]
    return torch_module.tensor(rows, dtype=torch_module.long)


def answer_words_from_tokens(tokenizer, answer_tokens) -> List[List[str]]:
    rows = answer_tokens.detach().cpu().tolist()
    if rows and not isinstance(rows[0], list):
        rows = [[token_id] for token_id in rows]
    return [
        [
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            for token_id in row
        ]
        for row in rows
    ]


def position_fields_from_tokens(
    *,
    pp_utils,
    tokenizer,
    tokens,
    prefix: str,
    torch_module,
    prompts: Optional[Sequence[str]] = None,
    attention_mask=None,
) -> Dict[str, Any]:
    if not hasattr(pp_utils, "_compute_prompt_position_fields"):
        raise AttributeError(
            "pp_utils._compute_prompt_position_fields is required for formatted prompts."
        )
    if prompts is None:
        prompts = decode_prompts_from_tokens(tokenizer, tokens, torch_module)
    token_batches = tokens.tolist()
    if attention_mask is not None:
        token_batches = [
            [int(token_id) for token_id, keep in zip(row, mask_row) if int(keep)]
            for row, mask_row in zip(token_batches, attention_mask.tolist())
        ]
    return pp_utils._compute_prompt_position_fields(
        tokenizer,
        prompts,
        prefix=f"{prefix}_",
        token_batches=token_batches,
    )


def quiet_build_example_refactored(datascience, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return datascience.build_example_refactored(**kwargs)


def paired_contexts(
    *,
    Statement,
    num_entities: int,
    query_name: int,
    swap_pair: Tuple[int, int],
    counterfactual_mode: str,
) -> Tuple[List[Any], List[Any], Dict[str, Dict[int, int]]]:
    swap_a, swap_b = swap_pair
    if query_name not in swap_pair:
        raise ValueError(f"query_name={query_name} must appear in swap_pair={swap_pair}.")

    non_swap = [idx for idx in range(num_entities) if idx not in swap_pair]
    original_attrs = {idx: idx for idx in range(num_entities)}
    if counterfactual_mode == "same_objects":
        counterfactual_attrs = dict(original_attrs)
    elif counterfactual_mode == "new_swap_objects":
        counterfactual_attrs = dict(original_attrs)
        counterfactual_attrs[swap_b] = num_entities
        counterfactual_attrs[swap_a] = num_entities + 1
    else:
        raise ValueError(f"Unknown counterfactual mode: {counterfactual_mode}")

    original = [
        Statement(swap_a, original_attrs[swap_a], "normal"),
        Statement(swap_b, original_attrs[swap_b], "normal"),
        *[Statement(idx, original_attrs[idx], "normal") for idx in non_swap],
        Statement(swap_a, swap_b, "swap"),
    ]
    counterfactual = [
        Statement(swap_b, counterfactual_attrs[swap_b], "normal"),
        Statement(swap_a, counterfactual_attrs[swap_a], "normal"),
        *[Statement(idx, counterfactual_attrs[idx], "normal") for idx in non_swap],
        Statement(swap_b, swap_a, "swap"),
    ]
    return original, counterfactual, {
        "original_attrs": original_attrs,
        "counterfactual_attrs": counterfactual_attrs,
    }


def normal_context_index(context: Sequence[Any], box_idx: int) -> int:
    for idx, statement in enumerate(context):
        if statement.type in ("normal", "ref") and statement.name == box_idx:
            return idx
    raise ValueError(f"Could not find a normal context statement for box {box_idx}.")


def path_patching_contexts(
    *,
    vocab,
    num_entities: int,
    query_name: int,
    swap_pair: Tuple[int, int],
) -> Tuple[List[Any], List[Any]]:
    clean_context = vocab.default_context(
        num_entities=num_entities,
        swap_pair=swap_pair,
        query_name=query_name,
    )
    shifted = list(range(num_entities))
    shifted = shifted[1:] + shifted[:1]
    corrupt_context = vocab.default_context(
        num_entities=num_entities,
        entities=shifted,
        attributes=shifted,
        swap_pair=swap_pair,
        query_name=query_name,
    )
    return clean_context, corrupt_context


def swap_statement(context: Sequence[Any]):
    for statement in context:
        if statement.type == "swap":
            return statement
    raise ValueError("Could not find a swap statement in the context.")


def swap_target_box_from_context(context: Sequence[Any], query_name: int) -> int:
    statement = swap_statement(context)
    left_box, right_box = int(statement.name), int(statement.attr)
    if left_box == query_name:
        return right_box
    if right_box == query_name:
        return left_box
    raise ValueError(f"query_name={query_name} must appear in swap statement {statement}.")


def assign_swap_position_fields(
    data: Dict[str, Any],
    *,
    prefix: str,
    context: Sequence[Any],
    query_name: int,
) -> None:
    statement = swap_statement(context)
    left_box, right_box = int(statement.name), int(statement.attr)
    left_positions = data[f"{prefix}_swap_left_box_positions"]
    right_positions = data[f"{prefix}_swap_right_box_positions"]
    if left_box == query_name:
        data[f"{prefix}_swap_query_box_positions"] = left_positions
        data[f"{prefix}_swap_target_box_positions"] = right_positions
    elif right_box == query_name:
        data[f"{prefix}_swap_query_box_positions"] = right_positions
        data[f"{prefix}_swap_target_box_positions"] = left_positions
    else:
        raise ValueError(f"query_name={query_name} must appear in swap statement {statement}.")


def build_box_dataset(
    *,
    torch_module,
    pp_utils,
    datascience,
    load_vocab,
    Statement,
    tokenizer,
    model_name: str,
    vocab_tag: str,
    vocab_split: str,
    num_entities: int,
    num_samples: int,
    query_name: int,
    prompt_id_start: int,
    swap_pair: Tuple[int, int],
    template_type: str,
    counterfactual_mode: str,
    prompt_format: str,
) -> DictTensorDataset:
    vocab = load_vocab(vocab_tag, model_name, split=vocab_split, tokenizer=tokenizer)
    if counterfactual_mode == "path_patching_shifted":
        clean_context, corrupt_context = path_patching_contexts(
            vocab=vocab,
            num_entities=num_entities,
            query_name=query_name,
            swap_pair=swap_pair,
        )
    else:
        clean_context, corrupt_context, _ = paired_contexts(
            Statement=Statement,
            num_entities=num_entities,
            query_name=query_name,
            swap_pair=swap_pair,
            counterfactual_mode=counterfactual_mode,
        )
    target_box = swap_target_box_from_context(clean_context, query_name)
    template_context = {"query_name": query_name, "raw_query_name": None}

    base_tokens, base_answers, _, base_maps = quiet_build_example_refactored(
        datascience,
        batch_size=num_samples,
        vocab=vocab,
        num_entities=num_entities,
        context=clean_context,
        prompt_id_start=prompt_id_start,
        template_context=template_context,
        template_type=template_type,
    )
    source_tokens, source_answers, _, source_maps = quiet_build_example_refactored(
        datascience,
        batch_size=num_samples,
        vocab=vocab,
        num_entities=num_entities,
        context=corrupt_context,
        prompt_id_start=prompt_id_start,
        template_context=template_context,
        template_type=template_type,
    )

    resolved_prompt_format = resolve_prompt_format(model_name, prompt_format)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if resolved_prompt_format != "raw":
        base_prompts = decode_prompts_from_tokens(tokenizer, base_tokens, torch_module)
        source_prompts = decode_prompts_from_tokens(tokenizer, source_tokens, torch_module)
        formatted_base_prompts = [
            format_prompt_for_model(tokenizer, prompt, resolved_prompt_format)
            for prompt in base_prompts
        ]
        formatted_source_prompts = [
            format_prompt_for_model(tokenizer, prompt, resolved_prompt_format)
            for prompt in source_prompts
        ]
        formatted_base_tokens, formatted_base_attention_mask = encode_prompt_texts(
            pp_utils,
            tokenizer,
            formatted_base_prompts,
            torch_module,
        )
        formatted_source_tokens, formatted_source_attention_mask = encode_prompt_texts(
            pp_utils,
            tokenizer,
            formatted_source_prompts,
            torch_module,
        )
        base_last = last_token_indices_from_attention_mask(
            formatted_base_attention_mask,
            torch_module,
        )
        source_last = last_token_indices_from_attention_mask(
            formatted_source_attention_mask,
            torch_module,
        )

        base_answer_words = answer_words_from_tokens(tokenizer, base_answers)
        source_answer_words = answer_words_from_tokens(tokenizer, source_answers)
        label_alias_rows = [
            answer_token_ids(
                tokenizer,
                prompt,
                base_answer_words[row_idx][query_name],
            )
            for row_idx, prompt in enumerate(formatted_base_prompts)
        ]
        source_label_alias_rows = [
            answer_token_ids(
                tokenizer,
                prompt,
                source_answer_words[row_idx][query_name],
            )
            for row_idx, prompt in enumerate(formatted_source_prompts)
        ]
        labels = torch_module.tensor(
            [row[0] for row in label_alias_rows],
            dtype=torch_module.long,
        )
        source_labels = torch_module.tensor(
            [row[0] for row in source_label_alias_rows],
            dtype=torch_module.long,
        )
        candidate_labels = torch_module.tensor(
            [
                [
                    first_answer_token_id(tokenizer, formatted_base_prompts[row_idx], word)
                    for word in (
                        base_answer_words[row_idx] + source_answer_words[row_idx]
                    )
                ]
                for row_idx in range(num_samples)
            ],
            dtype=torch_module.long,
        )
        candidate_alias_labels = padded_token_id_tensor(
            [
                [
                    token_id
                    for word in (base_answer_words[row_idx] + source_answer_words[row_idx])
                    for token_id in answer_token_ids(
                        tokenizer,
                        formatted_base_prompts[row_idx],
                        word,
                    )
                ]
                for row_idx in range(num_samples)
            ],
            torch_module,
        )

        data = {
            "base_tokens": formatted_base_tokens.to(torch_module.long),
            "source_tokens": formatted_source_tokens.to(torch_module.long),
            "labels": labels,
            "label_aliases": padded_token_id_tensor(label_alias_rows, torch_module),
            "source_labels": source_labels,
            "source_label_aliases": padded_token_id_tensor(
                source_label_alias_rows,
                torch_module,
            ),
            "candidate_labels": candidate_labels,
            "candidate_alias_labels": candidate_alias_labels,
            "base_last_token_indices": base_last,
            "source_last_token_indices": source_last,
            "base_answer_colon_positions": base_last,
            "source_answer_colon_positions": source_last,
        }
        data.update(
            position_fields_from_tokens(
                pp_utils=pp_utils,
                tokenizer=tokenizer,
                tokens=formatted_base_tokens,
                prefix="base",
                torch_module=torch_module,
                prompts=formatted_base_prompts,
                attention_mask=formatted_base_attention_mask,
            )
        )
        data.update(
            position_fields_from_tokens(
                pp_utils=pp_utils,
                tokenizer=tokenizer,
                tokens=formatted_source_tokens,
                prefix="source",
                torch_module=torch_module,
                prompts=formatted_source_prompts,
                attention_mask=formatted_source_attention_mask,
            )
        )
        return DictTensorDataset(data)

    base_last = last_token_indices(base_tokens, pad_token_id, torch_module)
    source_last = last_token_indices(source_tokens, pad_token_id, torch_module)
    swap_idx = num_entities
    base_query_context_idx = normal_context_index(clean_context, query_name)
    base_target_context_idx = normal_context_index(clean_context, target_box)
    source_query_context_idx = normal_context_index(corrupt_context, query_name)
    source_target_context_idx = normal_context_index(corrupt_context, target_box)

    data = {
        "base_tokens": base_tokens.to(torch_module.long),
        "source_tokens": source_tokens.to(torch_module.long),
        "labels": base_answers[:, query_name].to(torch_module.long),
        "source_labels": source_answers[:, query_name].to(torch_module.long),
        "candidate_labels": torch_module.cat([base_answers, source_answers], dim=1).to(
            torch_module.long
        ),
        "base_last_token_indices": base_last,
        "source_last_token_indices": source_last,
        "base_answer_colon_positions": base_last,
        "source_answer_colon_positions": source_last,
        "base_question_query_box_positions": span_starts(base_maps["qn_subject"]).to(
            torch_module.long
        ),
        "source_question_query_box_positions": span_starts(
            source_maps["qn_subject"]
        ).to(torch_module.long),
        "base_context_query_box_positions": span_starts(
            base_maps["context"][base_query_context_idx]["subject"]
        ).to(torch_module.long),
        "source_context_query_box_positions": span_starts(
            source_maps["context"][source_query_context_idx]["subject"]
        ).to(torch_module.long),
        "base_context_target_box_positions": span_starts(
            base_maps["context"][base_target_context_idx]["subject"]
        ).to(torch_module.long),
        "source_context_target_box_positions": span_starts(
            source_maps["context"][source_target_context_idx]["subject"]
        ).to(torch_module.long),
        "base_swap_left_box_positions": span_starts(
            base_maps["context"][swap_idx]["left_box"]
        ).to(torch_module.long),
        "source_swap_left_box_positions": span_starts(
            source_maps["context"][swap_idx]["left_box"]
        ).to(torch_module.long),
        "base_swap_right_box_positions": span_starts(
            base_maps["context"][swap_idx]["right_box"]
        ).to(torch_module.long),
        "source_swap_right_box_positions": span_starts(
            source_maps["context"][swap_idx]["right_box"]
        ).to(torch_module.long),
    }

    assign_swap_position_fields(
        data,
        prefix="base",
        context=clean_context,
        query_name=query_name,
    )
    assign_swap_position_fields(
        data,
        prefix="source",
        context=corrupt_context,
        query_name=query_name,
    )

    return DictTensorDataset(data)


def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def position_tensor(batch: Mapping[str, Any], position_mode: str):
    if position_mode == "last_token":
        return batch["base_last_token_indices"]
    field = f"base_{position_mode}_positions"
    if field not in batch:
        raise KeyError(f"Missing position field {field!r}.")
    return batch[field]


def new_metrics() -> Dict[str, float]:
    return {
        "correct": 0,
        "exact_correct": 0,
        "candidate_correct": 0,
        "total": 0,
        "candidate_total": 0,
        "label_logit_sum": 0.0,
        "label_prob_sum": 0.0,
        "candidate_margin_sum": 0.0,
    }


def _valid_unique_token_ids(token_ids, *, device, fallback: Optional[Sequence[int]] = None):
    import torch

    pieces = []
    if token_ids is not None:
        ids = token_ids.detach().flatten().to(device=device, dtype=torch.long)
        ids = ids[ids >= 0]
        if ids.numel():
            pieces.append(ids)
    if fallback is not None:
        pieces.append(torch.tensor(list(fallback), device=device, dtype=torch.long))
    if not pieces:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.cat(pieces).unique()


def update_metrics(
    metrics: Dict[str, float],
    logits_at_answer,
    label: int,
    candidates,
    *,
    label_aliases=None,
    candidate_aliases=None,
):
    import torch

    device = logits_at_answer.device
    pred = int(logits_at_answer.argmax().item())
    probs = logits_at_answer.softmax(dim=-1)
    label_ids = _valid_unique_token_ids(
        label_aliases,
        device=device,
        fallback=[label],
    )
    label_logits = logits_at_answer.index_select(dim=0, index=label_ids)
    label_score = float(label_logits.max().item())
    label_prob = float(probs.index_select(dim=0, index=label_ids).sum().item())

    metrics["exact_correct"] += int(pred == label)
    metrics["correct"] += int(bool((label_ids == pred).any().item()))
    metrics["total"] += 1
    metrics["label_logit_sum"] += label_score
    metrics["label_prob_sum"] += label_prob

    candidate_source = candidate_aliases if candidate_aliases is not None else candidates
    if candidate_source is None:
        return
    candidate_ids = _valid_unique_token_ids(candidate_source, device=device)
    if candidate_ids.numel() == 0:
        return
    if not bool(torch.isin(label_ids, candidate_ids).all().item()):
        candidate_ids = torch.cat(
            [candidate_ids, label_ids]
        ).unique()
    candidate_logits = logits_at_answer.index_select(dim=0, index=candidate_ids)
    candidate_pred = int(candidate_ids[int(candidate_logits.argmax().item())].item())
    metrics["candidate_correct"] += int(bool((label_ids == candidate_pred).any().item()))
    metrics["candidate_total"] += 1

    other_mask = ~torch.isin(candidate_ids, label_ids)
    if bool(other_mask.any().item()):
        other_logits = logits_at_answer.index_select(dim=0, index=candidate_ids[other_mask])
        margin = float(label_score - other_logits.max().item())
    else:
        margin = float("nan")
    if not math.isnan(margin):
        metrics["candidate_margin_sum"] += margin


def finalize_metrics(metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
    total = int(metrics["total"])
    candidate_total = int(metrics["candidate_total"])
    return {
        "accuracy": metrics["correct"] / total if total else None,
        "full_vocab_accuracy": metrics["correct"] / total if total else None,
        "exact_token_accuracy": metrics["exact_correct"] / total if total else None,
        "candidate_accuracy": (
            metrics["candidate_correct"] / candidate_total if candidate_total else None
        ),
        "correct": int(metrics["correct"]),
        "exact_correct": int(metrics["exact_correct"]),
        "candidate_correct": int(metrics["candidate_correct"]),
        "total": total,
        "candidate_total": candidate_total,
        "mean_label_logit": metrics["label_logit_sum"] / total if total else None,
        "mean_label_prob": metrics["label_prob_sum"] / total if total else None,
        "mean_candidate_margin": (
            metrics["candidate_margin_sum"] / candidate_total if candidate_total else None
        ),
    }


def compute_mean_activations(model, dataloader, hook_names: Sequence[str], device: Any):
    hook_set = set(hook_names)
    mean_activations: Dict[str, Any] = {}
    total = 0
    import torch
    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mean activations"):
            batch = move_batch_to_device(batch, device)
            _, cache = model.run_with_cache(
                batch["base_tokens"],
                names_filter=lambda name: name in hook_set,
                return_type="logits",
            )
            batch_size = int(batch["base_tokens"].shape[0])
            total += batch_size
            for name in hook_names:
                acts = cache[name].detach().to("cpu", dtype=torch.float32)
                summed = acts.sum(dim=0)
                if name in mean_activations:
                    mean_activations[name] += summed
                else:
                    mean_activations[name] = summed
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if total == 0:
        raise ValueError("Cannot compute mean activations from an empty dataloader.")
    for name in hook_names:
        mean_activations[name] /= total
    return mean_activations


def mean_ablate_activation(
    activation,
    hook,
    *,
    batch: Mapping[str, Any],
    circuit_by_position: Mapping[str, Mapping[str, Sequence[int]]],
    mean_activations: Mapping[str, Any],
    ablate_non_circuit_positions: bool,
    preserve_all_heads_at_circuit_positions: bool,
):
    if activation.ndim != 4:
        raise ValueError(
            f"Expected head activation [batch, seq, head, dim], got "
            f"{tuple(activation.shape)} at {hook.name}."
        )
    mean_act = mean_activations[hook.name].to(
        device=activation.device,
        dtype=activation.dtype,
    )
    n_heads = int(activation.shape[2])
    all_heads = set(range(n_heads))

    for batch_idx in range(int(activation.shape[0])):
        pos_to_keep: Dict[int, set] = defaultdict(set)
        for mode, by_hook in circuit_by_position.items():
            heads = by_hook.get(hook.name, ())
            if not heads:
                continue
            pos = int(position_tensor(batch, mode)[batch_idx].item())
            pos_to_keep[pos].update(int(head) for head in heads if int(head) < n_heads)

        last_pos = int(batch["base_last_token_indices"][batch_idx].item())
        for token_pos in range(last_pos + 1):
            if token_pos >= mean_act.shape[0]:
                continue
            keep_heads = pos_to_keep.get(token_pos)
            if keep_heads is None:
                if ablate_non_circuit_positions:
                    activation[batch_idx, token_pos, :, :] = mean_act[token_pos, :, :]
                continue
            if preserve_all_heads_at_circuit_positions:
                continue
            ablate_heads = sorted(all_heads - keep_heads)
            if ablate_heads:
                activation[batch_idx, token_pos, ablate_heads, :] = mean_act[
                    token_pos, ablate_heads, :
                ]
    return activation


def evaluate_metrics(
    *,
    model,
    dataloader,
    device: Any,
    hook_names: Optional[Sequence[str]] = None,
    circuit_by_position: Optional[Mapping[str, Mapping[str, Sequence[int]]]] = None,
    mean_activations: Optional[Mapping[str, Any]] = None,
    ablate_non_circuit_positions: bool = True,
    preserve_all_heads_at_circuit_positions: bool = False,
):
    import torch
    from functools import partial
    from tqdm import tqdm

    metrics = new_metrics()
    use_hooks = bool(hook_names)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            batch = move_batch_to_device(batch, device)
            if use_hooks:
                hooks = [
                    (
                        name,
                        partial(
                            mean_ablate_activation,
                            batch=batch,
                            circuit_by_position=circuit_by_position,
                            mean_activations=mean_activations,
                            ablate_non_circuit_positions=ablate_non_circuit_positions,
                            preserve_all_heads_at_circuit_positions=preserve_all_heads_at_circuit_positions,
                        ),
                    )
                    for name in hook_names
                ]
                logits = model.run_with_hooks(
                    batch["base_tokens"],
                    fwd_hooks=hooks,
                    return_type="logits",
                )
            else:
                logits = model(batch["base_tokens"], return_type="logits")

            for batch_idx in range(int(batch["labels"].shape[0])):
                last_idx = int(batch["base_last_token_indices"][batch_idx].item())
                label = int(batch["labels"][batch_idx].item())
                candidates = (
                    batch["candidate_labels"][batch_idx]
                    if "candidate_labels" in batch
                    else None
                )
                label_aliases = (
                    batch["label_aliases"][batch_idx]
                    if "label_aliases" in batch
                    else None
                )
                candidate_aliases = (
                    batch["candidate_alias_labels"][batch_idx]
                    if "candidate_alias_labels" in batch
                    else None
                )
                update_metrics(
                    metrics,
                    logits[batch_idx, last_idx],
                    label,
                    candidates,
                    label_aliases=label_aliases,
                    candidate_aliases=candidate_aliases,
                )
            del logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return finalize_metrics(metrics)


def metrics_mean(metrics_list: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [m.get(key) for m in metrics_list if m.get(key) is not None]
    return float(statistics.mean(values)) if values else None


def metrics_std(metrics_list: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [m.get(key) for m in metrics_list if m.get(key) is not None]
    return float(statistics.pstdev(values)) if values else None


def selected_stage_heads(
    tensor,
    *,
    k: int,
    circuit_source: str,
    largest: bool,
    include_zero_contribution_heads: bool,
    zero_tolerance: float,
) -> Tuple[List[List[int]], List[float]]:
    if circuit_source == "all-negative":
        return all_filtered_components(
            tensor,
            largest=False,
            zero_tolerance=zero_tolerance,
        )
    if circuit_source == "all-positive":
        return all_filtered_components(
            tensor,
            largest=True,
            zero_tolerance=zero_tolerance,
        )
    heads, values = tensor_topk_components(tensor, k, largest=largest)
    if include_zero_contribution_heads:
        return heads, values
    filtered = [
        (head, value)
        for head, value in zip(heads, values)
        if abs(value) > zero_tolerance
    ]
    return [head for head, _ in filtered], [value for _, value in filtered]


def add_components(
    *,
    components: List[CircuitComponent],
    stage: str,
    heads: Sequence[Sequence[int]],
    values: Sequence[float],
    position_mode: str,
    hook_component: str,
    utils,
) -> None:
    for (layer, head), score in zip(heads, values):
        components.append(
            CircuitComponent(
                stage=stage,
                position_mode=position_mode,
                hook_name=hook_name(utils, hook_component, int(layer)),
                layer=int(layer),
                head=int(head),
                score=float(score),
            )
        )


def append_attention_pattern_context_reader_components(
    *,
    components: List[CircuitComponent],
    selected_heads: Sequence[Sequence[int]],
    selected_scores: Sequence[float],
    position_mode: str,
    hook_component: str,
    utils,
) -> List[CircuitComponent]:
    updated = list(components)
    add_components(
        components=updated,
        stage="struct_reader_context_target_box_to_target_box",
        heads=selected_heads,
        values=selected_scores,
        position_mode=position_mode,
        hook_component=hook_component,
        utils=utils,
    )
    return dedupe_components(updated)


def dedupe_components(components: Sequence[CircuitComponent]) -> List[CircuitComponent]:
    deduped: Dict[Tuple[str, str, int], CircuitComponent] = {}
    for component in components:
        key = (component.position_mode, component.hook_name, component.head)
        previous = deduped.get(key)
        if previous is None or abs(component.score) > abs(previous.score):
            deduped[key] = component
    return list(deduped.values())


def circuit_by_position_from_components(
    components: Sequence[CircuitComponent],
) -> Dict[str, Dict[str, List[int]]]:
    circuit: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for component in components:
        heads = circuit[component.position_mode][component.hook_name]
        if component.head not in heads:
            heads.append(component.head)
    return {
        mode: {hook: sorted(heads) for hook, heads in by_hook.items()}
        for mode, by_hook in circuit.items()
    }


def component_count_by_position(components: Sequence[CircuitComponent]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for component in components:
        counts[component.position_mode] += 1
    return dict(counts)


def slice_dict_tensor_dataset(dataset: DictTensorDataset, limit: int) -> DictTensorDataset:
    limit = max(0, min(int(limit), len(dataset)))
    if limit == 0:
        raise ValueError("Dataset slice limit must keep at least one example.")
    return DictTensorDataset(
        {
            key: value[:limit] if hasattr(value, "__getitem__") else value
            for key, value in dataset.data.items()
        }
    )


def ranges_overlap(start_a: int, len_a: int, start_b: int, len_b: int) -> bool:
    end_a = start_a + len_a
    end_b = start_b + len_b
    return start_a < end_b and start_b < end_a


def compute_context_reader_attention_scores(
    *,
    model,
    dataloader,
    device: Any,
    utils,
    torch_module,
):
    pattern_hook_names = [
        utils.get_act_name("pattern", layer)
        for layer in range(int(model.cfg.n_layers))
    ]
    pattern_hook_set = set(pattern_hook_names)
    score_sums = torch_module.zeros(
        (int(model.cfg.n_layers), int(model.cfg.n_heads)),
        dtype=torch_module.float64,
    )
    total = 0

    from tqdm import tqdm

    with torch_module.no_grad():
        for batch in tqdm(dataloader, desc="Attention pattern scores"):
            batch = move_batch_to_device(batch, device)
            _, cache = model.run_with_cache(
                batch["base_tokens"],
                names_filter=lambda name: name in pattern_hook_set,
                return_type="logits",
            )
            query_positions = batch["base_swap_target_box_positions"].to(
                device=device, dtype=torch_module.long
            )
            key_positions = batch["base_context_target_box_positions"].to(
                device=device, dtype=torch_module.long
            )
            for layer, name in enumerate(pattern_hook_names):
                pattern = cache[name]
                if pattern.ndim != 4:
                    raise ValueError(
                        f"Expected attention pattern [batch, head, query, key], got "
                        f"{tuple(pattern.shape)} at {name}."
                    )
                query_index = query_positions[:, None, None, None].expand(
                    -1, int(pattern.shape[1]), 1, int(pattern.shape[3])
                )
                key_index = key_positions[:, None, None].expand(
                    -1, int(pattern.shape[1]), 1
                )
                at_query = pattern.gather(dim=2, index=query_index).squeeze(dim=2)
                selected = at_query.gather(dim=2, index=key_index).squeeze(dim=2)
                score_sums[layer] += selected.detach().to(
                    "cpu", dtype=torch_module.float64
                ).sum(dim=0)
            total += int(batch["base_tokens"].shape[0])
            del cache
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

    if total == 0:
        raise ValueError("Cannot compute attention pattern scores from an empty dataloader.")
    return (score_sums / total).to(dtype=torch_module.float32)


def update_output_circuit_fields(output: Dict[str, Any], components: Sequence[CircuitComponent]) -> None:
    output["component_count"] = len(components)
    output["component_count_by_position"] = component_count_by_position(components)
    output["components"] = [asdict(component) for component in components]
    output["circuit_by_position"] = circuit_by_position_from_components(components)


def tensor_topk_from_candidates(
    tensor,
    candidates: Sequence[Sequence[int]],
    k: int,
    *,
    largest: bool,
) -> Tuple[List[List[int]], List[float]]:
    if k <= 0 or not candidates:
        return [], []
    scored = []
    for layer, head in candidates:
        scored.append(([int(layer), int(head)], float(tensor[int(layer), int(head)].item())))
    scored.sort(key=lambda item: item[1], reverse=largest)
    selected = scored[: min(k, len(scored))]
    return [head for head, _ in selected], [score for _, score in selected]


def load_circuit_components(
    *,
    results_dir: Path,
    utils,
    hook_component: str,
    circuit_source: str,
    largest: bool,
    include_zero_contribution_heads: bool,
    zero_tolerance: float,
    n_value_fetcher: int,
    n_pos_trans: int,
    n_pos_detect: int,
    n_struct_read: int,
    n_context_reader: int,
    pos_detector_position_mode: str,
    struct_reader_position_mode: str,
    include_context_target_reader_heads: bool,
) -> Tuple[List[CircuitComponent], Dict[str, Any]]:
    tensors = {}
    tensor_summaries = {}
    for stage, filename in STAGE_FILES.items():
        path = results_dir / filename
        if not path.exists():
            if stage == "struct_reader_context_target_box_to_target_box":
                continue
            raise FileNotFoundError(
                f"Missing path-patching score tensor: {path}\n"
                "This means circuit_eval.py is running in circuit-eval mode. "
                "For a full-model-only accuracy check, rerun with --model-only. "
                "For circuit metrics, run run_path_patching.py for this model first."
            )
        tensor = torch_load_cpu(path)
        if tensor.ndim != 2:
            raise ValueError(f"Expected [layer, head] tensor in {path}, got {tuple(tensor.shape)}.")
        tensors[stage] = tensor
        tensor_summaries[stage] = {
            "file": str(path),
            "shape": list(tensor.shape),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "nonzero": int((tensor != 0).sum().item()),
        }

    stage_ks = {
        "value_fetcher": n_value_fetcher,
        "pos_transmitter": n_pos_trans,
        "pos_detector": n_pos_detect,
        "struct_reader_swap_target": n_struct_read,
        "struct_reader_context_target_box_to_target_box": n_context_reader,
    }

    selected: Dict[str, Tuple[List[List[int]], List[float]]] = {}
    for stage, tensor in tensors.items():
        if stage == "struct_reader_context_target_box_to_target_box" and not include_context_target_reader_heads:
            continue
        selected[stage] = selected_stage_heads(
            tensor,
            k=stage_ks[stage],
            circuit_source=circuit_source,
            largest=largest,
            include_zero_contribution_heads=include_zero_contribution_heads,
            zero_tolerance=zero_tolerance,
        )

    if "value_fetcher" in selected and "pos_transmitter" in selected:
        value_heads, value_values = selected["value_fetcher"]
        pos_heads, _ = selected["pos_transmitter"]
        pos_set = {tuple(head) for head in pos_heads}
        filtered = [
            (head, value)
            for head, value in zip(value_heads, value_values)
            if tuple(head) not in pos_set
        ]
        selected["value_fetcher"] = (
            [head for head, _ in filtered],
            [value for _, value in filtered],
        )

    components: List[CircuitComponent] = []
    stage_positions = {
        "value_fetcher": "last_token",
        "pos_transmitter": "last_token",
        "pos_detector": pos_detector_position_mode,
        "struct_reader_swap_target": "swap_target_box",
        "struct_reader_context_target_box_to_target_box": struct_reader_position_mode,
    }
    for stage, (heads, values) in selected.items():
        add_components(
            components=components,
            stage=stage,
            heads=heads,
            values=values,
            position_mode=stage_positions[stage],
            hook_component=hook_component,
            utils=utils,
        )

    components = dedupe_components(components)
    return components, {
        "score_tensors": tensor_summaries,
        "selected_heads": {
            stage: heads for stage, (heads, _) in selected.items()
        },
        "selected_scores": {
            stage: values for stage, (_, values) in selected.items()
        },
    }


def random_components_like(
    *,
    components: Sequence[CircuitComponent],
    model,
    utils,
    hook_component: str,
    rng,
) -> List[CircuitComponent]:
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    all_pairs = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    by_position: Dict[str, int] = defaultdict(int)
    for component in components:
        by_position[component.position_mode] += 1

    random_components: List[CircuitComponent] = []
    for position_mode, count in by_position.items():
        if count > len(all_pairs):
            raise ValueError("Cannot sample more random heads than exist in the model.")
        sampled = rng.sample(all_pairs, count)
        for layer, head in sampled:
            random_components.append(
                CircuitComponent(
                    stage="random",
                    position_mode=position_mode,
                    hook_name=hook_name(utils, hook_component, layer),
                    layer=layer,
                    head=head,
                    score=float("nan"),
                )
            )
    return random_components


def metric_value(metrics: Mapping[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(f"Metric {metric_name!r} is not available in {metrics}.")
    return float(value)


def remove_one_component(
    components: Sequence[CircuitComponent], remove_idx: int
) -> List[CircuitComponent]:
    return [component for idx, component in enumerate(components) if idx != remove_idx]


def component_key(component: CircuitComponent) -> Tuple[str, str, int]:
    return (component.position_mode, component.hook_name, component.head)


def remove_component_set(
    components: Sequence[CircuitComponent],
    removal_set: Sequence[CircuitComponent],
) -> List[CircuitComponent]:
    removal_keys = {component_key(component) for component in removal_set}
    return [
        component
        for component in components
        if component_key(component) not in removal_keys
    ]


def full_model_components_for_positions(
    *,
    components: Sequence[CircuitComponent],
    model,
    utils,
    hook_component: str,
) -> List[CircuitComponent]:
    position_modes = sorted({component.position_mode for component in components})
    full_components: List[CircuitComponent] = []
    for position_mode in position_modes:
        for layer in range(int(model.cfg.n_layers)):
            for head in range(int(model.cfg.n_heads)):
                full_components.append(
                    CircuitComponent(
                        stage="full_model",
                        position_mode=position_mode,
                        hook_name=hook_name(utils, hook_component, layer),
                        layer=layer,
                        head=head,
                        score=float("nan"),
                    )
                )
    return full_components


def sample_component_subset(
    *,
    components: Sequence[CircuitComponent],
    rng,
    removal_probability: float,
) -> List[CircuitComponent]:
    return [
        component
        for component in components
        if rng.random() < removal_probability
    ]


def completeness_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> Dict[str, Optional[float]]:
    selected = [record for record in records if record["kind"] == prefix]
    scores = [float(record["incompleteness_score"]) for record in selected]
    circuit_scores = [float(record["circuit_score"]) for record in selected]
    model_scores = [float(record["model_minus_k_score"]) for record in selected]
    return {
        f"{prefix}_count": len(selected),
        f"{prefix}_incompleteness_mean": (
            float(statistics.mean(scores)) if scores else None
        ),
        f"{prefix}_incompleteness_std": (
            float(statistics.pstdev(scores)) if scores else None
        ),
        f"{prefix}_circuit_score_mean": (
            float(statistics.mean(circuit_scores)) if circuit_scores else None
        ),
        f"{prefix}_model_minus_k_score_mean": (
            float(statistics.mean(model_scores)) if model_scores else None
        ),
    }


def plot_completeness_records(
    *,
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
    figure_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random_records = [record for record in records if record["kind"] == "random"]
    stage_records = [record for record in records if record["kind"] == "stage"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    if random_records:
        ax.scatter(
            [record["circuit_score"] for record in random_records],
            [record["model_minus_k_score"] for record in random_records],
            c="tab:red",
            label="Random set",
            alpha=0.8,
        )

    colors = [
        "tab:green",
        "gold",
        "tab:purple",
        "tab:blue",
        "tab:cyan",
        "tab:brown",
        "tab:pink",
    ]
    for record, color in zip(stage_records, colors):
        ax.scatter(
            record["circuit_score"],
            record["model_minus_k_score"],
            c=color,
            label=record["name"],
            marker="D",
            s=70,
            edgecolor="black",
            linewidth=0.4,
        )

    all_scores = [
        float(record[key])
        for record in records
        for key in ("circuit_score", "model_minus_k_score")
        if record.get(key) is not None
    ]
    bounded_metric = metric_name in {
        "accuracy",
        "full_vocab_accuracy",
        "candidate_accuracy",
        "mean_label_prob",
    }
    if bounded_metric:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    elif all_scores:
        lo, hi = min(all_scores), max(all_scores)
        pad = max((hi - lo) * 0.08, 1e-6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    ax.axline((0, 0), slope=1, color="gainsboro", linestyle="--", label="y = x")
    ax.set_xlabel(f"F(C \\ K), {metric_name}")
    ax.set_ylabel(f"F(M \\ K), {metric_name}")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def resolve_completeness_figure_path(
    *,
    requested_path: Optional[Path],
    results_dir: Path,
    model_folder: str,
    metric_name: str,
) -> Path:
    filename = f"{model_folder}_completeness_{metric_name}.pdf"
    if requested_path is None:
        return results_dir / filename
    path = requested_path if requested_path.is_absolute() else Path.cwd() / requested_path
    if path.suffix:
        return path
    return path / filename


def evaluate_completeness_subset(
    *,
    name: str,
    kind: str,
    removal_set: Sequence[CircuitComponent],
    model,
    dataloader,
    device: Any,
    circuit_components: Sequence[CircuitComponent],
    full_model_components: Sequence[CircuitComponent],
    mean_activations: Mapping[str, Any],
    hook_names: Sequence[str],
    primary_metric: str,
    ablate_non_circuit_positions: bool,
) -> Dict[str, Any]:
    circuit_minus_k = remove_component_set(circuit_components, removal_set)
    model_minus_k = remove_component_set(full_model_components, removal_set)

    circuit_metrics = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position_from_components(circuit_minus_k),
        mean_activations=mean_activations,
        ablate_non_circuit_positions=ablate_non_circuit_positions,
        preserve_all_heads_at_circuit_positions=False,
    )
    model_minus_k_metrics = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position_from_components(model_minus_k),
        mean_activations=mean_activations,
        ablate_non_circuit_positions=False,
        preserve_all_heads_at_circuit_positions=False,
    )
    circuit_score = metric_value(circuit_metrics, primary_metric)
    model_minus_k_score = metric_value(model_minus_k_metrics, primary_metric)
    gap = abs(model_minus_k_score - circuit_score)

    return {
        "name": name,
        "kind": kind,
        "removed_component_count": len(removal_set),
        "removed_components": [asdict(component) for component in removal_set],
        "circuit_component_count": len(circuit_minus_k),
        "full_model_component_count": len(model_minus_k),
        "circuit_metrics": circuit_metrics,
        "model_minus_k_metrics": model_minus_k_metrics,
        "metric_name": primary_metric,
        "circuit_score": circuit_score,
        "model_minus_k_score": model_minus_k_score,
        "incompleteness_score": gap,
        "completeness_score": 1.0 - gap,
    }


def run_completeness_evaluation(
    *,
    model,
    dataloader,
    device: Any,
    components: Sequence[CircuitComponent],
    mean_activations: Mapping[str, Any],
    hook_names: Sequence[str],
    primary_metric: str,
    random_iters: int,
    random_removal_probability: float,
    seed: int,
    ablate_non_circuit_positions: bool,
    utils,
    hook_component: str,
    figure_path: Optional[Path],
    plot_title: str,
) -> Dict[str, Any]:
    if not 0.0 <= random_removal_probability <= 1.0:
        raise ValueError("--completeness-random-removal-probability must be in [0, 1].")

    circuit_components = list(components)
    full_components = full_model_components_for_positions(
        components=circuit_components,
        model=model,
        utils=utils,
        hook_component=hook_component,
    )
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    for idx in range(random_iters):
        removal_set = sample_component_subset(
            components=circuit_components,
            rng=rng,
            removal_probability=random_removal_probability,
        )
        print(
            f"Evaluating completeness random set {idx + 1}/{random_iters}...",
            file=sys.stderr,
        )
        records.append(
            evaluate_completeness_subset(
                name=f"random_{idx}",
                kind="random",
                removal_set=removal_set,
                model=model,
                dataloader=dataloader,
                device=device,
                circuit_components=circuit_components,
                full_model_components=full_components,
                mean_activations=mean_activations,
                hook_names=hook_names,
                primary_metric=primary_metric,
                ablate_non_circuit_positions=ablate_non_circuit_positions,
            )
        )

    components_by_stage: Dict[str, List[CircuitComponent]] = defaultdict(list)
    for component in circuit_components:
        components_by_stage[component.stage].append(component)
    for stage in sorted(components_by_stage):
        print(f"Evaluating completeness stage {stage}...", file=sys.stderr)
        records.append(
            evaluate_completeness_subset(
                name=stage,
                kind="stage",
                removal_set=components_by_stage[stage],
                model=model,
                dataloader=dataloader,
                device=device,
                circuit_components=circuit_components,
                full_model_components=full_components,
                mean_activations=mean_activations,
                hook_names=hook_names,
                primary_metric=primary_metric,
                ablate_non_circuit_positions=ablate_non_circuit_positions,
            )
        )

    result: Dict[str, Any] = {
        "metric_name": primary_metric,
        "random_iters": random_iters,
        "random_removal_probability": random_removal_probability,
        "component_count": len(circuit_components),
        "full_model_component_count": len(full_components),
        "ablate_non_circuit_positions_for_circuit": ablate_non_circuit_positions,
        "ablate_non_circuit_positions_for_model_minus_k": False,
        "summary": {
            **completeness_summary(records, prefix="random"),
            **completeness_summary(records, prefix="stage"),
        },
        "records": records,
    }

    if figure_path is not None:
        plot_completeness_records(
            records=records,
            metric_name=primary_metric,
            figure_path=figure_path,
            title=plot_title,
        )
        result["figure_path"] = str(figure_path)

    return result


def run_minimality_pruning(
    *,
    model,
    dataloader,
    device: Any,
    components: Sequence[CircuitComponent],
    mean_activations: Mapping[str, Any],
    hook_names: Sequence[str],
    primary_metric: str,
    minimality_threshold: float,
    minimality_criterion: str,
    ablate_non_circuit_positions: bool,
    preserve_all_heads_at_circuit_positions: bool,
) -> Tuple[List[CircuitComponent], Dict[str, Any]]:
    remaining = list(components)
    removed = []
    initial_metrics = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position_from_components(remaining),
        mean_activations=mean_activations,
        ablate_non_circuit_positions=ablate_non_circuit_positions,
        preserve_all_heads_at_circuit_positions=preserve_all_heads_at_circuit_positions,
    )
    current_metrics = initial_metrics
    current_score = metric_value(current_metrics, primary_metric)

    idx = 0
    step = 0
    while idx < len(remaining):
        component = remaining[idx]
        trial_components = remove_one_component(remaining, idx)
        trial_metrics = evaluate_metrics(
            model=model,
            dataloader=dataloader,
            device=device,
            hook_names=hook_names,
            circuit_by_position=circuit_by_position_from_components(trial_components),
            mean_activations=mean_activations,
            ablate_non_circuit_positions=ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=preserve_all_heads_at_circuit_positions,
        )
        trial_score = metric_value(trial_metrics, primary_metric)
        contribution = current_score - trial_score
        if minimality_criterion == "retention":
            remove_component = trial_score >= current_score * minimality_threshold
            normalized = (
                contribution / current_score if current_score not in (0.0, -0.0) else math.inf
            )
        elif minimality_criterion == "minimality-py":
            if primary_metric in MINIMALITY_RAW_DROP_METRICS:
                normalized = contribution
            else:
                normalized = (
                    current_score / trial_score - 1.0
                    if trial_score > 0
                    else math.inf
                )
            remove_component = normalized < minimality_threshold
        else:
            raise ValueError(f"Unknown minimality criterion {minimality_criterion!r}.")

        if remove_component:
            removed_component = remaining.pop(idx)
            removed.append(
                {
                    **asdict(removed_component),
                    "step": step,
                    "score_before": current_score,
                    "score_after": trial_score,
                    "contribution": contribution,
                    "normalized_contribution": normalized,
                }
            )
            current_metrics = trial_metrics
            current_score = trial_score
            step += 1
        else:
            idx += 1

    final_metrics = current_metrics
    return remaining, {
        "criterion": minimality_criterion,
        "metric_name": primary_metric,
        "threshold": minimality_threshold,
        "initial_component_count": len(components),
        "removed_component_count": len(removed),
        "final_component_count": len(remaining),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "removed_components": removed,
        "kept_components": [asdict(component) for component in remaining],
    }


def resolve_model_results_dir(
    *,
    binding_root: Path,
    results_dir: Optional[Path],
    model_folder: str,
) -> Path:
    candidates: List[Path] = []
    if results_dir is not None:
        candidates.append(results_dir)
        candidates.append(results_dir / model_folder)
    candidates.extend(
        [
            Path.cwd() / "results" / "path_patching" / model_folder,
            binding_root / "results" / "path_patching" / model_folder,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and any((candidate / filename).exists() for filename in STAGE_FILES.values()):
            return candidate.resolve()
    return candidates[-1].resolve()


def model_name_for_folder(model_folder: str, explicit_model_name: Optional[str]) -> str:
    if explicit_model_name:
        return explicit_model_name
    return MODEL_NAME_BY_FOLDER.get(model_folder, model_folder)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def evaluate_one_model(args, binding_root: Path, model_folder: str) -> Dict[str, Any]:
    model_name = model_name_for_folder(model_folder, args.model_name)
    resolved_prompt_format = resolve_prompt_format(model_name, args.prompt_format)
    results_dir = resolve_model_results_dir(
        binding_root=binding_root,
        results_dir=args.results_dir,
        model_folder=model_folder,
    )
    use_attention_context_reader = (
        args.include_context_target_reader_heads
        and args.context_reader_source == "attention-pattern"
    )
    if args.no_model_eval and use_attention_context_reader:
        raise ValueError(
            "--context-reader-source attention-pattern requires loading the model; "
            "remove --no-model-eval or use --context-reader-source path-patching."
        )
    if args.no_model_eval and args.run_completeness:
        raise ValueError(
            "--run-completeness requires loading and evaluating the model; "
            "remove --no-model-eval."
        )
    if args.no_model_eval:
        utils = SimpleTLUtils()
    else:
        modules = import_binding_modules(binding_root)
        torch_module = modules["torch"]
        np_module = modules["np"]
        utils = modules["utils"]
        DataLoader = modules["DataLoader"]
        pp_utils = modules["pp_utils"]
        datascience = modules["datascience"]
        load_vocab = modules["load_vocab"]
        Statement = modules["Statement"]

        selected_device = normalize_device(args.device, torch_module, pp_utils)
        set_seed(args.seed, torch_module=torch_module, np_module=np_module)

    requested_hook_component = args.hook_component
    hook_component = hook_component_name(requested_hook_component)
    heldout_samples = (
        args.num_samples if args.heldout_samples is None else args.heldout_samples
    )
    heldout_prompt_id_start = (
        args.prompt_id_start + args.num_samples
        if args.heldout_prompt_id_start is None
        else args.heldout_prompt_id_start
    )
    if args.run_heldout:
        if heldout_samples <= 0:
            raise ValueError("--heldout-samples must be positive.")
        if ranges_overlap(
            args.prompt_id_start,
            args.num_samples,
            heldout_prompt_id_start,
            heldout_samples,
        ):
            raise ValueError(
                "Held-out prompt range overlaps the selection/eval range: "
                f"A=[{args.prompt_id_start}, {args.prompt_id_start + args.num_samples}), "
                f"B=[{heldout_prompt_id_start}, {heldout_prompt_id_start + heldout_samples})."
            )
    base_output: Dict[str, Any] = {
        "model_folder": model_folder,
        "model_name": model_name,
        "results_dir": str(results_dir),
        "model_only": bool(args.model_only),
        "circuit_source": args.circuit_source,
        "context_reader_source": args.context_reader_source,
        "hook_component": requested_hook_component,
        "resolved_hook_component": hook_component,
        "num_samples": args.num_samples,
        "mean_samples": args.mean_samples,
        "attention_pattern_samples": args.attention_pattern_samples,
        "completeness_samples": args.completeness_samples,
        "batch_size": args.batch_size,
        "prompt_id_start": args.prompt_id_start,
        "prompt_format": resolved_prompt_format,
        "requested_prompt_format": args.prompt_format,
        "system_prompt": SYSTEM_PROMPT,
        "swap_pair": list(args.swap_pair),
        "counterfactual_mode": args.counterfactual_mode,
        "pos_detector_position_mode": args.pos_detector_position_mode,
        "struct_reader_position_mode": args.struct_reader_position_mode,
        "attention_pattern_position_mode": args.attention_pattern_position_mode,
        "attention_pattern_candidate_source": args.attention_pattern_candidate_source,
        "ablate_non_circuit_positions": args.ablate_non_circuit_positions,
        "preserve_all_heads_at_circuit_positions": args.preserve_all_heads_at_circuit_positions,
        "include_zero_contribution_heads": args.include_zero_contribution_heads,
        "zero_contribution_tolerance": args.zero_contribution_tolerance,
        "include_context_target_reader_heads": args.include_context_target_reader_heads,
        "primary_metric": args.primary_metric,
        "minimality_metric": args.minimality_metric,
        "run_heldout": bool(args.run_heldout),
        "heldout_samples": heldout_samples if args.run_heldout else None,
        "heldout_prompt_id_start": heldout_prompt_id_start if args.run_heldout else None,
    }
    if args.model_only:
        if args.no_model_eval:
            raise ValueError("--model-only requires loading the model.")
        print(f"Loading model {model_name} on {selected_device}...", file=sys.stderr)
        print(f"Using prompt format {resolved_prompt_format}...", file=sys.stderr)
        model, tokenizer = pp_utils.get_model_and_tokenizer(model_name)
        eval_dataset = build_box_dataset(
            torch_module=torch_module,
            pp_utils=pp_utils,
            datascience=datascience,
            load_vocab=load_vocab,
            Statement=Statement,
            tokenizer=tokenizer,
            model_name=model_name,
            vocab_tag=args.vocab_tag,
            vocab_split=args.vocab_split,
            num_entities=args.num_entities,
            num_samples=args.num_samples,
            query_name=args.query_name,
            prompt_id_start=args.prompt_id_start,
            swap_pair=args.swap_pair,
            template_type=args.template_type,
            counterfactual_mode=args.counterfactual_mode,
            prompt_format=resolved_prompt_format,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        print("Evaluating model...", file=sys.stderr)
        output = dict(base_output)
        output["model_metrics"] = evaluate_metrics(
            model=model,
            dataloader=eval_loader,
            device=selected_device,
        )
        if args.run_heldout:
            print("Evaluating held-out model...", file=sys.stderr)
            heldout_dataset = build_box_dataset(
                torch_module=torch_module,
                pp_utils=pp_utils,
                datascience=datascience,
                load_vocab=load_vocab,
                Statement=Statement,
                tokenizer=tokenizer,
                model_name=model_name,
                vocab_tag=args.vocab_tag,
                vocab_split=args.vocab_split,
                num_entities=args.num_entities,
                num_samples=heldout_samples,
                query_name=args.query_name,
                prompt_id_start=heldout_prompt_id_start,
                swap_pair=args.swap_pair,
                template_type=args.template_type,
                counterfactual_mode=args.counterfactual_mode,
                prompt_format=resolved_prompt_format,
            )
            heldout_loader = DataLoader(
                heldout_dataset, batch_size=args.batch_size, shuffle=False
            )
            output["heldout_estimate"] = {
                "prompt_id_start": heldout_prompt_id_start,
                "num_samples": heldout_samples,
                "model_metrics": evaluate_metrics(
                    model=model,
                    dataloader=heldout_loader,
                    device=selected_device,
                ),
            }
        return output

    components, selection_metadata = load_circuit_components(
        results_dir=results_dir,
        utils=utils,
        hook_component=hook_component,
        circuit_source=args.circuit_source,
        largest=args.largest,
        include_zero_contribution_heads=args.include_zero_contribution_heads,
        zero_tolerance=args.zero_contribution_tolerance,
        n_value_fetcher=args.n_value_fetcher,
        n_pos_trans=args.n_pos_trans,
        n_pos_detect=args.n_pos_detect,
        n_struct_read=args.n_struct_read,
        n_context_reader=args.n_context_reader,
        pos_detector_position_mode=args.pos_detector_position_mode,
        struct_reader_position_mode=args.struct_reader_position_mode,
        include_context_target_reader_heads=(
            args.include_context_target_reader_heads
            and args.context_reader_source == "path-patching"
        ),
    )
    circuit_by_position = circuit_by_position_from_components(components)
    hook_names = sorted(
        {
            hook_name(utils, hook_component, layer)
            for layer in range(
                selection_metadata["score_tensors"][
                    next(iter(selection_metadata["score_tensors"]))
                ]["shape"][0]
            )
        }
    )

    output: Dict[str, Any] = {
        **base_output,
        "component_count": len(components),
        "component_count_by_position": component_count_by_position(components),
        "components": [asdict(component) for component in components],
        "circuit_by_position": circuit_by_position,
        **selection_metadata,
    }

    if args.no_model_eval:
        return output

    print(f"Loading model {model_name} on {selected_device}...", file=sys.stderr)
    print(f"Using prompt format {resolved_prompt_format}...", file=sys.stderr)
    model, tokenizer = pp_utils.get_model_and_tokenizer(model_name)
    if hook_component == "result":
        if hasattr(model, "set_use_attn_result"):
            model.set_use_attn_result(True)
        else:
            model.cfg.use_attn_result = True

    eval_dataset = build_box_dataset(
        torch_module=torch_module,
        pp_utils=pp_utils,
        datascience=datascience,
        load_vocab=load_vocab,
        Statement=Statement,
        tokenizer=tokenizer,
        model_name=model_name,
        vocab_tag=args.vocab_tag,
        vocab_split=args.vocab_split,
        num_entities=args.num_entities,
        num_samples=args.num_samples,
        query_name=args.query_name,
        prompt_id_start=args.prompt_id_start,
        swap_pair=args.swap_pair,
        template_type=args.template_type,
        counterfactual_mode=args.counterfactual_mode,
        prompt_format=resolved_prompt_format,
    )
    mean_dataset = build_box_dataset(
        torch_module=torch_module,
        pp_utils=pp_utils,
        datascience=datascience,
        load_vocab=load_vocab,
        Statement=Statement,
        tokenizer=tokenizer,
        model_name=model_name,
        vocab_tag=args.vocab_tag,
        vocab_split=args.vocab_split,
        num_entities=args.num_entities,
        num_samples=args.mean_samples,
        query_name=args.query_name,
        prompt_id_start=args.prompt_id_start,
        swap_pair=args.swap_pair,
        template_type=args.template_type,
        counterfactual_mode=args.counterfactual_mode,
        prompt_format=resolved_prompt_format,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    mean_loader = DataLoader(mean_dataset, batch_size=args.batch_size, shuffle=False)

    if use_attention_context_reader:
        attention_samples = min(args.attention_pattern_samples, len(eval_dataset))
        attention_dataset = slice_dict_tensor_dataset(eval_dataset, attention_samples)
        attention_loader = DataLoader(
            attention_dataset, batch_size=args.batch_size, shuffle=False
        )
        print(
            "Selecting context-reader heads from attention patterns...",
            file=sys.stderr,
        )
        attention_scores = compute_context_reader_attention_scores(
            model=model,
            dataloader=attention_loader,
            device=selected_device,
            utils=utils,
            torch_module=torch_module,
        )
        if args.attention_pattern_candidate_source == "struct-reader-swap-target":
            candidate_heads = selection_metadata["selected_heads"].get(
                "struct_reader_swap_target", []
            )
            context_heads, context_scores = tensor_topk_from_candidates(
                attention_scores,
                candidate_heads,
                args.n_context_reader,
                largest=True,
            )
        else:
            context_heads, context_scores = tensor_topk_components(
                attention_scores,
                args.n_context_reader,
                largest=True,
            )
        if not args.include_zero_contribution_heads:
            filtered = [
                (head, score)
                for head, score in zip(context_heads, context_scores)
                if score > args.zero_contribution_tolerance
            ]
            context_heads = [head for head, _ in filtered]
            context_scores = [score for _, score in filtered]
        components = append_attention_pattern_context_reader_components(
            components=components,
            selected_heads=context_heads,
            selected_scores=context_scores,
            position_mode=args.attention_pattern_position_mode,
            hook_component=hook_component,
            utils=utils,
        )
        circuit_by_position = circuit_by_position_from_components(components)
        selection_metadata["selected_heads"][
            "struct_reader_context_target_box_to_target_box"
        ] = context_heads
        selection_metadata["selected_scores"][
            "struct_reader_context_target_box_to_target_box"
        ] = context_scores
        selection_metadata["attention_pattern_context_reader"] = {
            "samples": attention_samples,
            "query_position_field": "base_swap_target_box_positions",
            "key_position_field": "base_context_target_box_positions",
            "component_position_mode": args.attention_pattern_position_mode,
            "candidate_source": args.attention_pattern_candidate_source,
            "score_shape": list(attention_scores.shape),
            "score_min": float(attention_scores.min().item()),
            "score_max": float(attention_scores.max().item()),
            "selected_heads": context_heads,
            "selected_scores": context_scores,
        }
        output["selected_heads"] = selection_metadata["selected_heads"]
        output["selected_scores"] = selection_metadata["selected_scores"]
        output["attention_pattern_context_reader"] = selection_metadata[
            "attention_pattern_context_reader"
        ]
        update_output_circuit_fields(output, components)

    print("Computing mean activations...", file=sys.stderr)
    mean_activations = compute_mean_activations(
        model,
        mean_loader,
        hook_names,
        selected_device,
    )

    print("Evaluating model...", file=sys.stderr)
    model_metrics = evaluate_metrics(
        model=model,
        dataloader=eval_loader,
        device=selected_device,
    )
    print("Evaluating circuit...", file=sys.stderr)
    circuit_metrics = evaluate_metrics(
        model=model,
        dataloader=eval_loader,
        device=selected_device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position,
        mean_activations=mean_activations,
        ablate_non_circuit_positions=args.ablate_non_circuit_positions,
        preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
    )
    output["model_metrics"] = model_metrics
    output["circuit_metrics"] = circuit_metrics

    final_components = components
    if args.run_minimality:
        minimality_samples = min(args.minimality_samples, len(eval_dataset))
        minimality_dataset = slice_dict_tensor_dataset(eval_dataset, minimality_samples)
        minimality_loader = DataLoader(
            minimality_dataset, batch_size=args.batch_size, shuffle=False
        )
        print("Running minimality pruning...", file=sys.stderr)
        final_components, minimality_result = run_minimality_pruning(
            model=model,
            dataloader=minimality_loader,
            device=selected_device,
            components=components,
            mean_activations=mean_activations,
            hook_names=hook_names,
            primary_metric=args.minimality_metric,
            minimality_threshold=args.minimality_threshold,
            minimality_criterion=args.minimality_criterion,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
        )
        output["minimality_pruning"] = minimality_result
        output["minimal_circuit_by_position"] = circuit_by_position_from_components(
            final_components
        )
        print("Evaluating minimal circuit on full eval set...", file=sys.stderr)
        output["minimal_circuit_metrics"] = evaluate_metrics(
            model=model,
            dataloader=eval_loader,
            device=selected_device,
            hook_names=hook_names,
            circuit_by_position=circuit_by_position_from_components(final_components),
            mean_activations=mean_activations,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
        )

    if args.run_heldout:
        print("Building held-out eval set...", file=sys.stderr)
        heldout_dataset = build_box_dataset(
            torch_module=torch_module,
            pp_utils=pp_utils,
            datascience=datascience,
            load_vocab=load_vocab,
            Statement=Statement,
            tokenizer=tokenizer,
            model_name=model_name,
            vocab_tag=args.vocab_tag,
            vocab_split=args.vocab_split,
            num_entities=args.num_entities,
            num_samples=heldout_samples,
            query_name=args.query_name,
            prompt_id_start=heldout_prompt_id_start,
            swap_pair=args.swap_pair,
            template_type=args.template_type,
            counterfactual_mode=args.counterfactual_mode,
            prompt_format=resolved_prompt_format,
        )
        heldout_loader = DataLoader(
            heldout_dataset, batch_size=args.batch_size, shuffle=False
        )
        print("Evaluating held-out model...", file=sys.stderr)
        heldout_model_metrics = evaluate_metrics(
            model=model,
            dataloader=heldout_loader,
            device=selected_device,
        )
        print("Evaluating held-out circuit...", file=sys.stderr)
        heldout_circuit_metrics = evaluate_metrics(
            model=model,
            dataloader=heldout_loader,
            device=selected_device,
            hook_names=hook_names,
            circuit_by_position=circuit_by_position,
            mean_activations=mean_activations,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
        )
        heldout_result: Dict[str, Any] = {
            "prompt_id_start": heldout_prompt_id_start,
            "num_samples": heldout_samples,
            "selection_prompt_id_start": args.prompt_id_start,
            "selection_num_samples": args.num_samples,
            "mean_prompt_id_start": args.prompt_id_start,
            "mean_samples": args.mean_samples,
            "component_set": (
                "minimality_pruning.kept_components"
                if args.run_minimality
                else "components"
            ),
            "component_count": len(final_components),
            "metric_name": args.primary_metric,
            "model_metrics": heldout_model_metrics,
            "circuit_metrics": heldout_circuit_metrics,
        }
        heldout_final_metrics = heldout_circuit_metrics
        if args.run_minimality:
            print("Evaluating held-out minimal circuit...", file=sys.stderr)
            heldout_final_metrics = evaluate_metrics(
                model=model,
                dataloader=heldout_loader,
                device=selected_device,
                hook_names=hook_names,
                circuit_by_position=circuit_by_position_from_components(final_components),
                mean_activations=mean_activations,
                ablate_non_circuit_positions=args.ablate_non_circuit_positions,
                preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
            )
            heldout_result["minimal_circuit_metrics"] = heldout_final_metrics
        heldout_result["final_circuit_metrics"] = heldout_final_metrics
        heldout_model_primary = metric_value(heldout_model_metrics, args.primary_metric)
        heldout_circuit_primary = metric_value(heldout_final_metrics, args.primary_metric)
        heldout_result["faithfulness_ratio_circuit_over_model"] = (
            heldout_circuit_primary / heldout_model_primary
            if heldout_model_primary
            else None
        )
        output["heldout_estimate"] = heldout_result

    if args.run_completeness:
        completeness_samples = min(args.completeness_samples, len(eval_dataset))
        completeness_dataset = slice_dict_tensor_dataset(
            eval_dataset,
            completeness_samples,
        )
        completeness_loader = DataLoader(
            completeness_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        completeness_figure_path = None
        if args.completeness_plot:
            completeness_figure_path = resolve_completeness_figure_path(
                requested_path=args.completeness_figure,
                results_dir=results_dir,
                model_folder=model_folder,
                metric_name=args.primary_metric,
            )
        print("Running completeness evaluation...", file=sys.stderr)
        output["completeness"] = run_completeness_evaluation(
            model=model,
            dataloader=completeness_loader,
            device=selected_device,
            components=final_components,
            mean_activations=mean_activations,
            hook_names=hook_names,
            primary_metric=args.primary_metric,
            random_iters=args.completeness_random_iters,
            random_removal_probability=args.completeness_random_removal_probability,
            seed=args.seed + 2,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            utils=utils,
            hook_component=hook_component,
            figure_path=completeness_figure_path,
            plot_title=(
                f"{model_folder} circuit completeness "
                f"({args.primary_metric})"
            ),
        )

    if args.random_circuit_iters > 0:
        rng = random.Random(args.seed + 1)
        random_metrics = []
        for idx in range(args.random_circuit_iters):
            print(f"Evaluating random circuit {idx + 1}/{args.random_circuit_iters}...", file=sys.stderr)
            random_components = random_components_like(
                components=final_components,
                model=model,
                utils=utils,
                hook_component=hook_component,
                rng=rng,
            )
            random_metrics.append(
                evaluate_metrics(
                    model=model,
                    dataloader=eval_loader,
                    device=selected_device,
                    hook_names=hook_names,
                    circuit_by_position=circuit_by_position_from_components(
                        random_components
                    ),
                    mean_activations=mean_activations,
                    ablate_non_circuit_positions=args.ablate_non_circuit_positions,
                    preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
                )
            )
        output["random_circuit_metrics"] = random_metrics
        output["random_circuit_metric_mean"] = metrics_mean(
            random_metrics, args.primary_metric
        )
        output["random_circuit_metric_std"] = metrics_std(
            random_metrics, args.primary_metric
        )

    model_primary = metric_value(model_metrics, args.primary_metric)
    circuit_primary = metric_value(
        output.get("minimal_circuit_metrics", circuit_metrics), args.primary_metric
    )
    output["faithfulness_ratio_circuit_over_model"] = (
        circuit_primary / model_primary if model_primary else None
    )
    random_mean = output.get("random_circuit_metric_mean")
    if random_mean is not None:
        denom = model_primary - random_mean
        output["faithfulness_over_random"] = (
            (circuit_primary - random_mean) / denom if denom else None
        )

    return output


def parse_swap_pair(value: str) -> Tuple[int, int]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected a pair like 1,0.")
    try:
        first, second = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Swap pair entries must be integers.") from exc
    if first == second:
        raise argparse.ArgumentTypeError("Swap pair entries must differ.")
    return first, second


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BOXES circuits recovered from path-patching score tensors."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma9b"],
        help="Result subfolders to evaluate, e.g. gemma9b llama8b qwen7b.",
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        default=None,
        help="Override model name when evaluating exactly one --models entry.",
    )
    parser.add_argument(
        "--prompt-format",
        "--prompt_format",
        choices=PROMPT_FORMATS,
        default="raw",
        help=(
            "Prompt wrapper for generated BOXES eval data. Use the same value "
            "as run_path_patching.py. `auto` uses assistant-prefill chat prompts "
            "for instruct models, keeping `Answer:` in the assistant message so "
            "the next token is the object."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        "--system_prompt",
        default=SYSTEM_PROMPT,
        help=(
            "System instruction used for chat-formatted prompts. Pass an empty "
            "string to omit the system role when the tokenizer chat template is used."
        ),
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        type=Path,
        default=None,
        help="Path to a model result folder or to the parent results/path_patching folder.",
    )
    parser.add_argument(
        "--output-json",
        "--output_json",
        type=Path,
        default=None,
        help="Where to write metrics JSON. Defaults inside each model results folder.",
    )
    parser.add_argument("--num-samples", "--num_samples", type=int, default=300)
    parser.add_argument("--mean-samples", "--mean_samples", type=int, default=100)
    parser.add_argument(
        "--run-heldout",
        "--run_heldout",
        action="store_true",
        help=(
            "After circuit selection/pruning on the main prompt range, evaluate "
            "the frozen circuit on a disjoint held-out prompt range."
        ),
    )
    parser.add_argument(
        "--heldout-samples",
        "--heldout_samples",
        type=int,
        default=None,
        help="Held-out eval sample count. Defaults to --num-samples.",
    )
    parser.add_argument(
        "--heldout-prompt-id-start",
        "--heldout_prompt_id_start",
        type=int,
        default=None,
        help=(
            "First prompt id for held-out eval. Defaults to "
            "--prompt-id-start + --num-samples."
        ),
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument(
        "--random-circuit-iters",
        "--random_circuit_iters",
        type=int,
        default=10,
    )
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vocab-tag", "--vocab_tag", default="BOXES")
    parser.add_argument("--vocab-split", "--vocab_split", default="train")
    parser.add_argument("--num-entities", "--num_entities", type=int, default=3)
    parser.add_argument("--query-name", "--query_name", type=int, default=0)
    parser.add_argument("--prompt-id-start", "--prompt_id_start", type=int, default=0)
    parser.add_argument("--swap-pair", "--swap_pair", type=parse_swap_pair, default=(1, 0))
    parser.add_argument("--template-type", "--template_type", default="normal")
    parser.add_argument(
        "--counterfactual-mode",
        "--counterfactual_mode",
        choices=["path_patching_shifted", "same_objects", "new_swap_objects"],
        default="path_patching_shifted",
        help=(
            "How to build source prompts. The default matches "
            "run_path_patching.py: vocab.default_context with shifted "
            "entities/attributes."
        ),
    )
    parser.add_argument(
        "--hook-component",
        "--hook_component",
        choices=["z", "result", "o_proj"],
        default="o_proj",
        help=(
            "Activation hook to mean-ablate. `o_proj` means the attention "
            "output-projection input, which maps to TransformerLens `hook_z`."
        ),
    )
    parser.add_argument(
        "--circuit-source",
        "--circuit_source",
        choices=["topk", "all-negative", "all-positive"],
        default="topk",
    )
    parser.add_argument(
        "--context-reader-source",
        "--context_reader_source",
        choices=["path-patching", "attention-pattern"],
        default="path-patching",
        help=(
            "How to select struct_reader_context_target_box_to_target_box heads. "
            "attention-pattern ranks heads by average attention from the swap-target "
            "box token to the context-target box token."
        ),
    )
    parser.add_argument(
        "--attention-pattern-samples",
        "--attention_pattern_samples",
        type=int,
        default=100,
        help=(
            "Number of eval examples used when --context-reader-source "
            "attention-pattern."
        ),
    )
    parser.add_argument(
        "--attention-pattern-position-mode",
        "--attention_pattern_position_mode",
        default="swap_target_box",
        help=(
            "Position where attention-pattern context-reader heads are preserved. "
            "For swap_target_box -> context_target_box attention, the useful head "
            "output is at the query position, so the default is swap_target_box."
        ),
    )
    parser.add_argument(
        "--attention-pattern-candidate-source",
        "--attention_pattern_candidate_source",
        choices=["all", "struct-reader-swap-target"],
        default="all",
        help=(
            "Candidate heads to rank by attention pattern. Use all for a direct "
            "attention scan, or struct-reader-swap-target to only rank the "
            "path-patched swap-target reader heads."
        ),
    )
    parser.add_argument("--largest", action="store_true", help="Use largest top-k scores.")
    parser.add_argument("--n-value-fetcher", "--n_value_fetcher", type=int, default=50)
    parser.add_argument("--n-pos-trans", "--n_pos_trans", type=int, default=50)
    parser.add_argument("--n-pos-detect", "--n_pos_detect", type=int, default=25)
    parser.add_argument("--n-struct-read", "--n_struct_read", type=int, default=10)
    parser.add_argument("--n-context-reader", "--n_context_reader", type=int, default=10)
    parser.add_argument(
        "--pos-detector-position-mode",
        "--pos_detector_position_mode",
        default="question_query_box",
    )
    parser.add_argument(
        "--struct-reader-position-mode",
        "--struct_reader_position_mode",
        default="context_target_box",
    )
    parser.add_argument(
        "--include-context-target-reader-heads",
        "--include_context_target_reader_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-zero-contribution-heads",
        "--include_zero_contribution_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--zero-contribution-tolerance",
        "--zero_contribution_tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--ablate-non-circuit-positions",
        "--ablate_non_circuit_positions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preserve-all-heads-at-circuit-positions",
        "--preserve_all_heads_at_circuit_positions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--primary-metric",
        "--primary_metric",
        default="candidate_accuracy",
        choices=METRIC_CHOICES,
    )
    parser.add_argument("--run-minimality", "--run_minimality", action="store_true")
    parser.add_argument("--minimality-samples", "--minimality_samples", type=int, default=20)
    parser.add_argument(
        "--minimality-metric",
        "--minimality_metric",
        default="candidate_accuracy",
        choices=METRIC_CHOICES,
        help=(
            "Metric used for minimality pruning. Defaults to mean_label_logit "
            "so pruning is based on a continuous score instead of candidate_accuracy."
        ),
    )
    parser.add_argument(
        "--minimality-threshold",
        "--minimality_threshold",
        type=float,
        default=0.01,
        help=(
            "For minimality-py, remove a head when the metric drop is below "
            "this value. Accuracy/probability metrics use before/after - 1; "
            "logit/margin metrics use raw before - after. For retention, "
            "remove a head if the metric stays above threshold * current_metric."
        ),
    )
    parser.add_argument(
        "--minimality-criterion",
        "--minimality_criterion",
        choices=["minimality-py", "retention"],
        default="minimality-py",
    )
    parser.add_argument(
        "--run-completeness",
        "--run_completeness",
        action="store_true",
        help=(
            "Run completeness tests: remove K from the circuit and compare "
            "F(C \\ K) against F(M \\ K)."
        ),
    )
    parser.add_argument(
        "--completeness-samples",
        "--completeness_samples",
        type=int,
        default=20,
        help="Number of eval examples used for completeness tests.",
    )
    parser.add_argument(
        "--completeness-random-iters",
        "--completeness_random_iters",
        type=int,
        default=20,
        help="Number of random K subsets for completeness tests.",
    )
    parser.add_argument(
        "--completeness-random-removal-probability",
        "--completeness_random_removal_probability",
        type=float,
        default=0.5,
        help="Probability of including each circuit component in random K.",
    )
    parser.add_argument(
        "--completeness-plot",
        "--completeness_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a completeness scatter plot when --run-completeness is set.",
    )
    parser.add_argument(
        "--completeness-figure",
        "--completeness_figure",
        type=Path,
        default=None,
        help=(
            "Completeness plot path. With multiple models, pass a directory; "
            "defaults inside each model results folder."
        ),
    )
    parser.add_argument(
        "--no-model-eval",
        action="store_true",
        help="Only recover circuit heads and write JSON; do not load the model.",
    )
    parser.add_argument(
        "--model-only",
        "--model_only",
        action="store_true",
        help=(
            "Evaluate only the full model on generated BOXES prompts; skip all "
            "path-patching score tensors, mean activations, and circuit metrics."
        ),
    )
    return parser


def main() -> None:
    global SYSTEM_PROMPT
    args = build_argparser().parse_args()
    SYSTEM_PROMPT = args.system_prompt
    binding_root = find_binding_root()
    if args.model_name and len(args.models) != 1:
        raise ValueError("--model-name can only be used with a single --models entry.")
    if args.model_only and args.no_model_eval:
        raise ValueError("--model-only cannot be combined with --no-model-eval.")
    if args.run_heldout and args.no_model_eval:
        raise ValueError("--run-heldout requires model evaluation.")
    if args.model_only and (args.run_minimality or args.run_completeness):
        raise ValueError(
            "--model-only cannot be combined with --run-minimality or --run-completeness."
        )
    if args.completeness_samples <= 0:
        raise ValueError("--completeness-samples must be positive.")
    if args.completeness_random_iters < 0:
        raise ValueError("--completeness-random-iters cannot be negative.")
    if not 0.0 <= args.completeness_random_removal_probability <= 1.0:
        raise ValueError(
            "--completeness-random-removal-probability must be in [0, 1]."
        )
    if (
        args.completeness_figure is not None
        and args.completeness_figure.suffix
        and len(args.models) > 1
    ):
        raise ValueError(
            "--completeness-figure as a file can only be used with one model; "
            "pass a directory when evaluating multiple models."
        )

    results = [
        evaluate_one_model(args, binding_root=binding_root, model_folder=model_folder)
        for model_folder in args.models
    ]
    payload: Any = results[0] if len(results) == 1 else results

    if args.output_json is not None:
        output_path = args.output_json
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        write_json(output_path, payload)
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        for result in results:
            output_path = Path(result["results_dir"]) / "circuit_eval_metrics.json"
            write_json(output_path, result)
            print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
