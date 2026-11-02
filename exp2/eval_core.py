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
        if (base / "coref").is_dir() and (base / "exp2" / "pp_utils.py").exists():
            return base
        nested = base / "dynamicET"
        if (nested / "coref").is_dir() and (
            nested / "exp2" / "pp_utils.py"
        ).exists():
            return nested
    raise FileNotFoundError(
        "Could not find dynamicET with its integrated coref package from the "
        "current working directory."
    )


def import_binding_modules(binding_root: Path):
    if str(binding_root) not in sys.path:
        sys.path.insert(0, str(binding_root))

    import numpy as np
    import torch
    import transformer_lens.utils as utils
    from torch.utils.data import DataLoader

    from exp2 import pp_utils
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
    pp_utils.set_device(selected)
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
