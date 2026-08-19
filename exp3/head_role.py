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

from datasets import datascience
from datasets.api import load_vocab
from datasets.box import Statement
from exp2.pp_utils import compute_topk_components, get_model_and_tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "path_patching"

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


class DictTensorDataset(Dataset):
    def __init__(self, data: Mapping[str, torch.Tensor]):
        lengths = {
            key: int(value.shape[0])
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
        }
        if not lengths:
            raise ValueError("DictTensorDataset requires at least one tensor field.")
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Tensor fields have mismatched first dimensions: {lengths}")
        self.data = dict(data)
        self.length = next(iter(lengths.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            key: value[idx] if isinstance(value, torch.Tensor) else value[idx]
            for key, value in self.data.items()
        }


def parse_swap_pair(value: str) -> Tuple[int, int]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected a swap pair like '0,1'.")
    try:
        first, second = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Swap pair entries must be integers.") from exc
    if first == second:
        raise argparse.ArgumentTypeError("Swap pair entries must be different.")
    return first, second


def normalize_device(device: Optional[str]) -> torch.device:
    if device is None:
        selected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device.isdigit():
        selected = torch.device(f"cuda:{device}")
    else:
        selected = torch.device(device)

    if selected.type == "cuda" and selected.index is not None:
        torch.cuda.set_device(selected)
    pp_utils.device = selected
    return selected


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_cpu(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    return tensor.detach().cpu()


def resolve_score_path(results_dir: Path, score_file: str) -> Path:
    path = Path(score_file)
    if path.suffix == "":
        path = path.with_suffix(".pt")
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (results_dir / path).resolve()


def safe_path_stem(value: str) -> str:
    stem = Path(value).stem
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return safe.strip("._") or "score"


def default_output_path_for_score(score_file: str) -> Path:
    return Path(
        "./results/path_patching/"
        f"head_interchange_intervention_{safe_path_stem(score_file)}.json"
    )


def select_heads(score_path: Path, top_k: int, largest: bool) -> List[Tuple[int, int]]:
    scores = torch_load_cpu(score_path)
    if scores.ndim != 2:
        raise ValueError(
            f"Expected a [layer, head] score tensor in {score_path}, got {tuple(scores.shape)}."
        )
    k = min(top_k, int(scores.numel()))
    return [
        (int(layer), int(head))
        for layer, head in compute_topk_components(scores, k=k, largest=largest)
    ]


def normalize_component_site(position_mode: str) -> str:
    return POSITION_SITE_ALIASES.get(position_mode, position_mode)


def load_component_dicts(path: Path, component_set: str) -> Tuple[List[Dict[str, object]], str, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    selected_set = component_set
    if component_set == "auto":
        if payload.get("minimality_pruning", {}).get("kept_components"):
            selected_set = "minimality_kept"
        else:
            selected_set = "components"

    if selected_set == "minimality_kept":
        components = payload.get("minimality_pruning", {}).get("kept_components")
        if components is None:
            raise ValueError(
                f"{path} does not contain minimality_pruning.kept_components."
            )
    elif selected_set == "components":
        components = payload.get("components")
        if components is None:
            raise ValueError(f"{path} does not contain a top-level components list.")
    else:
        raise ValueError(f"Unknown component set: {component_set}")

    if not isinstance(components, list):
        raise ValueError(f"Expected component list in {path}, got {type(components)!r}.")
    return components, selected_set, payload


def filter_components_by_stage(
    components: Sequence[Dict[str, object]],
    stages: Optional[Sequence[str]],
) -> List[Dict[str, object]]:
    if stages is None:
        return list(components)
    stage_set = set(stages)
    return [
        component
        for component in components
        if str(component.get("stage")) in stage_set
    ]


def role_aligned_hook_name(component: Mapping[str, object]) -> str:
    recorded_hook = str(component["hook_name"])
    stage = str(component.get("stage"))
    layer_value = component.get("layer")
    layer_idx = (
        int(layer_value)
        if layer_value is not None
        else layer_from_hook_name(recorded_hook)
    )
    if stage in PAYLOAD_STAGES:
        return hook_name("o_proj", layer_idx)
    if stage in POINTER_STAGES:
        return hook_name("pattern", layer_idx)
    return recorded_hook


def component_hook_name(
    component: Mapping[str, object],
    component_hook_mode: str,
) -> str:
    if component_hook_mode == "recorded":
        return str(component["hook_name"])
    if component_hook_mode == "role_aligned":
        return role_aligned_hook_name(component)
    raise ValueError(f"Unknown component hook mode: {component_hook_mode}")


def component_count_by_stage(
    components: Sequence[Mapping[str, object]],
) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for component in components:
        counts[str(component.get("stage", "<missing>"))] += 1
    return dict(counts)


def component_heads_by_site_hook(
    components: Sequence[Mapping[str, object]],
    component_hook_mode: str = "recorded",
) -> Dict[str, Dict[str, List[int]]]:
    heads: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for component in components:
        position_mode = str(component["position_mode"])
        site = normalize_component_site(position_mode)
        if site not in SITE_CHOICES:
            raise ValueError(
                f"Component position_mode {position_mode!r} maps to unsupported "
                f"site {site!r}. Supported sites: {', '.join(SITE_CHOICES)}."
            )
        hook = component_hook_name(component, component_hook_mode)
        heads[site][hook].add(int(component["head"]))

    return {
        site: {hook: sorted(hook_heads) for hook, hook_heads in by_hook.items()}
        for site, by_hook in heads.items()
    }


def select_site_heads(
    all_heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    requested_sites: Optional[Sequence[str]],
    *,
    missing_ok: bool = False,
) -> Tuple[List[str], Dict[str, Dict[str, List[int]]], List[str]]:
    active_sites = (
        list(requested_sites)
        if requested_sites is not None
        else [site for site in SITE_CHOICES if site in all_heads_by_site_hook]
    )
    missing_sites = [site for site in active_sites if site not in all_heads_by_site_hook]
    if missing_sites and not missing_ok:
        raise ValueError(
            "No components found for requested site(s): "
            + ", ".join(missing_sites)
        )
    active_sites = [site for site in active_sites if site in all_heads_by_site_hook]
    heads_by_site_hook = {
        site: {
            hook: sorted(int(head) for head in heads)
            for hook, heads in all_heads_by_site_hook[site].items()
        }
        for site in active_sites
    }
    return active_sites, heads_by_site_hook, missing_sites


def score_heads_by_site_hook(
    *,
    heads_by_layer: Mapping[int, Sequence[int]],
    hook_component: str,
    sites: Sequence[str],
) -> Dict[str, Dict[str, List[int]]]:
    by_site: Dict[str, Dict[str, List[int]]] = {}
    for site in sites:
        by_site[site] = {
            hook_name(hook_component, layer_idx): sorted(int(head) for head in heads)
            for layer_idx, heads in heads_by_layer.items()
        }
    return by_site


def all_hook_names(heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]]) -> List[str]:
    return sorted(
        {
            hook
            for by_hook in heads_by_site_hook.values()
            for hook, heads in by_hook.items()
            if heads
        }
    )


def component_count_by_site(
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
) -> Dict[str, int]:
    return {
        site: sum(len(heads) for heads in by_hook.values())
        for site, by_hook in heads_by_site_hook.items()
    }


def unique_layer_heads(
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
) -> List[Tuple[int, int]]:
    return sorted(
        {
            (layer_from_hook_name(hook), int(head))
            for by_hook in heads_by_site_hook.values()
            for hook, heads in by_hook.items()
            for head in heads
        }
    )


def normalize_hook_component(component: str) -> str:
    if component == "o_proj":
        return "result"
    if component in ("attn_pattern", "attention_pattern"):
        return "pattern"
    return component


def hook_name(component: str, layer_idx: int) -> str:
    return utils.get_act_name(normalize_hook_component(component), layer_idx)


def layer_from_hook_name(name: str) -> int:
    parts = name.split(".")
    for idx, part in enumerate(parts):
        if part == "blocks" and idx + 1 < len(parts):
            return int(parts[idx + 1])
    raise ValueError(f"Could not infer layer index from hook name: {name}")


def quiet_build_example_refactored(**kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return datascience.build_example_refactored(**kwargs)


def last_token_indices(tokens: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected token tensor [batch, seq], got {tuple(tokens.shape)}.")
    if pad_token_id is None:
        return torch.full((tokens.shape[0],), tokens.shape[1] - 1, dtype=torch.long)
    non_pad = tokens.ne(int(pad_token_id)).to(torch.long)
    return non_pad.sum(dim=1).sub(1).clamp(min=0).to(torch.long)


def resolve_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def infer_prompt_format(model_name: str) -> str:
    resolved_name = resolve_model_name(model_name)
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
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
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
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_prefill},
        ],
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


def encode_prompt_texts(tokenizer, prompts: Sequence[str]) -> torch.Tensor:
    pp_utils._ensure_pad_token(tokenizer)
    encoded = tokenizer(
        list(prompts),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return encoded["input_ids"].to(torch.long)


def first_answer_token_id(tokenizer, formatted_prompt: str, answer_word: str) -> int:
    answer_word = answer_word.strip()
    if not answer_word:
        raise ValueError("Cannot encode an empty answer word.")

    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
    for suffix in (answer_word, f" {answer_word}"):
        full_ids = tokenizer(
            formatted_prompt + suffix,
            add_special_tokens=False,
        )["input_ids"]
        if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
            return int(full_ids[len(prompt_ids)])

    for suffix in (answer_word, f" {answer_word}"):
        token_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        if token_ids:
            return int(token_ids[0])

    raise ValueError(f"Could not encode answer {answer_word!r}.")


def answer_words_from_tokens(tokenizer, answer_tokens: torch.Tensor) -> List[List[str]]:
    rows = answer_tokens.detach().cpu().tolist()
    if rows and not isinstance(rows[0], list):
        rows = [rows]
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


def decode_prompts_from_tokens(tokenizer, tokens: torch.Tensor) -> List[str]:
    last_indices = last_token_indices(tokens, tokenizer.pad_token_id)
    return [
        tokenizer.decode(
            row[: int(last_idx.item()) + 1].tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for row, last_idx in zip(tokens, last_indices)
    ]


def span_starts(token_map_entry: torch.Tensor) -> torch.Tensor:
    if token_map_entry.ndim != 2 or token_map_entry.shape[1] < 1:
        raise ValueError(f"Expected token-map spans [batch, 2], got {tuple(token_map_entry.shape)}.")
    return token_map_entry[:, 0].to(torch.long)


def paired_contexts(
    *,
    num_entities: int,
    query_name: int,
    swap_pair: Tuple[int, int],
    counterfactual_mode: str,
) -> Tuple[List[Statement], List[Statement], Dict[str, Dict[int, int]]]:
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
    elif counterfactual_mode == "value_fetcher_objects":
        counterfactual_attrs = dict(original_attrs)
        counterfactual_attrs[swap_a] = num_entities
        counterfactual_attrs[swap_b] = num_entities + 1
    else:
        raise ValueError(f"Unknown counterfactual mode: {counterfactual_mode}")

    original = [
        Statement(swap_a, original_attrs[swap_a], "normal"),
        Statement(swap_b, original_attrs[swap_b], "normal"),
        *[Statement(idx, original_attrs[idx], "normal") for idx in non_swap],
        Statement(swap_a, swap_b, "swap"),
    ]
    if counterfactual_mode == "value_fetcher_objects":
        counterfactual = [
            Statement(swap_a, counterfactual_attrs[swap_a], "normal"),
            Statement(swap_b, counterfactual_attrs[swap_b], "normal"),
            *[Statement(idx, counterfactual_attrs[idx], "normal") for idx in non_swap],
            Statement(swap_a, swap_b, "swap"),
        ]
    else:
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


def normal_context_index(context: Sequence[Statement], box_idx: int) -> int:
    for idx, statement in enumerate(context):
        if statement.type in ("normal", "ref") and statement.name == box_idx:
            return idx
    raise ValueError(f"Could not find a normal context statement for box {box_idx}.")


def swap_target_box_from_context(context: Sequence[Statement], query_name: int) -> int:
    for statement in context:
        if statement.type != "swap":
            continue
        left_box, right_box = int(statement.name), int(statement.attr)
        if left_box == query_name:
            return right_box
        if right_box == query_name:
            return left_box
        break
    raise ValueError(f"query_name={query_name} must appear in the swap statement.")


def swap_ordinal_box_char(prompt: str, group_idx: int) -> int:
    match = pp_utils.SWAP_BOXES_RE.search(prompt)
    if match is None:
        raise ValueError(f"Could not find swap sentence in prompt:\n{prompt}")
    return int(match.start(group_idx))


def site_char(prompt: str, site: str) -> int:
    if site == "question_query_box":
        return pp_utils._question_query_box_char(prompt)
    if site == "swap_query_box":
        return pp_utils._swap_query_box_char(prompt)
    if site == "swap_target_box":
        return pp_utils._swap_target_box_char(prompt)
    if site == "context_target_box":
        return pp_utils._context_source_box_char(prompt)
    if site == "swap_left_box":
        return swap_ordinal_box_char(prompt, 1)
    if site == "swap_right_box":
        return swap_ordinal_box_char(prompt, 2)
    raise ValueError(f"Cannot compute a character position for site {site!r}.")


def compute_formatted_site_positions(
    *,
    tokenizer,
    prompts: Sequence[str],
    tokens: torch.Tensor,
    last_indices: torch.Tensor,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    fields: Dict[str, torch.Tensor] = {
        f"{prefix}_answer_colon_positions": last_indices.to(torch.long),
    }
    for site in SITE_CHOICES:
        if site == "answer_colon":
            continue
        values = []
        for prompt, token_row in zip(prompts, tokens):
            values.append(
                pp_utils._char_to_token_index(
                    tokenizer,
                    prompt,
                    site_char(prompt, site),
                    token_ids=token_row.tolist(),
                )
            )
        fields[f"{prefix}_{site}_positions"] = torch.tensor(values, dtype=torch.long)
    return fields


def object_words_for_attrs(
    *,
    vocab,
    attrs: Mapping[int, int],
    num_samples: int,
    prompt_id_start: int,
    box_idx: int,
) -> List[str]:
    return [
        vocab.fetch_shuffled_attr(
            attrs[box_idx],
            prompt_id_start + sample_idx,
        )[1]
        for sample_idx in range(num_samples)
    ]


def build_paired_dataset(
    *,
    tokenizer,
    model_name: str,
    vocab,
    num_entities: int,
    num_samples: int,
    query_name: int,
    prompt_id_start: int,
    swap_pair: Tuple[int, int],
    template_type: str,
    counterfactual_mode: str,
    prompt_format: str,
) -> DictTensorDataset:
    original_context, counterfactual_context, attr_metadata = paired_contexts(
        num_entities=num_entities,
        query_name=query_name,
        swap_pair=swap_pair,
        counterfactual_mode=counterfactual_mode,
    )
    max_attr_id = max(
        statement.attr
        for context in (original_context, counterfactual_context)
        for statement in context
        if statement.type in ("normal", "ref")
    )
    if max_attr_id >= len(vocab.filtered_country_capital_pairs):
        raise ValueError(
            f"Counterfactual mode {counterfactual_mode!r} needs attr id {max_attr_id}, "
            f"but the vocab only has {len(vocab.filtered_country_capital_pairs)} objects."
        )

    template_context = {"query_name": query_name, "raw_query_name": None}

    base_tokens, base_answers, _, base_maps = quiet_build_example_refactored(
        batch_size=num_samples,
        vocab=vocab,
        num_entities=num_entities,
        context=original_context,
        prompt_id_start=prompt_id_start,
        template_context=template_context,
        template_type=template_type,
    )
    source_tokens, source_answers, _, source_maps = quiet_build_example_refactored(
        batch_size=num_samples,
        vocab=vocab,
        num_entities=num_entities,
        context=counterfactual_context,
        prompt_id_start=prompt_id_start,
        template_context=template_context,
        template_type=template_type,
    )

    resolved_prompt_format = resolve_prompt_format(model_name, prompt_format)

    original_query_object_words = object_words_for_attrs(
        vocab=vocab,
        attrs=attr_metadata["original_attrs"],
        num_samples=num_samples,
        prompt_id_start=prompt_id_start,
        box_idx=query_name,
    )
    source_query_object_words = object_words_for_attrs(
        vocab=vocab,
        attrs=attr_metadata["counterfactual_attrs"],
        num_samples=num_samples,
        prompt_id_start=prompt_id_start,
        box_idx=query_name,
    )

    if resolved_prompt_format != "raw":
        base_prompts = decode_prompts_from_tokens(tokenizer, base_tokens)
        source_prompts = decode_prompts_from_tokens(tokenizer, source_tokens)
        formatted_base_prompts = [
            format_prompt_for_model(tokenizer, prompt, resolved_prompt_format)
            for prompt in base_prompts
        ]
        formatted_source_prompts = [
            format_prompt_for_model(tokenizer, prompt, resolved_prompt_format)
            for prompt in source_prompts
        ]
        formatted_base_tokens = encode_prompt_texts(tokenizer, formatted_base_prompts)
        formatted_source_tokens = encode_prompt_texts(tokenizer, formatted_source_prompts)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        base_last = last_token_indices(formatted_base_tokens, pad_token_id)
        source_last = last_token_indices(formatted_source_tokens, pad_token_id)

        base_answer_words = answer_words_from_tokens(tokenizer, base_answers)
        source_answer_words = answer_words_from_tokens(tokenizer, source_answers)
        labels = torch.tensor(
            [
                first_answer_token_id(
                    tokenizer,
                    prompt,
                    base_answer_words[row_idx][query_name],
                )
                for row_idx, prompt in enumerate(formatted_base_prompts)
            ],
            dtype=torch.long,
        )
        source_labels = torch.tensor(
            [
                first_answer_token_id(
                    tokenizer,
                    prompt,
                    source_answer_words[row_idx][query_name],
                )
                for row_idx, prompt in enumerate(formatted_source_prompts)
            ],
            dtype=torch.long,
        )
        candidate_labels = torch.tensor(
            [
                [
                    first_answer_token_id(tokenizer, formatted_base_prompts[row_idx], word)
                    for word in (
                        base_answer_words[row_idx] + source_answer_words[row_idx]
                    )
                ]
                for row_idx in range(num_samples)
            ],
            dtype=torch.long,
        )
        query_box_object_labels = torch.tensor(
            [
                first_answer_token_id(tokenizer, prompt, object_word)
                for prompt, object_word in zip(
                    formatted_base_prompts,
                    original_query_object_words,
                )
            ],
            dtype=torch.long,
        )
        source_query_box_object_labels = torch.tensor(
            [
                first_answer_token_id(tokenizer, prompt, object_word)
                for prompt, object_word in zip(
                    formatted_base_prompts,
                    source_query_object_words,
                )
            ],
            dtype=torch.long,
        )

        data = {
            "base_tokens": formatted_base_tokens.to(torch.long),
            "source_tokens": formatted_source_tokens.to(torch.long),
            "labels": labels,
            "source_labels": source_labels,
            "query_box_object_labels": query_box_object_labels,
            "source_query_box_object_labels": source_query_box_object_labels,
            "candidate_labels": candidate_labels,
            "base_last_token_indices": base_last,
            "source_last_token_indices": source_last,
        }
        data.update(
            compute_formatted_site_positions(
                tokenizer=tokenizer,
                prompts=formatted_base_prompts,
                tokens=formatted_base_tokens,
                last_indices=base_last,
                prefix="base",
            )
        )
        data.update(
            compute_formatted_site_positions(
                tokenizer=tokenizer,
                prompts=formatted_source_prompts,
                tokens=formatted_source_tokens,
                last_indices=source_last,
                prefix="source",
            )
        )
        return DictTensorDataset(data)

    labels = base_answers[:, query_name].to(torch.long)
    source_labels = source_answers[:, query_name].to(torch.long)
    candidate_labels = torch.cat([base_answers, source_answers], dim=1).to(torch.long)

    query_box_object_labels = torch.tensor(
        [vocab.encode_single_word(word) for word in original_query_object_words],
        dtype=torch.long,
    )
    source_query_box_object_labels = torch.tensor(
        [vocab.encode_single_word(word) for word in source_query_object_words],
        dtype=torch.long,
    )

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    base_last = last_token_indices(base_tokens, pad_token_id)
    source_last = last_token_indices(source_tokens, pad_token_id)
    swap_idx = num_entities
    target_box = swap_target_box_from_context(original_context, query_name)
    base_target_context_idx = normal_context_index(original_context, target_box)
    source_target_context_idx = normal_context_index(counterfactual_context, target_box)

    data = {
        "base_tokens": base_tokens.to(torch.long),
        "source_tokens": source_tokens.to(torch.long),
        "labels": labels,
        "source_labels": source_labels,
        "query_box_object_labels": query_box_object_labels,
        "source_query_box_object_labels": source_query_box_object_labels,
        "candidate_labels": candidate_labels,
        "base_last_token_indices": base_last,
        "source_last_token_indices": source_last,
        "base_answer_colon_positions": base_last,
        "source_answer_colon_positions": source_last,
        "base_question_query_box_positions": span_starts(base_maps["qn_subject"]),
        "source_question_query_box_positions": span_starts(source_maps["qn_subject"]),
        "base_context_target_box_positions": span_starts(
            base_maps["context"][base_target_context_idx]["subject"]
        ),
        "source_context_target_box_positions": span_starts(
            source_maps["context"][source_target_context_idx]["subject"]
        ),
        "base_swap_left_box_positions": span_starts(
            base_maps["context"][swap_idx]["left_box"]
        ),
        "source_swap_left_box_positions": span_starts(
            source_maps["context"][swap_idx]["left_box"]
        ),
        "base_swap_right_box_positions": span_starts(
            base_maps["context"][swap_idx]["right_box"]
        ),
        "source_swap_right_box_positions": span_starts(
            source_maps["context"][swap_idx]["right_box"]
        ),
    }

    swap_a, swap_b = swap_pair
    source_swap_is_reversed = counterfactual_mode != "value_fetcher_objects"
    if swap_a == query_name:
        data["base_swap_query_box_positions"] = data["base_swap_left_box_positions"]
        data["base_swap_target_box_positions"] = data["base_swap_right_box_positions"]
        if source_swap_is_reversed:
            data["source_swap_query_box_positions"] = data["source_swap_right_box_positions"]
            data["source_swap_target_box_positions"] = data["source_swap_left_box_positions"]
        else:
            data["source_swap_query_box_positions"] = data["source_swap_left_box_positions"]
            data["source_swap_target_box_positions"] = data["source_swap_right_box_positions"]
    elif swap_b == query_name:
        data["base_swap_query_box_positions"] = data["base_swap_right_box_positions"]
        data["base_swap_target_box_positions"] = data["base_swap_left_box_positions"]
        if source_swap_is_reversed:
            data["source_swap_query_box_positions"] = data["source_swap_left_box_positions"]
            data["source_swap_target_box_positions"] = data["source_swap_right_box_positions"]
        else:
            data["source_swap_query_box_positions"] = data["source_swap_right_box_positions"]
            data["source_swap_target_box_positions"] = data["source_swap_left_box_positions"]
    else:
        raise ValueError(f"query_name={query_name} must be in swap_pair={swap_pair}.")

    return DictTensorDataset(data)


def c_to_a_milk_site_char(prompt: str, site: str) -> int:
    if site == "question_query_box":
        return prompt.index("Question: Box X") + len("Question: Box ")
    if site == "context_target_box":
        return prompt.index("Box P") + len("Box ")
    if site in ("swap_query_box", "swap_target_box"):
        match = re.search(r"Swap\s+([A-Z])\s+and\s+([A-Z])\.", prompt)
        if match is None:
            raise ValueError(f"Could not find custom swap sentence in prompt:\n{prompt}")
        left_box, right_box = match.group(1), match.group(2)
        if site == "swap_query_box":
            return match.start(1) if left_box == "X" else match.start(2)
        if left_box == "P":
            return match.start(1)
        if right_box == "P":
            return match.start(2)
    raise ValueError(f"Cannot compute a custom character position for site {site!r}.")


def compute_c_to_a_milk_site_positions(
    *,
    tokenizer,
    prompt: str,
    tokens: torch.Tensor,
    last_index: torch.Tensor,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    fields: Dict[str, torch.Tensor] = {
        f"{prefix}_answer_colon_positions": last_index.to(torch.long),
    }
    for site in SITE_CHOICES:
        if site == "answer_colon":
            continue
        fields[f"{prefix}_{site}_positions"] = torch.tensor(
            [
                pp_utils._char_to_token_index(
                    tokenizer,
                    prompt,
                    c_to_a_milk_site_char(prompt, site),
                    token_ids=tokens[0].tolist(),
                )
            ],
            dtype=torch.long,
        )
    return fields


def build_c_to_a_milk_dataset(
    *,
    tokenizer,
    model_name: str,
    prompt_format: str,
) -> DictTensorDataset:
    resolved_prompt_format = resolve_prompt_format(model_name, prompt_format)
    formatted_base_prompt = format_prompt_for_model(
        tokenizer,
        C_TO_A_MILK_BASE_PROMPT,
        resolved_prompt_format,
    )
    formatted_source_prompt = format_prompt_for_model(
        tokenizer,
        C_TO_A_MILK_SOURCE_PROMPT,
        resolved_prompt_format,
    )
    base_tokens = encode_prompt_texts(tokenizer, [formatted_base_prompt])
    source_tokens = encode_prompt_texts(tokenizer, [formatted_source_prompt])
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    base_last = last_token_indices(base_tokens, pad_token_id)
    source_last = last_token_indices(source_tokens, pad_token_id)

    label = first_answer_token_id(
        tokenizer,
        formatted_base_prompt,
        C_TO_A_MILK_ANSWER,
    )
    source_label = first_answer_token_id(
        tokenizer,
        formatted_source_prompt,
        C_TO_A_MILK_ANSWER,
    )
    measured_object_label = first_answer_token_id(
        tokenizer,
        formatted_base_prompt,
        C_TO_A_MILK_MEASURED_OBJECT,
    )
    candidate_labels = [
        first_answer_token_id(tokenizer, formatted_base_prompt, word)
        for word in C_TO_A_MILK_CANDIDATES
    ]

    data = {
        "base_tokens": base_tokens.to(torch.long),
        "source_tokens": source_tokens.to(torch.long),
        "labels": torch.tensor([label], dtype=torch.long),
        "source_labels": torch.tensor([source_label], dtype=torch.long),
        "query_box_object_labels": torch.tensor(
            [measured_object_label],
            dtype=torch.long,
        ),
        "source_query_box_object_labels": torch.tensor(
            [measured_object_label],
            dtype=torch.long,
        ),
        "candidate_labels": torch.tensor([candidate_labels], dtype=torch.long),
        "base_last_token_indices": base_last,
        "source_last_token_indices": source_last,
    }
    data.update(
        compute_c_to_a_milk_site_positions(
            tokenizer=tokenizer,
            prompt=formatted_base_prompt,
            tokens=base_tokens,
            last_index=base_last,
            prefix="base",
        )
    )
    data.update(
        compute_c_to_a_milk_site_positions(
            tokenizer=tokenizer,
            prompt=formatted_source_prompt,
            tokens=source_tokens,
            last_index=source_last,
            prefix="source",
        )
    )
    return DictTensorDataset(data)


def move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def new_metrics() -> Dict[str, float]:
    return {
        "correct": 0,
        "candidate_correct": 0,
        "total": 0,
        "candidate_total": 0,
        "label_logit_sum": 0.0,
        "label_prob_sum": 0.0,
        "source_label_logit_sum": 0.0,
        "source_label_prob_sum": 0.0,
        "source_label_rank_sum": 0.0,
        "source_label_top1": 0,
        "source_label_total": 0,
        "source_label_minus_original_label_logit_margin_sum": 0.0,
        "source_label_minus_original_label_logit_margin_total": 0,
        "query_object_logit_sum": 0.0,
        "query_object_prob_sum": 0.0,
        "query_object_rank_sum": 0.0,
        "query_object_top1": 0,
        "query_object_total": 0,
        "query_label_minus_target_label_logit_margin_sum": 0.0,
        "query_label_minus_target_label_logit_margin_total": 0,
        "source_query_object_logit_sum": 0.0,
        "source_query_object_prob_sum": 0.0,
        "source_query_object_rank_sum": 0.0,
        "source_query_object_top1": 0,
        "source_query_object_total": 0,
    }


def token_stats(logits_at_answer: torch.Tensor, token_id: int) -> Dict[str, float]:
    prob = float(torch.softmax(logits_at_answer, dim=-1)[token_id].item())
    logit = float(logits_at_answer[token_id].item())
    rank = int((logits_at_answer > logits_at_answer[token_id]).sum().item()) + 1
    return {"logit": logit, "prob": prob, "rank": rank}


def token_delta(
    patched_stats: Mapping[str, float],
    original_stats: Mapping[str, float],
) -> Dict[str, float]:
    return {
        "logit": patched_stats["logit"] - original_stats["logit"],
        "prob": patched_stats["prob"] - original_stats["prob"],
        "rank": patched_stats["rank"] - original_stats["rank"],
    }


def logit_margin(
    logits_at_answer: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    return float(
        (
            logits_at_answer[positive_token_id]
            - logits_at_answer[negative_token_id]
        ).item()
    )


def scalar_delta(patched_value: float, original_value: float) -> float:
    return patched_value - original_value


def format_optional_float(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.4f}"


def optional_metric_delta(
    patched_metrics: Mapping[str, Optional[float]],
    original_metrics: Mapping[str, Optional[float]],
    key: str,
) -> Optional[float]:
    patched_value = patched_metrics[key]
    original_value = original_metrics[key]
    if patched_value is None or original_value is None:
        return None
    return patched_value - original_value


def update_metrics(
    metrics: Dict[str, float],
    logits_at_answer: torch.Tensor,
    label: int,
    candidate_labels: Optional[torch.Tensor],
    source_label: Optional[int] = None,
    query_object_label: Optional[int] = None,
    source_query_object_label: Optional[int] = None,
) -> None:
    pred = int(torch.argmax(logits_at_answer).item())
    label_logit = float(logits_at_answer[label].item())
    label_prob = float(torch.softmax(logits_at_answer, dim=-1)[label].item())
    metrics["correct"] += int(pred == label)
    metrics["total"] += 1
    metrics["label_logit_sum"] += label_logit
    metrics["label_prob_sum"] += label_prob

    if source_label is not None:
        stats = token_stats(logits_at_answer, source_label)
        metrics["source_label_logit_sum"] += stats["logit"]
        metrics["source_label_prob_sum"] += stats["prob"]
        metrics["source_label_rank_sum"] += stats["rank"]
        metrics["source_label_top1"] += int(pred == source_label)
        metrics["source_label_total"] += 1
        metrics["source_label_minus_original_label_logit_margin_sum"] += logit_margin(
            logits_at_answer,
            source_label,
            label,
        )
        metrics["source_label_minus_original_label_logit_margin_total"] += 1

    if query_object_label is not None:
        stats = token_stats(logits_at_answer, query_object_label)
        metrics["query_object_logit_sum"] += stats["logit"]
        metrics["query_object_prob_sum"] += stats["prob"]
        metrics["query_object_rank_sum"] += stats["rank"]
        metrics["query_object_top1"] += int(pred == query_object_label)
        metrics["query_object_total"] += 1
        metrics["query_label_minus_target_label_logit_margin_sum"] += logit_margin(
            logits_at_answer,
            query_object_label,
            label,
        )
        metrics["query_label_minus_target_label_logit_margin_total"] += 1

    if source_query_object_label is not None:
        stats = token_stats(logits_at_answer, source_query_object_label)
        metrics["source_query_object_logit_sum"] += stats["logit"]
        metrics["source_query_object_prob_sum"] += stats["prob"]
        metrics["source_query_object_rank_sum"] += stats["rank"]
        metrics["source_query_object_top1"] += int(pred == source_query_object_label)
        metrics["source_query_object_total"] += 1

    if candidate_labels is None:
        return

    candidates = torch.unique(candidate_labels.detach().to(logits_at_answer.device).to(torch.long))
    if not bool((candidates == label).any().item()):
        candidates = torch.cat(
            [candidates, torch.tensor([label], device=logits_at_answer.device)]
        )
    candidate_logits = logits_at_answer.index_select(dim=0, index=candidates)
    candidate_pred = int(candidates[int(torch.argmax(candidate_logits).item())].item())
    metrics["candidate_correct"] += int(candidate_pred == label)
    metrics["candidate_total"] += 1


def finalize_metrics(metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
    total = int(metrics["total"])
    candidate_total = int(metrics["candidate_total"])
    source_label_total = int(metrics["source_label_total"])
    query_object_total = int(metrics["query_object_total"])
    source_query_object_total = int(metrics["source_query_object_total"])
    source_label_margin_total = int(
        metrics["source_label_minus_original_label_logit_margin_total"]
    )
    query_label_margin_total = int(
        metrics["query_label_minus_target_label_logit_margin_total"]
    )
    return {
        "accuracy": metrics["correct"] / total if total else None,
        "candidate_accuracy": (
            metrics["candidate_correct"] / candidate_total if candidate_total else None
        ),
        "correct": int(metrics["correct"]),
        "candidate_correct": int(metrics["candidate_correct"]),
        "total": total,
        "candidate_total": candidate_total,
        "mean_label_logit": metrics["label_logit_sum"] / total if total else None,
        "mean_label_prob": metrics["label_prob_sum"] / total if total else None,
        "mean_source_label_logit": (
            metrics["source_label_logit_sum"] / source_label_total
            if source_label_total
            else None
        ),
        "mean_source_label_prob": (
            metrics["source_label_prob_sum"] / source_label_total
            if source_label_total
            else None
        ),
        "mean_source_label_rank": (
            metrics["source_label_rank_sum"] / source_label_total
            if source_label_total
            else None
        ),
        "source_label_top1_rate": (
            metrics["source_label_top1"] / source_label_total
            if source_label_total
            else None
        ),
        "source_label_total": source_label_total,
        "mean_source_label_minus_original_label_logit_margin": (
            metrics["source_label_minus_original_label_logit_margin_sum"]
            / source_label_margin_total
            if source_label_margin_total
            else None
        ),
        "source_label_minus_original_label_logit_margin_total": (
            source_label_margin_total
        ),
        "mean_query_object_logit": (
            metrics["query_object_logit_sum"] / query_object_total
            if query_object_total
            else None
        ),
        "mean_query_object_prob": (
            metrics["query_object_prob_sum"] / query_object_total
            if query_object_total
            else None
        ),
        "mean_query_object_rank": (
            metrics["query_object_rank_sum"] / query_object_total
            if query_object_total
            else None
        ),
        "query_object_top1_rate": (
            metrics["query_object_top1"] / query_object_total
            if query_object_total
            else None
        ),
        "query_object_total": query_object_total,
        "mean_query_label_minus_target_label_logit_margin": (
            metrics["query_label_minus_target_label_logit_margin_sum"]
            / query_label_margin_total
            if query_label_margin_total
            else None
        ),
        "query_label_minus_target_label_logit_margin_total": (
            query_label_margin_total
        ),
        "mean_source_query_object_logit": (
            metrics["source_query_object_logit_sum"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "mean_source_query_object_prob": (
            metrics["source_query_object_prob_sum"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "mean_source_query_object_rank": (
            metrics["source_query_object_rank_sum"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "source_query_object_top1_rate": (
            metrics["source_query_object_top1"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "source_query_object_total": source_query_object_total,
    }


def positions_for_sites(
    batch: Mapping[str, torch.Tensor],
    sites: Sequence[str],
    *,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    out = {}
    for site in sites:
        field = f"{prefix}_{site}_positions"
        if field not in batch:
            raise KeyError(f"Missing position field {field!r}.")
        out[site] = batch[field]
    return out


def patch_offsets_for_site(site: str) -> Tuple[int, ...]: 
    return (0,)
   


def patch_selected_attention_patterns(
    pattern: torch.Tensor,
    hook,
    *,
    source_cache: Mapping[str, torch.Tensor],
    original_cache: Optional[Mapping[str, torch.Tensor]] = None,
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    base_positions: Mapping[str, torch.Tensor],
    source_positions: Mapping[str, torch.Tensor],
    restore_heads_by_hook: Optional[Mapping[str, Sequence[int]]] = None,
    restore_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pattern.ndim != 4:
        raise ValueError(
            f"Expected attention pattern [batch, head, dest, src], got "
            f"{tuple(pattern.shape)} at {hook.name}."
        )

    source_pattern = source_cache[hook.name].to(pattern.device)
    if source_pattern.ndim != 4:
        raise ValueError(
            f"Expected source attention pattern [batch, head, dest, src], got "
            f"{tuple(source_pattern.shape)} at {hook.name}."
        )
    if pattern.shape[3] != source_pattern.shape[3]:
        raise ValueError(
            f"Cannot patch attention patterns with mismatched source lengths at "
            f"{hook.name}: base src={pattern.shape[3]}, source src={source_pattern.shape[3]}."
        )

    batch_size = int(pattern.shape[0])
    for site, base_pos_tensor in base_positions.items():
        heads = [
            head
            for head in heads_by_site_hook.get(site, {}).get(hook.name, ())
            if head < pattern.shape[1]
        ]
        if not heads:
            continue
        source_pos_tensor = source_positions[site]
        patch_offsets = patch_offsets_for_site(site)
        for batch_idx in range(batch_size):
            base_start = int(base_pos_tensor[batch_idx].item())
            source_start = int(source_pos_tensor[batch_idx].item())
            for offset in patch_offsets:
                base_pos = base_start + offset
                source_pos = source_start + offset
                if (
                    base_pos < 0
                    or base_pos >= pattern.shape[2]
                    or source_pos < 0
                    or source_pos >= source_pattern.shape[2]
                ):
                    continue
                for head_idx in heads:
                    pattern[batch_idx, head_idx, base_pos, :] = source_pattern[
                        batch_idx,
                        head_idx,
                        source_pos,
                        :,
                    ]
    if (
        original_cache is not None
        and restore_heads_by_hook is not None
        and restore_positions is not None
    ):
        restore_heads = [
            head
            for head in restore_heads_by_hook.get(hook.name, ())
            if head < pattern.shape[1]
        ]
        if restore_heads:
            original_pattern = original_cache[hook.name].to(pattern.device)
            for batch_idx in range(batch_size):
                restore_pos = int(restore_positions[batch_idx].item())
                if restore_pos < 0 or restore_pos >= pattern.shape[2]:
                    continue
                for head_idx in restore_heads:
                    pattern[batch_idx, head_idx, restore_pos, :] = original_pattern[
                        batch_idx,
                        head_idx,
                        restore_pos,
                        :,
                    ]
    return pattern


def patch_selected_heads(
    head_output: torch.Tensor,
    hook,
    *,
    source_cache: Mapping[str, torch.Tensor],
    original_cache: Optional[Mapping[str, torch.Tensor]] = None,
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    base_positions: Mapping[str, torch.Tensor],
    source_positions: Mapping[str, torch.Tensor],
    restore_heads_by_hook: Optional[Mapping[str, Sequence[int]]] = None,
    restore_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hook.name.endswith("hook_pattern"):
        return patch_selected_attention_patterns(
            head_output,
            hook,
            source_cache=source_cache,
            original_cache=original_cache,
            heads_by_site_hook=heads_by_site_hook,
            base_positions=base_positions,
            source_positions=source_positions,
            restore_heads_by_hook=restore_heads_by_hook,
            restore_positions=restore_positions,
        )

    if head_output.ndim != 4:
        raise ValueError(
            f"Expected head activation [batch, seq, head, dim], got "
            f"{tuple(head_output.shape)} at {hook.name}."
        )

    source_output = source_cache[hook.name].to(head_output.device)
    batch_size = int(head_output.shape[0])
    for site, base_pos_tensor in base_positions.items():
        heads = [
            head
            for head in heads_by_site_hook.get(site, {}).get(hook.name, ())
            if head < head_output.shape[2]
        ]
        if not heads:
            continue
        source_pos_tensor = source_positions[site]
        patch_offsets = patch_offsets_for_site(site)
        for batch_idx in range(batch_size):
            base_start = int(base_pos_tensor[batch_idx].item())
            source_start = int(source_pos_tensor[batch_idx].item())
            for offset in patch_offsets:
                base_pos = base_start + offset
                source_pos = source_start + offset
                if (
                    base_pos < 0
                    or base_pos >= head_output.shape[1]
                    or source_pos < 0
                    or source_pos >= source_output.shape[1]
                ):
                    continue
                for head_idx in heads:
                    head_output[batch_idx, base_pos, head_idx, :] = source_output[
                        batch_idx,
                        source_pos,
                        head_idx,
                        :,
                    ]
    if (
        original_cache is not None
        and restore_heads_by_hook is not None
        and restore_positions is not None
    ):
        restore_heads = [
            head
            for head in restore_heads_by_hook.get(hook.name, ())
            if head < head_output.shape[2]
        ]
        if restore_heads:
            original_output = original_cache[hook.name].to(head_output.device)
            for batch_idx in range(batch_size):
                restore_pos = int(restore_positions[batch_idx].item())
                for head_idx in restore_heads:
                    head_output[batch_idx, restore_pos, head_idx, :] = original_output[
                        batch_idx,
                        restore_pos,
                        head_idx,
                        :,
                    ]
    return head_output


def heads_by_hook_for_sites(
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    sites: Sequence[str],
) -> Dict[str, List[int]]:
    heads_by_hook: Dict[str, set] = defaultdict(set)
    for site in sites:
        for hook, heads in heads_by_site_hook.get(site, {}).items():
            heads_by_hook[hook].update(int(head) for head in heads)
    return {hook: sorted(heads) for hook, heads in heads_by_hook.items()}


def evaluate_interchange(
    *,
    model,
    tokenizer,
    dataloader: DataLoader,
    hook_names: Sequence[str],
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    sites: Sequence[str],
    device: torch.device,
    restore_answer_colon: bool,
) -> Dict[str, Dict[str, Optional[float]]]:
    metrics = {
        "original": new_metrics(),
        "counterfactual": new_metrics(),
        "patched": new_metrics(),
    }
    query_object_token_effects = []
    source_label_token_effects = []
    sample_index = 0
    hook_name_set = set(hook_names)
    should_restore_answer_colon = restore_answer_colon and "answer_colon" not in sites
    restore_heads_by_hook = (
        heads_by_hook_for_sites(heads_by_site_hook, sites)
        if should_restore_answer_colon
        else None
    )
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Interchange sites={','.join(sites)}"):
            batch = move_batch_to_device(batch, device)
            base_positions = positions_for_sites(batch, sites, prefix="base")
            source_positions = positions_for_sites(batch, sites, prefix="source")

            source_logits, source_cache = model.run_with_cache(
                batch["source_tokens"],
                names_filter=lambda name: name in hook_name_set,
                return_type="logits",
            )
            if should_restore_answer_colon:
                original_logits, original_cache = model.run_with_cache(
                    batch["base_tokens"],
                    names_filter=lambda name: name in hook_name_set,
                    return_type="logits",
                )
            else:
                original_logits = model(batch["base_tokens"], return_type="logits")
                original_cache = None
            hooks = [
                (
                    name,
                    lambda z, hook, name=name: patch_selected_heads(
                        z,
                        hook,
                        source_cache=source_cache,
                        original_cache=original_cache if should_restore_answer_colon else None,
                        heads_by_site_hook=heads_by_site_hook,
                        base_positions=base_positions,
                        source_positions=source_positions,
                        restore_heads_by_hook=restore_heads_by_hook,
                        restore_positions=(
                            batch["base_answer_colon_positions"]
                            if should_restore_answer_colon
                            else None
                        ),
                    ),
                )
                for name in hook_names
            ]
            patched_logits = model.run_with_hooks(
                batch["base_tokens"],
                fwd_hooks=hooks,
                return_type="logits",
            )

            for batch_idx in range(batch["labels"].shape[0]):
                label = int(batch["labels"][batch_idx].item())
                source_label = int(batch["source_labels"][batch_idx].item())
                query_object_label = int(
                    batch["query_box_object_labels"][batch_idx].item()
                )
                source_query_object_label = int(
                    batch["source_query_box_object_labels"][batch_idx].item()
                )
                base_last = int(batch["base_last_token_indices"][batch_idx].item())
                source_last = int(batch["source_last_token_indices"][batch_idx].item())
                candidates = (
                    batch["candidate_labels"][batch_idx]
                    if "candidate_labels" in batch
                    else None
                )
                update_metrics(
                    metrics["original"],
                    original_logits[batch_idx, base_last],
                    label,
                    candidates,
                    source_label=source_label,
                    query_object_label=query_object_label,
                    source_query_object_label=source_query_object_label,
                )
                update_metrics(
                    metrics["counterfactual"],
                    source_logits[batch_idx, source_last],
                    label,
                    candidates,
                    source_label=source_label,
                    query_object_label=query_object_label,
                    source_query_object_label=source_query_object_label,
                )
                update_metrics(
                    metrics["patched"],
                    patched_logits[batch_idx, base_last],
                    label,
                    candidates,
                    source_label=source_label,
                    query_object_label=query_object_label,
                    source_query_object_label=source_query_object_label,
                )
                original_source_label_stats = token_stats(
                    original_logits[batch_idx, base_last],
                    source_label,
                )
                counterfactual_source_label_stats = token_stats(
                    source_logits[batch_idx, source_last],
                    source_label,
                )
                patched_source_label_stats = token_stats(
                    patched_logits[batch_idx, base_last],
                    source_label,
                )
                original_object_stats = token_stats(
                    original_logits[batch_idx, base_last],
                    query_object_label,
                )
                counterfactual_object_stats = token_stats(
                    source_logits[batch_idx, source_last],
                    query_object_label,
                )
                patched_object_stats = token_stats(
                    patched_logits[batch_idx, base_last],
                    query_object_label,
                )
                original_source_object_stats = token_stats(
                    original_logits[batch_idx, base_last],
                    source_query_object_label,
                )
                counterfactual_source_object_stats = token_stats(
                    source_logits[batch_idx, source_last],
                    source_query_object_label,
                )
                patched_source_object_stats = token_stats(
                    patched_logits[batch_idx, base_last],
                    source_query_object_label,
                )
                original_source_label_margin = logit_margin(
                    original_logits[batch_idx, base_last],
                    source_label,
                    label,
                )
                counterfactual_source_label_margin = logit_margin(
                    source_logits[batch_idx, source_last],
                    source_label,
                    label,
                )
                patched_source_label_margin = logit_margin(
                    patched_logits[batch_idx, base_last],
                    source_label,
                    label,
                )
                original_query_label_margin = logit_margin(
                    original_logits[batch_idx, base_last],
                    query_object_label,
                    label,
                )
                counterfactual_query_label_margin = logit_margin(
                    source_logits[batch_idx, source_last],
                    query_object_label,
                    label,
                )
                patched_query_label_margin = logit_margin(
                    patched_logits[batch_idx, base_last],
                    query_object_label,
                    label,
                )
                query_object_token_effects.append(
                    {
                        "sample_index": sample_index,
                        "query_box_object_token_id": query_object_label,
                        "query_box_object_token": tokenizer.decode(
                            [query_object_label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "answer_label_token_id": label,
                        "answer_label_token": tokenizer.decode(
                            [label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "original": original_object_stats,
                        "counterfactual": counterfactual_object_stats,
                        "patched": patched_object_stats,
                        "patched_minus_original": token_delta(
                            patched_object_stats,
                            original_object_stats,
                        ),
                        "query_label_minus_target_label_logit_margin": {
                            "original": original_query_label_margin,
                            "counterfactual": counterfactual_query_label_margin,
                            "patched": patched_query_label_margin,
                            "patched_minus_original": scalar_delta(
                                patched_query_label_margin,
                                original_query_label_margin,
                            ),
                            "patched_minus_counterfactual": scalar_delta(
                                patched_query_label_margin,
                                counterfactual_query_label_margin,
                            ),
                        },
                        "source_query_box_object_token_id": source_query_object_label,
                        "source_query_box_object_token": tokenizer.decode(
                            [source_query_object_label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "source_query_box_object_original": original_source_object_stats,
                        "source_query_box_object_counterfactual": (
                            counterfactual_source_object_stats
                        ),
                        "source_query_box_object_patched": patched_source_object_stats,
                        "source_query_box_object_patched_minus_original": token_delta(
                            patched_source_object_stats,
                            original_source_object_stats,
                        ),
                    }
                )
                source_label_token_effects.append(
                    {
                        "sample_index": sample_index,
                        "source_label_token_id": source_label,
                        "source_label_token": tokenizer.decode(
                            [source_label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "answer_label_token_id": label,
                        "answer_label_token": tokenizer.decode(
                            [label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "original": original_source_label_stats,
                        "counterfactual": counterfactual_source_label_stats,
                        "patched": patched_source_label_stats,
                        "patched_minus_original": token_delta(
                            patched_source_label_stats,
                            original_source_label_stats,
                        ),
                        "patched_minus_counterfactual": token_delta(
                            patched_source_label_stats,
                            counterfactual_source_label_stats,
                        ),
                        "source_label_minus_original_label_logit_margin": {
                            "original": original_source_label_margin,
                            "counterfactual": counterfactual_source_label_margin,
                            "patched": patched_source_label_margin,
                            "patched_minus_original": scalar_delta(
                                patched_source_label_margin,
                                original_source_label_margin,
                            ),
                            "patched_minus_counterfactual": scalar_delta(
                                patched_source_label_margin,
                                counterfactual_source_label_margin,
                            ),
                        },
                    }
                )
                sample_index += 1

            del source_logits, source_cache, original_logits, patched_logits
            if original_cache is not None:
                del original_cache
            torch.cuda.empty_cache()

    finalized = {name: finalize_metrics(value) for name, value in metrics.items()}
    patched = finalized["patched"]
    original = finalized["original"]
    counterfactual = finalized["counterfactual"]
    finalized["deltas"] = {
        "patched_minus_original_accuracy": (
            None
            if patched["accuracy"] is None or original["accuracy"] is None
            else patched["accuracy"] - original["accuracy"]
        ),
        "patched_minus_original_candidate_accuracy": (
            None
            if patched["candidate_accuracy"] is None
            or original["candidate_accuracy"] is None
            else patched["candidate_accuracy"] - original["candidate_accuracy"]
        ),
        "patched_minus_original_mean_label_logit": (
            None
            if patched["mean_label_logit"] is None
            or original["mean_label_logit"] is None
            else patched["mean_label_logit"] - original["mean_label_logit"]
        ),
        "patched_minus_counterfactual_mean_label_logit": (
            None
            if patched["mean_label_logit"] is None
            or counterfactual["mean_label_logit"] is None
            else patched["mean_label_logit"] - counterfactual["mean_label_logit"]
        ),
        "patched_minus_original_mean_source_label_logit": (
            None
            if patched["mean_source_label_logit"] is None
            or original["mean_source_label_logit"] is None
            else patched["mean_source_label_logit"]
            - original["mean_source_label_logit"]
        ),
        "patched_minus_original_mean_source_label_prob": (
            None
            if patched["mean_source_label_prob"] is None
            or original["mean_source_label_prob"] is None
            else patched["mean_source_label_prob"]
            - original["mean_source_label_prob"]
        ),
        "patched_minus_counterfactual_mean_source_label_logit": (
            None
            if patched["mean_source_label_logit"] is None
            or counterfactual["mean_source_label_logit"] is None
            else patched["mean_source_label_logit"]
            - counterfactual["mean_source_label_logit"]
        ),
        "patched_minus_original_mean_source_label_minus_original_label_logit_margin": (
            optional_metric_delta(
                patched,
                original,
                "mean_source_label_minus_original_label_logit_margin",
            )
        ),
        "patched_minus_counterfactual_mean_source_label_minus_original_label_logit_margin": (
            optional_metric_delta(
                patched,
                counterfactual,
                "mean_source_label_minus_original_label_logit_margin",
            )
        ),
        "patched_minus_original_mean_query_object_logit": (
            None
            if patched["mean_query_object_logit"] is None
            or original["mean_query_object_logit"] is None
            else patched["mean_query_object_logit"]
            - original["mean_query_object_logit"]
        ),
        "patched_minus_original_mean_query_object_prob": (
            None
            if patched["mean_query_object_prob"] is None
            or original["mean_query_object_prob"] is None
            else patched["mean_query_object_prob"]
            - original["mean_query_object_prob"]
        ),
        "patched_minus_counterfactual_mean_query_object_logit": (
            None
            if patched["mean_query_object_logit"] is None
            or counterfactual["mean_query_object_logit"] is None
            else patched["mean_query_object_logit"]
            - counterfactual["mean_query_object_logit"]
        ),
        "patched_minus_original_mean_query_label_minus_target_label_logit_margin": (
            optional_metric_delta(
                patched,
                original,
                "mean_query_label_minus_target_label_logit_margin",
            )
        ),
        "patched_minus_counterfactual_mean_query_label_minus_target_label_logit_margin": (
            optional_metric_delta(
                patched,
                counterfactual,
                "mean_query_label_minus_target_label_logit_margin",
            )
        ),
        "patched_minus_original_mean_source_query_object_logit": (
            None
            if patched["mean_source_query_object_logit"] is None
            or original["mean_source_query_object_logit"] is None
            else patched["mean_source_query_object_logit"]
            - original["mean_source_query_object_logit"]
        ),
        "patched_minus_original_mean_source_query_object_prob": (
            None
            if patched["mean_source_query_object_prob"] is None
            or original["mean_source_query_object_prob"] is None
            else patched["mean_source_query_object_prob"]
            - original["mean_source_query_object_prob"]
        ),
        "patched_minus_counterfactual_mean_source_query_object_logit": (
            None
            if patched["mean_source_query_object_logit"] is None
            or counterfactual["mean_source_query_object_logit"] is None
            else patched["mean_source_query_object_logit"]
            - counterfactual["mean_source_query_object_logit"]
        ),
    }
    finalized["query_box_object_token_effects"] = query_object_token_effects
    finalized["source_label_token_effects"] = source_label_token_effects
    return finalized


def ensure_attn_result_enabled(model, hook_names: Sequence[str]) -> None:
    if not any("hook_result" in hook for hook in hook_names):
        return
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    else:
        model.cfg.use_attn_result = True


def evaluation_plan_for_sites(
    *,
    active_sites: Sequence[str],
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    combine_sites: bool,
) -> List[Tuple[str, List[str], Dict[str, Dict[str, Sequence[int]]]]]:
    if combine_sites:
        return [("combined", list(active_sites), dict(heads_by_site_hook))]
    return [
        (site, [site], {site: heads_by_site_hook[site]})
        for site in active_sites
    ]


def evaluate_role_prompt_scores(
    *,
    model,
    tokenizer,
    vocab,
    components: Sequence[Mapping[str, object]],
    roles: Sequence[str],
    requested_sites: Optional[Sequence[str]],
    component_hook_mode: str,
    model_name: str,
    num_entities: int,
    num_samples: int,
    query_name: int,
    prompt_id_start: int,
    swap_pair: Tuple[int, int],
    template_type: str,
    prompt_format: str,
    batch_size: int,
    combine_sites: bool,
    device: torch.device,
    restore_answer_colon: bool,
    include_c_to_a_milk_experiment: bool,
) -> Dict[str, object]:
    datasets = {
        prompt_label: build_paired_dataset(
            tokenizer=tokenizer,
            model_name=model_name,
            vocab=vocab,
            num_entities=num_entities,
            num_samples=num_samples,
            query_name=query_name,
            prompt_id_start=prompt_id_start,
            swap_pair=swap_pair,
            template_type=template_type,
            counterfactual_mode=counterfactual_mode,
            prompt_format=prompt_format,
        )
        for prompt_label, counterfactual_mode in ROLE_PROMPT_SCORE_MODES.items()
    }
    dataloaders = {
        prompt_label: DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for prompt_label, dataset in datasets.items()
    }
    c_to_a_milk_dataset = (
        build_c_to_a_milk_dataset(
            tokenizer=tokenizer,
            model_name=model_name,
            prompt_format=prompt_format,
        )
        if include_c_to_a_milk_experiment
        else None
    )
    c_to_a_milk_dataloader = (
        DataLoader(
            c_to_a_milk_dataset,
            batch_size=min(batch_size, len(c_to_a_milk_dataset)),
            shuffle=False,
        )
        if c_to_a_milk_dataset is not None
        else None
    )

    role_results: Dict[str, object] = {}
    for role in roles:
        role_components = filter_components_by_stage(components, [role])
        if not role_components:
            role_results[role] = {
                "skipped": True,
                "reason": "no components for role",
            }
            continue

        all_role_heads_by_site_hook = component_heads_by_site_hook(
            role_components,
            component_hook_mode=component_hook_mode,
        )
        role_requested_sites = (
            requested_sites
            if requested_sites is not None
            else ROLE_DEFAULT_SCORE_SITES.get(role)
        )
        active_sites, role_heads_by_site_hook, missing_sites = select_site_heads(
            all_role_heads_by_site_hook,
            role_requested_sites,
            missing_ok=True,
        )
        if not active_sites:
            role_results[role] = {
                "skipped": True,
                "reason": "no components for requested sites",
                "missing_sites": missing_sites,
            }
            continue

        role_hook_names = all_hook_names(role_heads_by_site_hook)
        ensure_attn_result_enabled(model, role_hook_names)
        plan = evaluation_plan_for_sites(
            active_sites=active_sites,
            heads_by_site_hook=role_heads_by_site_hook,
            combine_sites=combine_sites,
        )

        prompt_results: Dict[str, object] = {}
        for prompt_label, counterfactual_mode in ROLE_PROMPT_SCORE_MODES.items():
            prompt_result = {
                "counterfactual_mode": counterfactual_mode,
                "preview": preview_dataset(
                    datasets[prompt_label],
                    tokenizer,
                    active_sites,
                ),
                "results": {},
            }
            for name, sites, site_heads_by_hook in plan:
                prompt_result["results"][name] = evaluate_interchange(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=dataloaders[prompt_label],
                    hook_names=role_hook_names,
                    heads_by_site_hook=site_heads_by_hook,
                    sites=sites,
                    device=device,
                    restore_answer_colon=restore_answer_colon,
                )
            prompt_results[prompt_label] = prompt_result

        custom_experiments = {}
        if c_to_a_milk_dataloader is not None and c_to_a_milk_dataset is not None:
            custom_results = {}
            for name, sites, site_heads_by_hook in plan:
                custom_results[name] = evaluate_interchange(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=c_to_a_milk_dataloader,
                    hook_names=role_hook_names,
                    heads_by_site_hook=site_heads_by_hook,
                    sites=sites,
                    device=device,
                    restore_answer_colon=restore_answer_colon,
                )
            custom_experiments["c_to_a_milk"] = {
                "direction": "C_to_A",
                "base_prompt_name": "A",
                "source_prompt_name": "C",
                "base_prompt": C_TO_A_MILK_BASE_PROMPT,
                "source_prompt": C_TO_A_MILK_SOURCE_PROMPT,
                "answer": C_TO_A_MILK_ANSWER,
                "measured_object": C_TO_A_MILK_MEASURED_OBJECT,
                "measurement_delta_key": (
                    "patched_minus_original_mean_query_object_logit"
                ),
                "preview": preview_dataset(
                    c_to_a_milk_dataset,
                    tokenizer,
                    active_sites,
                ),
                "results": custom_results,
            }

        role_results[role] = {
            "skipped": False,
            "component_count": len(role_components),
            "component_count_by_site": component_count_by_site(role_heads_by_site_hook),
            "default_sites": list(ROLE_DEFAULT_SCORE_SITES.get(role, ())),
            "requested_sites": (
                list(role_requested_sites)
                if role_requested_sites is not None
                else None
            ),
            "active_sites": list(active_sites),
            "heads_by_site_hook": role_heads_by_site_hook,
            "hook_names": role_hook_names,
            "heads": [[layer, head] for layer, head in unique_layer_heads(role_heads_by_site_hook)],
            "missing_sites": missing_sites,
            "prompt_scores": prompt_results,
            "custom_experiments": custom_experiments,
        }

    return role_results


def decode_prompt(tokenizer, tokens: torch.Tensor, last_idx: int) -> str:
    return tokenizer.decode(
        tokens[: last_idx + 1].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def preview_dataset(dataset: DictTensorDataset, tokenizer, sites: Sequence[str]) -> Dict[str, object]:
    item = dataset[0]
    base_last = int(item["base_last_token_indices"].item())
    source_last = int(item["source_last_token_indices"].item())
    preview = {
        "original": decode_prompt(tokenizer, item["base_tokens"], base_last),
        "counterfactual": decode_prompt(tokenizer, item["source_tokens"], source_last),
        "label_token_id": int(item["labels"].item()),
        "label_token": tokenizer.decode([int(item["labels"].item())]),
        "source_label_token_id": int(item["source_labels"].item()),
        "source_label_token": tokenizer.decode(
            [int(item["source_labels"].item())],
            clean_up_tokenization_spaces=False,
        ),
        "query_box_object_token_id": int(item["query_box_object_labels"].item()),
        "query_box_object_token": tokenizer.decode(
            [int(item["query_box_object_labels"].item())],
            clean_up_tokenization_spaces=False,
        ),
        "source_query_box_object_token_id": int(
            item["source_query_box_object_labels"].item()
        ),
        "source_query_box_object_token": tokenizer.decode(
            [int(item["source_query_box_object_labels"].item())],
            clean_up_tokenization_spaces=False,
        ),
        "sites": {},
    }
    for site in sites:
        base_pos = int(item[f"base_{site}_positions"].item())
        source_pos = int(item[f"source_{site}_positions"].item())
        patch_offsets = patch_offsets_for_site(site)
        base_patch_positions = [
            base_pos + offset
            for offset in patch_offsets
            if 0 <= base_pos + offset < int(item["base_tokens"].shape[0])
        ]
        source_patch_positions = [
            source_pos + offset
            for offset in patch_offsets
            if 0 <= source_pos + offset < int(item["source_tokens"].shape[0])
        ]
        preview["sites"][site] = {
            "base_pos": base_pos,
            "base_token": tokenizer.decode([int(item["base_tokens"][base_pos].item())]),
            "source_pos": source_pos,
            "source_token": tokenizer.decode([int(item["source_tokens"][source_pos].item())]),
            "patch_offsets": list(patch_offsets),
            "base_patch_positions": [
                {
                    "pos": pos,
                    "token": tokenizer.decode(
                        [int(item["base_tokens"][pos].item())],
                        clean_up_tokenization_spaces=False,
                    ),
                }
                for pos in base_patch_positions
            ],
            "source_patch_positions": [
                {
                    "pos": pos,
                    "token": tokenizer.decode(
                        [int(item["source_tokens"][pos].item())],
                        clean_up_tokenization_spaces=False,
                    ),
                }
                for pos in source_patch_positions
            ],
        }
    return preview


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run interchange interventions by patching selected attention-head "
            "activations from paired counterfactual BOXES prompts into original prompts."
        )
    )
    parser.add_argument("--model_name", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--prompt-format",
        "--prompt_format",
        choices=PROMPT_FORMATS,
        default="raw",
        help=(
            "Prompt wrapper for generated BOXES data. Use the same value as "
            "run_path_patching.py and circuit_eval.py. `raw` preserves the old "
            "completion prompt. `auto` uses assistant-prefill chat prompts for "
            "instruct models, keeping `Answer:` in the assistant message so the "
            "next token is the object."
        ),
    )
    parser.add_argument("--vocab_tag", default="BOXES")
    parser.add_argument("--vocab_split", default="train")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--score_file",
        default="value_fetcher.pt",
        help=(
            "Score tensor to select heads from. May be an absolute path, a path "
            "relative to cwd, or a filename inside --results_dir."
        ),
    )
    parser.add_argument(
        "--components-json",
        "--components_json",
        type=Path,
        default=None,
        help=(
            "Optional circuit_eval metrics JSON. When provided, patch the "
            "filtered component heads from that file at their recorded positions "
            "instead of selecting top-k heads from --score_file."
        ),
    )
    parser.add_argument(
        "--component-set",
        "--component_set",
        choices=("auto", "components", "minimality_kept"),
        default="auto",
        help=(
            "Which component list to read from --components_json. `components` "
            "uses the filtered circuit; `minimality_kept` uses pruning survivors "
            "when present; `auto` prefers pruning survivors if available."
        ),
    )
    parser.add_argument(
        "--component-hook-mode",
        "--component_hook_mode",
        choices=("role_aligned", "recorded"),
        default="role_aligned",
        help=(
            "When --components_json is provided, choose which activation hook "
            "to patch for component heads. `role_aligned` patches value_fetcher "
            "components at o_proj/result hooks and pointer/address roles at "
            "attention pattern hooks. `recorded` preserves the hook_name stored "
            "in the component JSON."
        ),
    )
    parser.add_argument(
        "--stages",
        "--roles",
        nargs="+",
        choices=STAGE_CHOICES,
        default=None,
        help=(
            "When --components_json is provided, keep only components with these "
            "stage/role names before patching. For example: --roles pos_transmitter."
        ),
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--largest",
        action="store_true",
        help="Select largest scores. Default selects smallest/most-negative scores.",
    )
    parser.add_argument(
        "--hook_component",
        choices=("v", "z", "result", "o_proj", "pattern"),
        default="o_proj",
        help=(
            "Attention-head activation to patch in score-file mode. "
            "`pattern` patches where heads look. `v` patches value vectors. "
            "`z` patches pre-W_O head outputs. `o_proj`/`result` patch "
            "post-W_O per-head outputs."
        ),
    )
    parser.add_argument("--num_entities", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--query_name", type=int, default=0)
    parser.add_argument(
        "--swap_pair",
        type=parse_swap_pair,
        default=(0,1),
        help=(
            "Original swap order as entity ids, e.g. '0,1' gives original "
            "`Swap X P` and counterfactual `Swap P X` under the same prompt id."
        ),
    )
    parser.add_argument("--prompt_id_start", type=int, default=0)
    parser.add_argument("--template_type", default="normal")
    parser.add_argument(
        "--counterfactual_mode",
        choices=("new_swap_objects", "same_objects", "value_fetcher_objects"),
        default="new_swap_objects",
        help=(
            "`new_swap_objects` reverses the swapped boxes and gives those boxes "
            "fresh object attributes in the counterfactual, while leaving non-swap "
            "boxes unchanged. `same_objects` keeps the older same-object behavior. "
            "`value_fetcher_objects` keeps the same box slots, swap order, and query "
            "in the counterfactual, but assigns fresh objects to the swapped slots."
        ),
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=SITE_CHOICES,
        default=None,
        help=(
            "Sites to evaluate. For --components_json, the default is the sites "
            "present in the component file; otherwise the default is "
            f"{', '.join(DEFAULT_SCORE_SITES)}. By default each site is run separately. "
            "`swap_query_box` is the X/query token; `swap_target_box` is the "
            "P/source token; `context_target_box` is the earlier context mention "
            "of the target/source box. Each site patches the selected site token."
        ),
    )
    parser.add_argument(
        "--combine_sites",
        action="store_true",
        help="Patch all requested --sites in one combined intervention.",
    )
    parser.add_argument(
        "--role-prompt-scores",
        "--role_prompt_scores",
        action="store_true",
        help=(
            "With --components_json, additionally save per-role scores for both "
            "prompt structures: payload_prompt uses --counterfactual_mode "
            "value_fetcher_objects, pointer_prompt uses new_swap_objects. "
            "Unless --sites is passed, each role uses its canonical score site: "
            "value_fetcher/pos_transmitter answer_colon, pos_detector "
            "question_query_box, struct_reader_swap_target swap_target_box."
        ),
    )
    parser.add_argument(
        "--restore-answer-colon",
        "--restore_answer_colon",
        dest="restore_answer_colon",
        action="store_true",
        default=False,
        help=(
            "For interventions whose requested sites do not include answer_colon, "
            "restore the patched heads at answer_colon from the original/base "
            "activation cache. Disabled by default."
        ),
    )
    parser.add_argument(
        "--no-restore-answer-colon",
        "--no_restore_answer_colon",
        dest="restore_answer_colon",
        action="store_false",
        help="Keep answer_colon activation restoration disabled.",
    )
    parser.add_argument(
        "--c-to-a-milk-experiment",
        "--c_to_a_milk_experiment",
        dest="c_to_a_milk_experiment",
        action="store_true",
        default=True,
        help=(
            "Also run the fixed C -> A diagnostic: source prompt `Box P cup. "
            "Box X milk. ... Swap P and X.` patched into base prompt `Box X "
            "milk. Box P cup. ... Swap X and P.`, measuring the increase of "
            "the `milk` logit on the base prompt."
        ),
    )
    parser.add_argument(
        "--no-c-to-a-milk-experiment",
        "--no_c_to_a_milk_experiment",
        dest="c_to_a_milk_experiment",
        action="store_false",
        help="Skip the fixed C -> A milk-logit diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.results_dir = args.results_dir.resolve()
    resolved_model_name = resolve_model_name(args.model_name)
    resolved_prompt_format = resolve_prompt_format(
        resolved_model_name,
        args.prompt_format,
    )
    hook_component = normalize_hook_component(args.hook_component)
    score_path: Optional[Path] = None
    components_path: Optional[Path] = None
    components: Optional[List[Dict[str, object]]] = None
    unfiltered_components: Optional[List[Dict[str, object]]] = None
    component_set_used: Optional[str] = None
    component_payload: Dict[str, object] = {}

    if args.components_json is not None:
        components_path = args.components_json.expanduser().resolve()
        if not components_path.exists():
            raise FileNotFoundError(f"Could not find components JSON: {components_path}")
        unfiltered_components, component_set_used, component_payload = load_component_dicts(
            components_path,
            args.component_set,
        )
        components = filter_components_by_stage(unfiltered_components, args.stages)
        if not components:
            available = ", ".join(sorted(component_count_by_stage(unfiltered_components)))
            requested = ", ".join(args.stages or ())
            raise ValueError(
                f"No components remain after --roles/--stages filter ({requested}). "
                f"Available stages in {components_path}: {available}."
            )
        all_heads_by_site_hook = component_heads_by_site_hook(
            components,
            component_hook_mode=args.component_hook_mode,
        )
        active_sites, heads_by_site_hook, _ = select_site_heads(
            all_heads_by_site_hook,
            args.sites,
        )
    else:
        if args.role_prompt_scores:
            raise ValueError("--role-prompt-scores requires --components_json.")
        if args.stages is not None:
            raise ValueError(
                "--roles/--stages requires --components_json because raw score "
                "tensors do not carry component role metadata. For score-file "
                "mode, choose a role by setting --score_file, e.g. "
                "--score_file pos_transmitter.pt."
            )
        score_path = resolve_score_path(args.results_dir, args.score_file)
        if not score_path.exists():
            raise FileNotFoundError(f"Could not find score file: {score_path}")
        active_sites = list(args.sites) if args.sites is not None else list(DEFAULT_SCORE_SITES)
        heads = select_heads(score_path, args.top_k, args.largest)
        heads_by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer_idx, head_idx in heads:
            heads_by_layer[layer_idx].append(head_idx)
        heads_by_site_hook = score_heads_by_site_hook(
            heads_by_layer=heads_by_layer,
            hook_component=hook_component,
            sites=active_sites,
        )

    hook_names = all_hook_names(heads_by_site_hook)
    heads = unique_layer_heads(heads_by_site_hook)

    selected_device = normalize_device(args.device)
    set_seed(args.seed)

    print(f"Model: {resolved_model_name}", file=sys.stderr)
    if args.model_name != resolved_model_name:
        print(f"Requested model alias: {args.model_name}", file=sys.stderr)
    print(f"Prompt format: {resolved_prompt_format}", file=sys.stderr)
    if args.prompt_format != resolved_prompt_format:
        print(f"Requested prompt format: {args.prompt_format}", file=sys.stderr)
    if components_path is not None:
        print(f"Components JSON: {components_path}", file=sys.stderr)
        print(f"Component set: {component_set_used}", file=sys.stderr)
        print(f"Component hook mode: {args.component_hook_mode}", file=sys.stderr)
        if args.stages is not None:
            print(f"Stage/role filter: {', '.join(args.stages)}", file=sys.stderr)
    else:
        print(f"Score file: {score_path}", file=sys.stderr)
        print(f"Hook component: {args.hook_component} ({hook_component})", file=sys.stderr)
    print(
        "Restore answer_colon for non-answer_colon interventions: "
        f"{args.restore_answer_colon}",
        file=sys.stderr,
    )
    print(f"Device: {selected_device}", file=sys.stderr)

    model, tokenizer = get_model_and_tokenizer(resolved_model_name)
    ensure_attn_result_enabled(model, hook_names)

    vocab = load_vocab(
        args.vocab_tag,
        resolved_model_name,
        split=args.vocab_split,
        tokenizer=tokenizer,
    )
    dataset = build_paired_dataset(
        tokenizer=tokenizer,
        model_name=resolved_model_name,
        vocab=vocab,
        num_entities=args.num_entities,
        num_samples=args.num_samples,
        query_name=args.query_name,
        prompt_id_start=args.prompt_id_start,
        swap_pair=args.swap_pair,
        template_type=args.template_type,
        counterfactual_mode=args.counterfactual_mode,
        prompt_format=resolved_prompt_format,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Selected {len(heads)} unique heads:", heads, file=sys.stderr)
    print("Component count by site:", component_count_by_site(heads_by_site_hook), file=sys.stderr)
    print("Dataset preview:", file=sys.stderr)
    preview = preview_dataset(dataset, tokenizer, active_sites)
    print(json.dumps(preview, indent=2), file=sys.stderr)

    eval_plan = evaluation_plan_for_sites(
        active_sites=active_sites,
        heads_by_site_hook=heads_by_site_hook,
        combine_sites=args.combine_sites,
    )

    results = {}
    for name, sites, site_heads_by_hook in eval_plan:
        results[name] = evaluate_interchange(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            hook_names=hook_names,
            heads_by_site_hook=site_heads_by_hook,
            sites=sites,
            device=selected_device,
            restore_answer_colon=args.restore_answer_colon,
        )
        patched = results[name]["patched"]
        delta = results[name]["deltas"]
        print(
            f"{name}: patched acc={format_optional_float(patched['accuracy'])} "
            f"candidate_acc={format_optional_float(patched['candidate_accuracy'])} "
            "delta_label_logit="
            f"{format_optional_float(delta['patched_minus_original_mean_label_logit'])} "
            "delta_query_obj_logit="
            f"{format_optional_float(delta['patched_minus_original_mean_query_object_logit'])} "
            "delta_source_minus_original_margin="
            f"{format_optional_float(delta['patched_minus_original_mean_source_label_minus_original_label_logit_margin'])} "
            "delta_query_minus_target_margin="
            f"{format_optional_float(delta['patched_minus_original_mean_query_label_minus_target_label_logit_margin'])}",
            file=sys.stderr,
        )

    custom_experiments = {}
    if args.c_to_a_milk_experiment:
        custom_dataset = build_c_to_a_milk_dataset(
            tokenizer=tokenizer,
            model_name=resolved_model_name,
            prompt_format=resolved_prompt_format,
        )
        custom_dataloader = DataLoader(
            custom_dataset,
            batch_size=min(args.batch_size, len(custom_dataset)),
            shuffle=False,
        )
        custom_preview = preview_dataset(custom_dataset, tokenizer, active_sites)
        print("C -> A milk-logit experiment preview:", file=sys.stderr)
        print(json.dumps(custom_preview, indent=2), file=sys.stderr)
        custom_results = {}
        for name, sites, site_heads_by_hook in eval_plan:
            custom_results[name] = evaluate_interchange(
                model=model,
                tokenizer=tokenizer,
                dataloader=custom_dataloader,
                hook_names=hook_names,
                heads_by_site_hook=site_heads_by_hook,
                sites=sites,
                device=selected_device,
                restore_answer_colon=args.restore_answer_colon,
            )
            delta = custom_results[name]["deltas"]
            print(
                f"c_to_a_milk/{name}: delta_milk_logit="
                f"{format_optional_float(delta['patched_minus_original_mean_query_object_logit'])}",
                file=sys.stderr,
            )
        custom_experiments["c_to_a_milk"] = {
            "direction": "C_to_A",
            "base_prompt_name": "A",
            "source_prompt_name": "C",
            "base_prompt": C_TO_A_MILK_BASE_PROMPT,
            "source_prompt": C_TO_A_MILK_SOURCE_PROMPT,
            "answer": C_TO_A_MILK_ANSWER,
            "measured_object": C_TO_A_MILK_MEASURED_OBJECT,
            "measurement_delta_key": (
                "patched_minus_original_mean_query_object_logit"
            ),
            "preview": custom_preview,
            "results": custom_results,
        }

    role_prompt_scores = None
    if args.role_prompt_scores:
        if components is None:
            raise ValueError("--role-prompt-scores requires component data.")
        component_stage_counts = component_count_by_stage(components)
        roles_for_prompt_scores = (
            list(args.stages)
            if args.stages is not None
            else [
                stage
                for stage in STAGE_CHOICES
                if stage in component_stage_counts
            ]
        )
        print(
            "Evaluating per-role payload/pointer prompt scores for roles: "
            + ", ".join(roles_for_prompt_scores),
            file=sys.stderr,
        )
        role_prompt_scores = evaluate_role_prompt_scores(
            model=model,
            tokenizer=tokenizer,
            vocab=vocab,
            components=components,
            roles=roles_for_prompt_scores,
            requested_sites=args.sites,
            component_hook_mode=args.component_hook_mode,
            model_name=resolved_model_name,
            num_entities=args.num_entities,
            num_samples=args.num_samples,
            query_name=args.query_name,
            prompt_id_start=args.prompt_id_start,
            swap_pair=args.swap_pair,
            template_type=args.template_type,
            prompt_format=resolved_prompt_format,
            batch_size=args.batch_size,
            combine_sites=args.combine_sites,
            device=selected_device,
            restore_answer_colon=args.restore_answer_colon,
            include_c_to_a_milk_experiment=args.c_to_a_milk_experiment,
        )

    payload = {
        "model_name": resolved_model_name,
        "requested_model_name": args.model_name,
        "vocab_tag": args.vocab_tag,
        "vocab_split": args.vocab_split,
        "results_dir": str(args.results_dir),
        "score_file": str(score_path) if score_path is not None else None,
        "components_json": str(components_path) if components_path is not None else None,
        "component_set": component_set_used,
        "component_hook_mode": (
            args.component_hook_mode if components_path is not None else None
        ),
        "stage_filter": list(args.stages) if args.stages is not None else None,
        "role_kind_by_stage": ROLE_KIND_BY_STAGE,
        "patch_point_interpretation": PATCH_POINT_INTERPRETATION,
        "role_prompt_score_modes": ROLE_PROMPT_SCORE_MODES,
        "role_default_score_sites": ROLE_DEFAULT_SCORE_SITES,
        "role_prompt_scores_enabled": args.role_prompt_scores,
        "patch_offsets_by_site": {
            site: list(patch_offsets_for_site(site)) for site in active_sites
        },
        "restore_answer_colon": args.restore_answer_colon,
        "restore_answer_colon_applied_by_result": {
            name: bool(args.restore_answer_colon and "answer_colon" not in sites)
            for name, sites, _ in eval_plan
        },
        "source_component_count": (
            len(unfiltered_components) if unfiltered_components is not None else None
        ),
        "source_component_count_after_stage_filter": (
            len(components) if components is not None else None
        ),
        "source_component_count_by_stage": (
            component_count_by_stage(unfiltered_components)
            if unfiltered_components is not None
            else None
        ),
        "filtered_component_count_by_stage": (
            component_count_by_stage(components) if components is not None else None
        ),
        "source_component_count_by_position": component_payload.get(
            "component_count_by_position"
        ),
        "component_count_by_site": component_count_by_site(heads_by_site_hook),
        "heads_by_site_hook": heads_by_site_hook,
        "hook_names": hook_names,
        "top_k": args.top_k,
        "largest": args.largest,
        "hook_component": args.hook_component,
        "source_resolved_hook_component": (
            component_payload.get("resolved_hook_component")
            if components_path is not None
            else None
        ),
        "resolved_hook_component": (
            args.component_hook_mode
            if components_path is not None
            else hook_component
        ),
        "heads": [[layer, head] for layer, head in heads],
        "num_entities": args.num_entities,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "query_name": args.query_name,
        "swap_pair": list(args.swap_pair),
        "prompt_id_start": args.prompt_id_start,
        "template_type": args.template_type,
        "counterfactual_mode": args.counterfactual_mode,
        "prompt_format": resolved_prompt_format,
        "requested_prompt_format": args.prompt_format,
        "sites": list(active_sites),
        "combine_sites": args.combine_sites,
        "preview": preview,
        "results": results,
        "custom_experiments": custom_experiments,
        "role_prompt_scores": role_prompt_scores,
    }

    output_path = (
        default_output_path_for_score(
            components_path.stem if components_path is not None else args.score_file
        )
        if args.output_json is None
        else args.output_json
    )
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
