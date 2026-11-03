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

from dynamicet_data import datascience
from dynamicet_data.api import load_vocab
from dynamicet_data.box import Statement
from exp2 import pp_utils
from exp2.pp_utils import compute_topk_components, get_model_and_tokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results" / "path_patching"

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
    pp_utils.set_device(selected)
    return selected


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_cpu(path: Path) -> torch.Tensor:
    from exp2.artifacts import validate_score_artifact

    validate_score_artifact(path)
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
