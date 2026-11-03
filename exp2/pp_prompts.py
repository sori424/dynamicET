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
from dynamicet_data import Dataset
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
    "llama8b": "llama-8b-hf",
    "gemma9b": "gemma2-9b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen8b": "Qwen/Qwen3-8B",
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

def _model_device(model: HookedTransformer) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return device


def _model_num_layers(model: HookedTransformer) -> int:
    return int(model.cfg.n_layers)


def _model_num_heads(model: HookedTransformer) -> int:
    return int(model.cfg.n_heads)


def _ensure_pad_token(tokenizer: PreTrainedTokenizerBase) -> None:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"


def _z_hook_name(layer_idx: int) -> str:
    return utils.get_act_name("z", layer_idx)


def _o_proj_hook_name(layer_idx: int) -> str:
    """Hook name for the o_proj (W_O) input of each attention layer.

    In TransformerLens, hook_z captures each head's output *before* W_O is
    applied — the same tensor the reference code accesses via the HuggingFace
    o_proj forward pre-hook.  Shape: [batch, seq, n_heads, d_head].
    """
    return utils.get_act_name("z", layer_idx)


def _attn_hook_name(component: str, layer_idx: int) -> str:
    return utils.get_act_name(component, layer_idx)


def _receiver_key(layer_idx: int) -> str:
    return _z_hook_name(layer_idx)


def _layer_from_hook_name(hook_name: str) -> int:
    parts = hook_name.split(".")
    for idx, part in enumerate(parts):
        if part == "blocks" and idx + 1 < len(parts):
            return int(parts[idx + 1])
    raise ValueError(f"Could not infer layer index from hook name: {hook_name}")


def _move_batch_to_device(batch: Dict[str, torch.Tensor], model: HookedTransformer) -> Dict[str, torch.Tensor]:
    model_device = _model_device(model)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(model_device)
    return batch


def _read_jsonl_rows(datafile: str, num_samples: int) -> List[Dict]:
    rows: List[Dict] = []
    with open(datafile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if num_samples and len(rows) >= num_samples:
                break
    return rows


def _first_present(row: Dict, keys: Sequence[str]):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _clean_prompt(row: Dict) -> str:
    prompt = _first_present(row, ("original_input", "input", "clean_input", "base_input"))
    if prompt is None:
        raise KeyError("Could not find a clean/original prompt in the JSONL row.")
    return str(prompt)


def _corrupt_prompt(row: Dict) -> str:
    prompt = _first_present(
        row,
        ("counterfactual_input", "source_input", "corrupted_input", "alt_input"),
    )
    if prompt is None:
        raise KeyError("Could not find a counterfactual/source prompt in the JSONL row.")
    return str(prompt)


def _clean_answer(row: Dict) -> str:
    direct_answer = _first_present(
        row,
        (
            "after_swap_answer_input",
            "original_answer",
            "answer_original",
            "clean_answer",
            "answer",
        ),
    )
    if direct_answer is not None:
        return str(direct_answer)

    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        meta_answer = _first_present(
            metadata,
            ("answer", "original_answer", "clean_answer"),
        )
        if meta_answer is not None:
            return str(meta_answer)

    raise KeyError("Could not find a clean answer token in the JSONL row.")


def _encode_prompts(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    _ensure_pad_token(tokenizer)
    encoded = tokenizer(
        list(prompts),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(torch.long)
    attention_mask = encoded["attention_mask"].to(torch.long)
    attended = attention_mask.to(dtype=torch.bool)
    if not torch.all(attended.any(dim=1)):
        raise ValueError("Found an all-padding prompt.")
    positions_from_end = torch.flip(attended, dims=[1]).to(torch.long).argmax(dim=1)
    last_token_indices = attention_mask.shape[1] - 1 - positions_from_end
    return input_ids, last_token_indices


def _encode_single_token_answer(
    tokenizer: PreTrainedTokenizerBase,
    answer: str,
) -> int:
    for candidate in (f" {answer}", answer):
        token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if len(token_ids) == 1:
            return int(token_ids[0])
    raise ValueError(f"Answer {answer!r} is not a single token for this tokenizer.")


def _as_long_tensor(data: Any) -> torch.Tensor:
    tensor = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    return tensor.to(torch.long)


def _slice_tensor_rows(tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    if num_samples and tensor.shape[0] > num_samples:
        return tensor[:num_samples]
    return tensor


def _compute_last_token_indices_from_tokens(
    tokens: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    if pad_token_id is None:
        return torch.full(
            (tokens.shape[0],),
            tokens.shape[1] - 1,
            dtype=torch.long,
        )

    non_pad = tokens.ne(pad_token_id)
    if not torch.all(non_pad.any(dim=1)):
        raise ValueError("Found an all-padding example while inferring last token indices.")
    positions_from_end = torch.flip(non_pad, dims=[1]).to(torch.long).argmax(dim=1)
    return tokens.shape[1] - 1 - positions_from_end


def _squeeze_label_tensor(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 0:
        return labels.unsqueeze(0)
    if labels.ndim == 2 and labels.shape[1] == 1:
        return labels.squeeze(1)
    if labels.ndim != 1:
        raise ValueError(
            "Path patching expects one label token id per example. "
            "Pass a 1D label tensor such as `base_answers[:, query_name]`."
        )
    return labels


def _decode_prompt_from_tokens(
    tokenizer: PreTrainedTokenizerBase,
    tokens: torch.Tensor,
    last_token_index: int,
) -> str:
    token_slice = tokens[: last_token_index + 1].tolist()
    return tokenizer.decode(
        token_slice,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def _parse_swap_prompt(prompt: str) -> Dict[str, str]:
    assignments = list(ASSIGNMENT_RE.finditer(prompt))
    if len(assignments) < 2:
        raise ValueError(f"Prompt did not match the expected box-swap format:\n{prompt}")

    swap_match = SWAP_BOXES_RE.search(prompt)
    question_match = QUESTION_SENTENCE_RE.search(prompt)
    answer_match = QUERY_SENTENCE_RE.search(prompt)
    if swap_match is None or (question_match is None and answer_match is None):
        raise ValueError(f"Prompt did not match the expected box-swap format:\n{prompt}")

    assignments_by_box = {
        match.group(1).upper(): match.group(2) for match in assignments
    }
    swap_a, swap_b = swap_match.group(1).upper(), swap_match.group(2).upper()
    if swap_a not in assignments_by_box or swap_b not in assignments_by_box:
        raise ValueError(f"Swap boxes do not align with prompt context:\n{prompt}")

    box1, obj1 = swap_a, assignments_by_box[swap_a]
    box2, obj2 = swap_b, assignments_by_box[swap_b]
    query_box = (
        question_match.group(1).upper()
        if question_match is not None
        else answer_match.group(1).upper()
    )

    if (
        question_match is not None
        and answer_match is not None
        and question_match.group(1).upper() != answer_match.group(1).upper()
    ):
        raise ValueError(
            "Question and answer query-box labels disagree in prompt:\n"
            f"{prompt}"
        )

    if query_box not in {box1, box2}:
        raise ValueError(
            "Query box is not part of the swap pair in prompt:\n"
            f"{prompt}"
        )

    return {
        "box1": box1,
        "obj1": obj1,
        "box2": box2,
        "obj2": obj2,
        "query_box": query_box,
        "all_boxes": tuple(assignments_by_box.keys()),
        "all_objects": tuple(assignments_by_box.values()),
    }


def _query_source_box(meta: Mapping[str, str]) -> str:
    if meta["query_box"] == meta["box1"]:
        return meta["box2"]
    if meta["query_box"] == meta["box2"]:
        return meta["box1"]
    raise ValueError(
        f"Query box {meta['query_box']} is neither {meta['box1']} nor {meta['box2']}."
    )


def _query_source_object(meta: Mapping[str, str]) -> str:
    source_box = _query_source_box(meta)
    if source_box == meta["box1"]:
        return meta["obj1"]
    return meta["obj2"]


def _query_box_object(meta: Mapping[str, str]) -> str:
    if meta["query_box"] == meta["box1"]:
        return meta["obj1"]
    if meta["query_box"] == meta["box2"]:
        return meta["obj2"]
    raise ValueError(
        f"Query box {meta['query_box']} is neither {meta['box1']} nor {meta['box2']}."
    )


def _char_to_token_index(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    char_idx: int,
    token_ids: Optional[Sequence[int]] = None,
) -> int:
    if char_idx < 0 or char_idx >= len(text):
        raise ValueError(f"Character index {char_idx} is out of range for the prompt.")

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    pad_token_id = tokenizer.pad_token_id

    filtered_pairs = None
    if token_ids is not None:
        filtered_pairs = [
            (orig_idx, int(tok_id))
            for orig_idx, tok_id in enumerate(token_ids)
            if int(tok_id) not in special_ids
            and (pad_token_id is None or int(tok_id) != int(pad_token_id))
        ]

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mapping = encoded["offset_mapping"]
        encoded_ids = encoded["input_ids"]
        if token_ids is None:
            token_ids = encoded_ids
            filtered_pairs = list(enumerate(int(tok_id) for tok_id in token_ids))

        if list(encoded_ids) == [int(tok_id) for tok_id in token_ids]:
            for token_idx, (start, end) in enumerate(offset_mapping):
                if int(start) <= char_idx < int(end):
                    return token_idx

        filtered_ids = (
            [tok_id for _, tok_id in filtered_pairs]
            if filtered_pairs is not None
            else list(encoded_ids)
        )
        if list(encoded_ids) == filtered_ids and len(offset_mapping) == len(filtered_ids):
            for token_idx, (start, end) in enumerate(offset_mapping):
                if int(start) <= char_idx < int(end):
                    if filtered_pairs is None:
                        return token_idx
                    return filtered_pairs[token_idx][0]
    except (TypeError, KeyError, NotImplementedError):
        pass

    if token_ids is None:
        token_ids = tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        filtered_pairs = list(enumerate(int(tok_id) for tok_id in token_ids))

    if filtered_pairs is None:
        filtered_pairs = list(enumerate(int(tok_id) for tok_id in token_ids))

    token_pieces = []
    for orig_idx, token_id in filtered_pairs:
        piece = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if piece:
            token_pieces.append((orig_idx, piece))
    if not token_pieces:
        raise ValueError(
            f"Could not map character index {char_idx} to a token position."
        )

    cursor = 0
    for orig_idx, piece in token_pieces:
        next_cursor = cursor + len(piece)
        if cursor <= char_idx < next_cursor:
            return orig_idx
        cursor = next_cursor

    raise ValueError(
        f"Could not map character index {char_idx} inside the decoded token sequence."
    )


def _tokenize_ids(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> List[int]:
    return [
        int(token_id)
        for token_id in tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    ]


def _find_subsequence_starts(
    token_ids: Sequence[int],
    pattern_ids: Sequence[int],
) -> List[int]:
    if not pattern_ids or len(pattern_ids) > len(token_ids):
        return []
    return [
        start
        for start in range(len(token_ids) - len(pattern_ids) + 1)
        if list(token_ids[start : start + len(pattern_ids)]) == list(pattern_ids)
    ]


def _query_box_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    query_box: str,
) -> List[int]:
    for candidate in (f" {query_box}", query_box):
        token_ids = _tokenize_ids(tokenizer, candidate)
        if token_ids:
            return token_ids
    raise ValueError(f"Could not tokenize query-box label {query_box!r}.")


def _matched_slice_with_boundary(prompt: str, match: re.Match) -> str:
    start = match.start(0)
    if start > 0 and prompt[start - 1].isspace():
        start -= 1
    return prompt[start : match.end(0)]


def _resolve_sentence_match(prompt: str, mode: str) -> Tuple[str, str]:
    meta = _parse_swap_prompt(prompt)
    query_box = meta["query_box"]

    if mode == "question_query_box":
        match = QUESTION_SENTENCE_RE.search(prompt)
        if match is not None:
            sentence = _matched_slice_with_boundary(prompt, match)
            return sentence, query_box
        match = QUERY_SENTENCE_RE.search(prompt)
        if match is not None:
            sentence = _matched_slice_with_boundary(prompt, match)
            return sentence, query_box
        raise ValueError("Could not find question or query sentence in prompt.")

    if mode == "answer_query_box":
        match = QUERY_SENTENCE_RE.search(prompt)
        if match is None:
            return _resolve_sentence_match(prompt, "question_query_box")
        sentence = _matched_slice_with_boundary(prompt, match)
        return sentence, query_box

    if mode == "swap_query_box":
        match = SWAP_BOXES_RE.search(prompt)
        if match is None:
            raise ValueError("Could not find swap sentence in prompt.")
        return _matched_slice_with_boundary(prompt, match), query_box

    if mode == "swap_target_box":
        target_box = _query_source_box(meta)
        match = SWAP_BOXES_RE.search(prompt)
        if match is None:
            raise ValueError("Could not find swap sentence in prompt.")
        return _matched_slice_with_boundary(prompt, match), target_box

    if mode == "context_query_box":
        for match in ASSIGNMENT_RE.finditer(prompt):
            if match.group(1).upper() == query_box:
                return _matched_slice_with_boundary(prompt, match), query_box
        raise ValueError(
            f"Query box {query_box} does not appear in an earlier context statement."
        )

    if mode in ("context_target_box", "context_source_box"):
        source_box = _query_source_box(meta)
        for match in ASSIGNMENT_RE.finditer(prompt):
            if match.group(1).upper() == source_box:
                return _matched_slice_with_boundary(prompt, match), source_box
        raise ValueError(
            f"Source box {source_box} does not appear in an earlier context statement."
        )

    if mode in ("source_object", "context_source_object", "context_query_object"):
        if mode == "context_query_object":
            target_box = meta["query_box"]
            target_object = _query_box_object(meta)
        else:
            target_box = _query_source_box(meta)
            target_object = _query_source_object(meta)
        for match in ASSIGNMENT_RE.finditer(prompt):
            if match.group(1).upper() == target_box:
                return _matched_slice_with_boundary(prompt, match), target_object
        raise ValueError(
            f"Target box {target_box} does not appear in an earlier context statement."
        )

    raise ValueError(f"Unknown prompt position mode: {mode}")


def _find_query_box_token_position(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    token_ids: Sequence[int],
    mode: str,
) -> int:
    if mode == "question_query_box":
        char_idx = _question_query_box_char(prompt)
    elif mode == "answer_query_box":
        char_idx = _answer_query_box_char(prompt)
    elif mode == "swap_query_box":
        char_idx = _swap_query_box_char(prompt)
    elif mode == "swap_target_box":
        char_idx = _swap_target_box_char(prompt)
    elif mode == "context_query_box":
        char_idx = _context_query_box_char(prompt)
    elif mode in ("context_target_box", "context_source_box"):
        char_idx = _context_source_box_char(prompt)
    else:
        raise ValueError(f"Unknown prompt position mode: {mode}")

    return _char_to_token_index(
        tokenizer,
        prompt,
        char_idx,
        token_ids=token_ids,
    )


def _question_query_box_char(prompt: str) -> int:
    match = QUESTION_SENTENCE_RE.search(prompt)
    if match is not None:
        return int(match.start(1))
    match = QUERY_SENTENCE_RE.search(prompt)
    if match is None:
        raise ValueError("Could not find question or query sentence in prompt.")
    return int(match.start(1))


def _answer_query_box_char(prompt: str) -> int:
    match = QUERY_SENTENCE_RE.search(prompt)
    if match is None:
        return _question_query_box_char(prompt)
    return int(match.start(1))


def _swap_query_box_char(prompt: str) -> int:
    meta = _parse_swap_prompt(prompt)
    match = SWAP_BOXES_RE.search(prompt)
    if match is None:
        raise ValueError("Could not find swap sentence in prompt.")
    if match.group(1).upper() == meta["query_box"]:
        return int(match.start(1))
    if match.group(2).upper() == meta["query_box"]:
        return int(match.start(2))
    raise ValueError(
        f"Query box {meta['query_box']} does not appear in the swap sentence."
    )


def _swap_target_box_char(prompt: str) -> int:
    meta = _parse_swap_prompt(prompt)
    target_box = _query_source_box(meta)
    match = SWAP_BOXES_RE.search(prompt)
    if match is None:
        raise ValueError("Could not find swap sentence in prompt.")
    if match.group(1).upper() == target_box:
        return int(match.start(1))
    if match.group(2).upper() == target_box:
        return int(match.start(2))
    raise ValueError(
        f"Target box {target_box} does not appear in the swap sentence."
    )


def _context_query_box_char(prompt: str) -> int:
    meta = _parse_swap_prompt(prompt)
    for match in ASSIGNMENT_RE.finditer(prompt):
        if match.group(1).upper() == meta["query_box"]:
            return int(match.start(1))
    raise ValueError(
        f"Query box {meta['query_box']} does not appear in an earlier context statement."
    )


def _context_source_box_char(prompt: str) -> int:
    meta = _parse_swap_prompt(prompt)
    source_box = _query_source_box(meta)
    for match in ASSIGNMENT_RE.finditer(prompt):
        if match.group(1).upper() == source_box:
            return int(match.start(1))
    raise ValueError(
        f"Source box {source_box!r} does not appear in an earlier context statement."
    )


def _source_object_char(prompt: str) -> int:
    meta = _parse_swap_prompt(prompt)
    source_box = _query_source_box(meta)
    for match in ASSIGNMENT_RE.finditer(prompt):
        if match.group(1).upper() == source_box:
            return int(match.start(2))
    raise ValueError(
        f"Source box {source_box!r} does not appear in an earlier context statement."
    )


def _trim_trailing_pad_tokens(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Sequence[int],
) -> List[int]:
    trimmed_token_ids = list(token_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        while trimmed_token_ids and int(trimmed_token_ids[-1]) == int(pad_token_id):
            trimmed_token_ids.pop()
    return trimmed_token_ids


def _find_context_object_token_position(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    token_ids: Sequence[int],
    mode: str,
) -> int:
    meta = _parse_swap_prompt(prompt)
    if mode == "context_query_object":
        target_box = meta["query_box"]
        target_object = _query_box_object(meta)
    elif mode in ("source_object", "context_source_object"):
        target_box = _query_source_box(meta)
        target_object = _query_source_object(meta)
    else:
        raise ValueError(f"Unknown object position mode: {mode}")

    trimmed_token_ids = _trim_trailing_pad_tokens(tokenizer, token_ids)

    target_match = None
    for match in ASSIGNMENT_RE.finditer(prompt):
        if match.group(1).upper() == target_box:
            target_match = match
            break

    if target_match is None:
        raise ValueError(
            f"Could not find the assignment for Box {target_box!r}."
        )

    try:
        return _char_to_token_index(
            tokenizer,
            prompt,
            int(target_match.start(2)),
            token_ids=trimmed_token_ids,
        )
    except ValueError:
        pass

    sentence_text = _matched_slice_with_boundary(prompt, target_match)
    sentence_ids = _tokenize_ids(tokenizer, sentence_text)
    sentence_starts = _find_subsequence_starts(trimmed_token_ids, sentence_ids)
    if len(sentence_starts) != 1:
        raise ValueError(
            "Expected exactly one tokenized context assignment match, "
            f"found {len(sentence_starts)} for prompt:\n{prompt}"
        )
    sentence_start = sentence_starts[0]

    object_candidates = [f" {target_object}", target_object]
    for candidate in object_candidates:
        object_ids = _tokenize_ids(tokenizer, candidate)
        object_starts = _find_subsequence_starts(sentence_ids, object_ids)
        if len(object_starts) == 1:
            return sentence_start + object_starts[0]

    raise ValueError(
        f"Could not uniquely localize context object {target_object!r} in prompt:\n{prompt}"
    )


def _compute_prompt_position_fields(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    prefix: str,
    token_batches: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, torch.Tensor]:
    positions = {mode: [] for mode in PROMPT_POSITION_MODES}

    for prompt_idx, prompt in enumerate(prompts):
        token_ids = (
            list(token_batches[prompt_idx])
            if token_batches is not None
            else _tokenize_ids(tokenizer, prompt)
        )
        for mode in PROMPT_POSITION_MODES:
            if mode in ("source_object", "context_source_object", "context_query_object"):
                positions[mode].append(
                    _find_context_object_token_position(
                        tokenizer,
                        prompt,
                        token_ids,
                        mode,
                    )
                )
            else:
                positions[mode].append(
                    _find_query_box_token_position(
                        tokenizer,
                        prompt,
                        token_ids,
                        mode,
                    )
                )

    return {
        f"{prefix}{mode}_positions": torch.tensor(values, dtype=torch.long)
        for mode, values in positions.items()
    }
