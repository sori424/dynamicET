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
from datasets import Dataset
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


def _load_custom_llama_family_model(model_source: str) -> Tuple[HookedTransformer, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    _ensure_pad_token(tokenizer)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        "llama-7b-hf",
        hf_model=hf_model,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        fold_value_biases=False,
        device=device,
        dtype=torch.float32,
    )
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer


def _load_goat_model() -> Tuple[HookedTransformer, PreTrainedTokenizerBase]:
    base_model = "decapoda-research/llama-7b-hf"
    lora_weights = "tiedong/goat-lora-7b"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    _ensure_pad_token(tokenizer)

    hf_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
    )
    hf_model = PeftModel.from_pretrained(
        hf_model,
        lora_weights,
        torch_dtype=torch.float32,
    )
    if hasattr(hf_model, "merge_and_unload"):
        hf_model = hf_model.merge_and_unload()

    model = HookedTransformer.from_pretrained(
        "llama-7b-hf",
        hf_model=hf_model,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        refactor_factored_attn_matrices=False,
        fold_value_biases=False,
        device=device,
        dtype=torch.float32,
    )
    model.tokenizer = tokenizer
    model.eval()
    return model, tokenizer


def compute_topk_components(
    patching_scores: torch.Tensor, k: int, largest=True, return_values=False
):
    """
    Computes the topk most influential components (i.e. heads) for patching.

    Args:
        patching_scores: patching scores for the components.
        k: number of components to return.
        largest: whether to return the largest or smallest components.
        return_values: whether to return the values of the components or not.
    """

    top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices
    top_values = torch.topk(patching_scores.flatten(), k, largest=largest).values

    row_indices = top_indices // patching_scores.shape[1]
    col_indices = top_indices % patching_scores.shape[1]
    top_components = torch.stack((row_indices, col_indices), dim=1).tolist()

    if return_values:
        return top_components, top_values.tolist()
    return top_components


def compute_prev_query_box_pos(input_ids, last_token_index):
    """
    Computes the position of the previous query box label token.

    Args:
        input_ids: input ids of the example.
        last_token_index: last token index of the example.
    """

    query_box_token = input_ids[last_token_index - 2]
    prev_query_box_token_pos = (
        (input_ids[: last_token_index - 2] == query_box_token).nonzero().item()
    )
    return prev_query_box_token_pos


def get_model_and_tokenizer(model_name: str):
    """
    Loads the model and tokenizer through TransformerLens.

    Args:
        model_name (str): Name of the model to load.
    """

    resolved_name = MODEL_ALIASES.get(model_name, model_name)

    if model_name in CUSTOM_LLAMA_FAMILY_MODELS:
        return _load_custom_llama_family_model(CUSTOM_LLAMA_FAMILY_MODELS[model_name])

    if model_name == "goat":
        return _load_goat_model()

    model = fetch_model(resolved_name, device=str(device))
    tokenizer = model.tokenizer
    _ensure_pad_token(tokenizer)
    return model, tokenizer


def load_dataloader(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    datafile: str = None,
    num_samples: int = None,
    num_boxes: int = 7,
    batch_size: int = 1,
    data_rows=None,
):
    """
    Loads clean/corrupt prompt pairs from a JSONL datafile.

    Args:
        model: model to use for tokenization compatibility.
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        num_boxes: unused legacy argument, retained for API compatibility.
        batch_size: batch size to use for the dataloader.
        data_rows: optional in-memory training data. This may be a DataLoader,
            Dataset, list of dict rows, or pretokenized mapping with
            base/source tokens and labels. If only clean/base prompts or clean
            tokens are provided, counterfactual prompts are synthesized from the
            anchor prompts using other generated samples as donors.
    """

    del model, num_boxes

    if isinstance(data_rows, DataLoader):
        return data_rows

    if data_rows is not None:
        dataset = _coerce_in_memory_dataset(tokenizer, data_rows, num_samples)
    else:
        if datafile is None:
            raise ValueError("Either `datafile` or `data_rows` must be provided.")
        rows = _read_jsonl_rows(datafile, num_samples)
        dataset = _build_dataset_from_rows(tokenizer, rows)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_caches(
    model: HookedTransformer,
    dataloader: torch.utils.data.DataLoader,
    hook_points: Optional[Sequence[str]] = None,
):
    """
    Computes the clean and corrupt caches for the model.

    Args:
        model: model to compute the caches for.
        dataloader: dataloader containing clean and corrupt inputs.
    """

    if hook_points is None:
        hook_points = [_z_hook_name(layer) for layer in range(_model_num_layers(model))]
    hook_name_set = set(hook_points)

    apply_softmax = torch.nn.Softmax(dim=-1)
    clean_cache, corrupt_cache = {}, {}
    clean_logit_outputs, corrupt_logit_outputs = defaultdict(dict), defaultdict(dict)

    with torch.no_grad():
        for bi, inp in tqdm(enumerate(dataloader), desc="Clean cache"):
            batch_size = inp["base_tokens"].size(0)
            inp = _move_batch_to_device(inp, model)

            logits, cache = model.run_with_cache(
                inp["base_tokens"],
                names_filter=lambda name: name in hook_name_set,
                return_type="logits",
            )

            clean_cache[bi] = {
                hook_name: cache[hook_name].detach().cpu() for hook_name in hook_points
            }

            for i in range(batch_size):
                logits_i = apply_softmax(
                    logits[i, int(inp["base_last_token_indices"][i].item())]
                )
                clean_logit_outputs[bi][i] = logits_i[int(inp["labels"][i].item())].item()

            del logits, cache
            torch.cuda.empty_cache()
        print("CLEAN CACHE COMPUTED")

        for bi, inp in tqdm(enumerate(dataloader), desc="Corrupt cache"):
            batch_size = inp["source_tokens"].size(0)
            inp = _move_batch_to_device(inp, model)

            logits, cache = model.run_with_cache(
                inp["source_tokens"],
                names_filter=lambda name: name in hook_name_set,
                return_type="logits",
            )

            corrupt_cache[bi] = {
                hook_name: cache[hook_name].detach().cpu() for hook_name in hook_points
            }

            for i in range(batch_size):
                logits_i = apply_softmax(
                    logits[i, int(inp["source_last_token_indices"][i].item())]
                )
                corrupt_logit_outputs[bi][i] = logits_i[int(inp["labels"][i].item())].item()

            del logits, cache
            torch.cuda.empty_cache()
        print("CORRUPT CACHE COMPUTED")

    return (
        clean_cache,
        corrupt_cache,
        clean_logit_outputs,
        corrupt_logit_outputs,
        hook_points,
    )


def patching_sender_heads(
    z: torch.Tensor,
    hook,
    model: HookedTransformer = None,
    clean_cache: Dict[str, torch.Tensor] = None,
    corrupt_cache: Dict[str, torch.Tensor] = None,
    base_tokens: torch.Tensor = None,
    sender_layer: int = None,
    sender_head: int = None,
    clean_last_token_indices: torch.Tensor = None,
    corrupt_last_token_indices: torch.Tensor = None,
    clean_positions: torch.Tensor = None,
    corrupt_positions: torch.Tensor = None,
    rel_pos: int = None,
    batch_size: int = None,
):
    """
    Patches head outputs at `hook_z` and stores the receiver activations via
    `run_with_cache` in the caller.
    """

    del model, clean_cache

    corrupt_head_outputs = corrupt_cache[hook.name].to(z.device)
    layer_idx = _layer_from_hook_name(hook.name)

    for bi in range(batch_size):
        clean_last_idx = int(clean_last_token_indices[bi].item())

        if clean_positions is not None and corrupt_positions is not None:
            sender_head_pos_in_clean = int(clean_positions[bi].item())
            sender_head_pos_in_corrupt = int(corrupt_positions[bi].item())
        elif rel_pos == -1:
            sender_head_pos_in_clean = compute_prev_query_box_pos(
                base_tokens[bi], clean_last_idx
            )
            sender_head_pos_in_corrupt = random.choice(range(6, 49, 7))
        else:
            sender_head_pos_in_clean = clean_last_idx - rel_pos
            sender_head_pos_in_corrupt = int(corrupt_last_token_indices[bi].item()) - rel_pos

        if layer_idx == sender_layer:
            z[bi, sender_head_pos_in_clean, sender_head] = corrupt_head_outputs[
                bi, sender_head_pos_in_corrupt, sender_head
            ]

    return z


def patching_receiver_heads(
    act: torch.Tensor,
    hook,
    model: HookedTransformer = None,
    base_tokens: torch.Tensor = None,
    patched_cache=None,
    receiver_heads: List[Tuple[int, int]] = None,
    clean_last_token_indices: torch.Tensor = None,
    clean_positions: torch.Tensor = None,
    rel_pos: int = None,
    batch_size: int = None,
):
    """
    Patches the input of the receiver head, i.e. key, query or value vectors.
    """

    current_layer = _layer_from_hook_name(hook.name)
    receiver_heads_in_curr_layer = [
        head_idx for layer_idx, head_idx in receiver_heads if layer_idx == current_layer
    ]
    if not receiver_heads_in_curr_layer:
        return act

    is_kv_receiver = hook.name.endswith("hook_k") or hook.name.endswith("hook_v")
    model_num_heads = int(getattr(model.cfg, "n_heads", act.shape[2]))
    model_num_kv_heads = int(
        getattr(
            model.cfg,
            "n_key_value_heads",
            getattr(model.cfg, "n_kv_heads", act.shape[2]),
        )
    )
    if is_kv_receiver and act.shape[2] != model_num_heads and model_num_kv_heads > 0:
        kv_group_size = max(1, model_num_heads // model_num_kv_heads)
        receiver_heads_in_curr_layer = sorted(
            {
                min(model_num_kv_heads - 1, head_idx // kv_group_size)
                for head_idx in receiver_heads_in_curr_layer
            }
        )

    patched_head_outputs = patched_cache[hook.name]
    if patched_head_outputs.device != act.device:
        patched_head_outputs = patched_head_outputs.to(act.device)

    for receiver_head in receiver_heads_in_curr_layer:
        for bi in range(batch_size):
            clean_last_idx = int(clean_last_token_indices[bi].item())
            if clean_positions is not None:
                receiver_head_pos_in_clean = int(clean_positions[bi].item())
            elif rel_pos == -1:
                receiver_head_pos_in_clean = compute_prev_query_box_pos(
                    base_tokens[bi], clean_last_idx
                )
            else:
                receiver_head_pos_in_clean = clean_last_idx - rel_pos

            act[bi, receiver_head_pos_in_clean, receiver_head] = patched_head_outputs[
                bi, receiver_head_pos_in_clean, receiver_head
            ]

    return act


def get_receiver_layers(
    model: HookedTransformer, receiver_heads: List[Tuple[int, int]], composition: str
):
    """
    Gets the receiver hook names from the receiver heads.

    Args:
        model: model under investigation.
        receiver_heads: receiver heads.
        composition: attention component to use for the receiver heads (k/q/v).
    """

    del model

    return sorted(
        {
            _attn_hook_name(composition, layer_idx)
            for layer_idx, _ in receiver_heads
        }
    )


def load_eval_data(
    tokenizer: PreTrainedTokenizerBase, datafile: str, num_samples: int, batch_size: int
):
    """
    Loads the dataset for evaluation.
    """

    print("Loading dataset...")
    rows = _read_jsonl_rows(datafile, num_samples)
    prompts = [_clean_prompt(row) for row in rows]
    labels = [_encode_single_token_answer(tokenizer, _clean_answer(row)) for row in rows]
    input_ids, last_token_indices = _encode_prompts(tokenizer, prompts)

    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "last_token_indices": last_token_indices,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    ).with_format("torch")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_ablation_data(
    tokenizer: PreTrainedTokenizerBase,
    datafile: str,
    num_samples: int,
    batch_size: int,
    num_boxes: int = 7,
):
    """
    Loads the dataset for ablation.
    """

    del num_boxes

    rows = _read_jsonl_rows(datafile, num_samples)
    prompts = [_clean_prompt(row) for row in rows]
    input_ids, last_token_indices = _encode_prompts(tokenizer, prompts)

    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "last_token_indices": last_token_indices,
        }
    ).with_format("torch")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_mean_activations(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    datafile: str,
    num_samples: int,
    batch_size: int,
):
    """
    Computes the mean activations of every attention head at all positions.
    """

    print("Computing mean activations...")
    ablation_dataloader = load_ablation_data(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    modules = [_z_hook_name(layer) for layer in range(_model_num_layers(model))]
    module_set = set(modules)

    mean_activations: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for inp in tqdm(ablation_dataloader):
            inp = _move_batch_to_device(inp, model)

            _, cache = model.run_with_cache(
                inp["input_ids"],
                names_filter=lambda name: name in module_set,
                return_type="logits",
            )

            for layer in modules:
                layer_acts = cache[layer].detach().cpu()
                if layer in mean_activations:
                    mean_activations[layer] += torch.sum(layer_acts, dim=0)
                else:
                    mean_activations[layer] = torch.sum(layer_acts, dim=0)

            del cache
            torch.cuda.empty_cache()

        for layer in modules:
            mean_activations[layer] /= len(ablation_dataloader.dataset)

    return mean_activations, modules


def mean_ablate(
    z: torch.Tensor,
    hook,
    model: HookedTransformer = None,
    circuit_components: Dict[int, Dict[str, List[int]]] = None,
    mean_activations: Dict[str, torch.Tensor] = None,
    input_tokens: torch.Tensor = None,
    ablate_non_vital_pos: bool = None,
):
    """
    Ablates the model components that are not present in `circuit_components`
    by substituting their output with their corresponding mean activations.
    """

    mean_act = mean_activations[hook.name].to(z.device)
    pad_token_id = getattr(model.tokenizer, "pad_token_id", None)

    for bi in range(z.size(0)):
        if pad_token_id is None:
            last_pos = z.size(1) - 1
        else:
            non_pad = (input_tokens[bi] != pad_token_id).nonzero(as_tuple=False)
            last_pos = int(non_pad[-1].item()) if non_pad.numel() else z.size(1) - 1

        prev_query_box_pos = compute_prev_query_box_pos(input_tokens[bi], last_pos)

        for token_pos in range(last_pos + 1):
            if (
                token_pos != prev_query_box_pos
                and token_pos != last_pos - 2
                and token_pos != last_pos
                and ablate_non_vital_pos
            ):
                z[bi, token_pos, :] = mean_act[token_pos, :]
            elif token_pos == prev_query_box_pos:
                for head_idx in range(_model_num_heads(model)):
                    if head_idx not in circuit_components[-1][hook.name]:
                        z[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]
            elif token_pos == last_pos - 2:
                for head_idx in range(_model_num_heads(model)):
                    if head_idx not in circuit_components[2][hook.name]:
                        z[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]
            elif token_pos == last_pos:
                for head_idx in range(_model_num_heads(model)):
                    if head_idx not in circuit_components[0][hook.name]:
                        z[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]

    return z


def eval_circuit_performance(
    model: HookedTransformer,
    dataloader: torch.utils.data.DataLoader,
    modules: List[str],
    circuit_components: Dict[int, Dict[str, List[int]]],
    mean_activations: Dict[str, torch.Tensor],
    ablate_non_vital_pos: bool = True,
):
    """
    Evaluates the performance of the model/circuit.
    """

    correct_count, total_count = 0, 0
    hooks = [
        (
            module_name,
            partial(
                mean_ablate,
                model=model,
                circuit_components=circuit_components,
                mean_activations=mean_activations,
                ablate_non_vital_pos=ablate_non_vital_pos,
            ),
        )
        for module_name in modules
    ]

    with torch.no_grad():
        for inp in tqdm(dataloader):
            inp = _move_batch_to_device(inp, model)

            outputs = model.run_with_hooks(
                inp["input_ids"],
                fwd_hooks=[
                    (
                        module_name,
                        partial(
                            hook_fn,
                            input_tokens=inp["input_ids"],
                        ),
                    )
                    for module_name, hook_fn in hooks
                ],
                return_type="logits",
            )

            for bi in range(inp["labels"].size(0)):
                label = int(inp["labels"][bi].item())
                last_idx = int(inp["last_token_indices"][bi].item())
                pred = int(torch.argmax(outputs[bi, last_idx]).item())

                if label == pred:
                    correct_count += 1
                total_count += 1

            del outputs
            torch.cuda.empty_cache()

    return round(correct_count / total_count, 2)


def get_circuit(
    model: HookedTransformer,
    circuit_root_path: str,
    n_value_fetcher: int,
    n_pos_trans: int,
    n_pos_detect: int,
    n_struct_read: int,
):
    """
    Computes the circuit components.
    """

    circuit_components = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}

    value_fetcher_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "value_fetcher.pt"),
        k=n_value_fetcher,
        largest=False,
    )
    pos_transmitter_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "pos_transmitter.pt"),
        k=n_pos_trans,
        largest=False,
    )
    pos_detector_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "pos_detector.pt"),
        k=n_pos_detect,
        largest=False,
    )
    struct_reader_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "struct_reader.pt"),
        k=n_struct_read,
        largest=False,
    )

    intersection = [head for head in value_fetcher_heads if head in pos_transmitter_heads]
    for head in intersection:
        value_fetcher_heads.remove(head)

    for layer_idx, head in value_fetcher_heads:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in pos_transmitter_heads:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in pos_detector_heads:
        circuit_components[2][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in struct_reader_heads:
        circuit_components[-1][_receiver_key(layer_idx)].append(head)

    return (
        circuit_components,
        value_fetcher_heads,
        pos_transmitter_heads,
        pos_detector_heads,
        struct_reader_heads,
    )


def get_random_circuit(
    model: HookedTransformer,
    circuit: Dict[str, List[Tuple[int, int]]],
):
    """
    Computes a random circuit with same #heads in each group as in the circuit.
    """

    random_circuit = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}

    num_heads = _model_num_heads(model)
    num_layers = _model_num_layers(model)
    n_value_fetcher = len(circuit["value_fetcher"])
    n_pos_transmitter = len(circuit["pos_transmitter"])
    n_pos_detector = len(circuit["pos_detector"])
    n_struct_reader = len(circuit["struct_reader"])

    heads_at_last_pos = np.random.choice(
        list(range(num_heads * num_layers)),
        n_value_fetcher + n_pos_transmitter,
        replace=False,
    )
    heads_at_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)),
        n_pos_detector,
        replace=False,
    )
    heads_at_prev_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)),
        n_struct_reader,
        replace=False,
    )

    heads_at_last_pos = [[idx // num_heads, idx % num_heads] for idx in heads_at_last_pos]
    heads_at_query_box_pos = [[idx // num_heads, idx % num_heads] for idx in heads_at_query_box_pos]
    heads_at_prev_query_box_pos = [
        [idx // num_heads, idx % num_heads] for idx in heads_at_prev_query_box_pos
    ]

    for layer_idx, head in heads_at_last_pos:
        random_circuit[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in heads_at_query_box_pos:
        random_circuit[2][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in heads_at_prev_query_box_pos:
        random_circuit[-1][_receiver_key(layer_idx)].append(head)

    return random_circuit


def compute_pair_drop_values(
    model: HookedTransformer,
    heads: List[Tuple[int, int]],
    circuit_components: Dict[int, Dict[str, List[int]]],
    dataloader: torch.utils.data.DataLoader,
    modules: List[str],
    mean_activations: Dict[str, torch.Tensor],
    rel_pos: int = 0,
):
    """
    Computes the pair drop values for the given heads.
    """

    greedy_res = defaultdict(lambda: defaultdict(float))

    for layer_idx_1, head_1 in tqdm(heads, total=len(heads), desc="Pair drop values"):
        layer_1 = _receiver_key(layer_idx_1)
        circuit_components[rel_pos][layer_1].remove(head_1)

        for layer_idx_2, head_2 in heads:
            layer_2 = _receiver_key(layer_idx_2)

            if greedy_res[(layer_2, head_2)][(layer_1, head_1)] > 0.0:
                continue
            if layer_1 != layer_2 or head_1 != head_2:
                circuit_components[rel_pos][layer_2].remove(head_2)

            greedy_res[(layer_1, head_1)][(layer_2, head_2)] = eval_circuit_performance(
                model, dataloader, modules, circuit_components, mean_activations
            )
            if layer_1 != layer_2 or head_1 != head_2:
                circuit_components[rel_pos][layer_2].append(head_2)

        circuit_components[rel_pos][layer_1].append(head_1)

    res = defaultdict(lambda: defaultdict(float))
    for k in greedy_res:
        for k_2 in greedy_res[k]:
            if greedy_res[k][k_2] > 0.0:
                res[str(k)][str(k_2)] = greedy_res[k][k_2]
                res[str(k_2)][str(k)] = greedy_res[k][k_2]

    return res


def get_head_significance_score(
    model: HookedTransformer,
    heads: List[Tuple[int, int]],
    ranked: Dict,
    percentage: float,
    circuit_components: Dict[int, Dict[str, List[int]]],
    dataloader: torch.utils.data.DataLoader,
    modules: List[str],
    mean_activations: Dict[str, torch.Tensor],
    rel_pos: int,
):
    """
    Computes the head significance score for the given heads.
    """

    res = {}

    for layer_idx, head in tqdm(heads, total=len(heads), desc="Head significance score"):
        layer = _receiver_key(layer_idx)

        for r in ranked[str((layer, head))][
            : math.ceil(percentage * len(ranked.values()))
        ]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].remove(top_head)

        before = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        circuit_components[rel_pos][layer].remove(head)
        after = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        res[(layer, head)] = (before, after)

        for r in ranked[str((layer, head))][
            : math.ceil(percentage * len(ranked.values()))
        ]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].append(top_head)
        circuit_components[rel_pos][layer].append(head)

    return res


def get_final_circuit(model: HookedTransformer, circuit_heads: Dict[str, List[Tuple[int, int]]]):
    """
    Computes the final circuit.
    """

    del model

    circuit_components = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}

    for layer_idx, head in circuit_heads["value_fetcher"]:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in circuit_heads["pos_transmitter"]:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in circuit_heads["pos_detector"]:
        circuit_components[2][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in circuit_heads["struct_reader"]:
        circuit_components[-1][_receiver_key(layer_idx)].append(head)

    return circuit_components
