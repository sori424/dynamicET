import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import fire
import numpy as np

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch

from exp1.tracing_utils import (
    build_tracing_dataset,
    find_correct_samples,
    load_model,
    normalize_trace_category,
    resolve_query_trace_mode,
    resolve_trace_modes,
    run_tracing_experiment,
)


def configure_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


REPO_ROOT = configure_repo_root()
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "tracing_entity"
DEFAULT_DEVICE_ID = "0"
SYSTEM_PROMPT = "Respond with a single word only. No punctuation, no explanation."
ANSWER_PREFILL_RE = re.compile(r"\s*Answer:\s*$", re.IGNORECASE)
MODEL_NAME_ALIASES = {
    "gemma9b": "google/gemma-2-9b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama8b": "llama-8b-hf"
}
PROMPT_FORMATS = {
    "raw",
    "auto",
    "chat-template",
    "chat-prefill",
    "gemma-it-prefill",
    "llama-instruct-prefill",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("tracing")

def resolve_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def infer_prompt_format(model_name: str) -> str:
    resolved_name = resolve_model_name(model_name)
    model_name_lower = resolved_name.lower()
 
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
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"Unknown prompt_format {prompt_format!r}. Expected one of {sorted(PROMPT_FORMATS)}."
        )
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


def _split_assistant_prefill(prompt: str) -> tuple[str, str]:
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

    raise ValueError(f"Unknown prompt format: {prompt_format}")


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def _last_token_indices(tokens: torch.Tensor, pad_token_id: Optional[int]) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected token tensor [batch, seq], got {tuple(tokens.shape)}.")
    if pad_token_id is None:
        return torch.full((tokens.shape[0],), tokens.shape[1] - 1, dtype=torch.long)
    non_pad = tokens.ne(int(pad_token_id))
    if not torch.all(non_pad.any(dim=1)):
        raise ValueError("Found an all-padding prompt.")
    positions_from_end = torch.flip(non_pad, dims=[1]).to(torch.long).argmax(dim=1)
    return tokens.shape[1] - 1 - positions_from_end


def _last_token_indices_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected attention mask [batch, seq], got {tuple(attention_mask.shape)}."
        )
    attended = attention_mask.to(dtype=torch.bool)
    if not torch.all(attended.any(dim=1)):
        raise ValueError("Found an all-padding prompt.")
    positions_from_end = torch.flip(attended, dims=[1]).to(torch.long).argmax(dim=1)
    return attention_mask.shape[1] - 1 - positions_from_end


def _encode_prompt_texts(tokenizer, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_pad_token(tokenizer)
    encoded = tokenizer(
        list(prompts),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=True,
    )
    return encoded["input_ids"].to(torch.long), encoded["attention_mask"].to(torch.long)


def _answer_words_from_tokens(tokenizer, answer_tokens: torch.Tensor) -> list[list[str]]:
    return [
        [
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            for token_id in row
        ]
        for row in answer_tokens.detach().cpu().tolist()
    ]


def _first_answer_token_id(tokenizer, formatted_prompt: str, answer_word: str) -> int:
    answer_word = answer_word.strip()
    if not answer_word:
        raise ValueError("Cannot encode an empty answer word.")

    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"]
    suffixes = (
        (answer_word, f" {answer_word}")
        if formatted_prompt[-1:].isspace()
        else (f" {answer_word}", answer_word)
    )
    for suffix in suffixes:
        full_ids = tokenizer(
            formatted_prompt + suffix,
            add_special_tokens=False,
        )["input_ids"]
        if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
            return int(full_ids[len(prompt_ids)])

    for suffix in suffixes:
        token_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        if token_ids:
            return int(token_ids[0])

    raise ValueError(f"Could not encode answer {answer_word!r}.")


def _format_answer_tokens(tokenizer, prompts: Sequence[str], answer_tokens: torch.Tensor) -> torch.Tensor:
    answer_words = _answer_words_from_tokens(tokenizer, answer_tokens)
    return torch.tensor(
        [
            [
                _first_answer_token_id(tokenizer, prompt, answer_word)
                for answer_word in row
            ]
            for prompt, row in zip(prompts, answer_words)
        ],
        dtype=torch.long,
    )


def apply_prompt_format_to_dataset(*, dataset, tokenizer, prompt_format: str):
    if prompt_format == "raw":
        return dataset

    payload = dict(dataset.payload)
    clean_prompts = [
        format_prompt_for_model(tokenizer, prompt, prompt_format)
        for prompt in payload["clean_prompts"]
    ]
    source_prompts = [
        format_prompt_for_model(tokenizer, prompt, prompt_format)
        for prompt in payload["source_prompts"]
    ]
    clean_tokens, clean_attention_mask = _encode_prompt_texts(tokenizer, clean_prompts)
    source_tokens, source_attention_mask = _encode_prompt_texts(tokenizer, source_prompts)

    payload.update(
        {
            "clean_tokens": clean_tokens,
            "source_tokens": source_tokens,
            "clean_answer_tokens": _format_answer_tokens(
                tokenizer,
                clean_prompts,
                payload["clean_answer_tokens"],
            ),
            "source_answer_tokens": _format_answer_tokens(
                tokenizer,
                source_prompts,
                payload["source_answer_tokens"],
            ),
            "clean_query_positions": _last_token_indices_from_attention_mask(
                clean_attention_mask,
            ),
            "source_query_positions": _last_token_indices_from_attention_mask(
                source_attention_mask,
            ),
            "clean_prompts": clean_prompts,
            "source_prompts": source_prompts,
        }
    )
    return dataset.__class__(payload)
