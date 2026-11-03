from __future__ import annotations

from exp3.head_config import (
    ANSWER_PREFILL_RE,
    C_TO_A_MILK_ANSWER,
    C_TO_A_MILK_BASE_PROMPT,
    C_TO_A_MILK_CANDIDATES,
    C_TO_A_MILK_MEASURED_OBJECT,
    C_TO_A_MILK_SOURCE_PROMPT,
    DictTensorDataset,
    MODEL_NAME_ALIASES,
    SITE_CHOICES,
    SYSTEM_PROMPT,
)

from dynamicet_data import datascience
from dynamicet_data.box import Statement
from exp2 import pp_utils
import contextlib
import io
import re
import torch

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
