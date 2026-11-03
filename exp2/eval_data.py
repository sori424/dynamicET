from __future__ import annotations

from exp2.eval_core import (
    DictTensorDataset,
    answer_token_ids,
    answer_words_from_tokens,
    decode_prompts_from_tokens,
    encode_prompt_texts,
    first_answer_token_id,
    format_prompt_for_model,
    last_token_indices,
    last_token_indices_from_attention_mask,
    padded_token_id_tensor,
    position_fields_from_tokens,
    resolve_prompt_format,
    span_starts,
)

import contextlib
import io

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
        # Keep box positions fixed while changing their object assignments.
        # Shifting both lists merely reorders sentences and preserves answers.
        entities=list(range(num_entities)),
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
