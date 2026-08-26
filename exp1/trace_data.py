from __future__ import annotations

from exp1.trace_contexts import (
    _default_binding_selector_extractor,
    _hide_torchvision_imports,
    _stack_tokens,
    build_trace_contexts,
    default_template_context,
    normalize_trace_category,
    resolve_prompt_query_names,
    resolve_source_swap_pair,
)

from torch.utils.data import DataLoader, Dataset
import torch

def _decode_prompt(tokenizer, token_ids: torch.Tensor, query_position: int) -> str:
    return tokenizer.decode(
        token_ids[: query_position + 1].tolist(),
        skip_special_tokens=True,
    )


def _selector_positions(token_maps) -> Dict[str, torch.Tensor]:
    selector = _default_binding_selector_extractor(token_maps)
    return {
        category: torch.stack([positions.detach().cpu() for positions in category_positions], dim=1)
        for category, category_positions in selector.items()
        if category_positions
    }


def summarize_selector_positions(
    *,
    tokenizer,
    tokens: torch.Tensor,
    selector_positions: Dict[str, torch.Tensor],
    sample_idx: int = 0,
) -> Dict[str, List[Dict[str, object]]]:
    summaries: Dict[str, List[Dict[str, object]]] = {}
    for category, positions_tensor in selector_positions.items():
        entries = []
        for idx, pos in enumerate(positions_tensor[sample_idx].tolist()):
            token_id = int(tokens[sample_idx, pos].item())
            decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            entries.append(
                {
                    "index": idx,
                    "pos": int(pos),
                    "tok_id": token_id,
                    "decoded": decoded,
                    "stripped": decoded.strip(),
                }
            )
        summaries[category] = entries
    return summaries


def _build_prompt_batch(
    *,
    vocab,
    contexts,
    num_entities,
    batch_size: int,
    prompt_id_start: int,
    query_names: Sequence[int],
    raw_query_names: Sequence[str],
    vocab_tag: str,
):
    with _hide_torchvision_imports():
        from coref.representations import build_example_refactored

    token_rows = []
    answer_rows = []
    query_name_rows = []
    query_positions = []
    prompts = []
    position_debug = None

    for offset, (context, query_name, raw_query_name) in enumerate(
        zip(contexts, query_names, raw_query_names)
    ):
        template_context = default_template_context(vocab_tag, int(query_name))
        template_context["raw_query_name"] = raw_query_name
        tokens, answer_tokens, query_names, token_maps = build_example_refactored(
            batch_size=1,
            vocab=vocab,
            context=context,
            num_entities=num_entities,
            prompt_id_start=prompt_id_start + offset,
            template_context=template_context,
            template_type="normal",
        )
        query_position = int(token_maps["prompt"][0, 1].item()) - 1
        token_rows.append(tokens[0].tolist())
        answer_rows.append(answer_tokens[0].detach().cpu())
        query_name_rows.append(int(query_names[0].item()))
        query_positions.append(query_position)
        prompts.append(_decode_prompt(vocab.tokenizer, tokens[0], query_position))

        if position_debug is None:
            selector_positions = _selector_positions(token_maps)
            position_debug = summarize_selector_positions(
                tokenizer=vocab.tokenizer,
                tokens=tokens,
                selector_positions=selector_positions,
            )

    return {
        "tokens": _stack_tokens(vocab, token_rows),
        "answer_tokens": torch.stack(answer_rows),
        "query_names": torch.tensor(query_name_rows),
        "query_positions": torch.tensor(query_positions),
        "prompts": prompts,
        "position_debug": position_debug or {},
    }


class TracePromptDataset(Dataset):
    def __init__(self, payload: Dict[str, object]):
        self.payload = payload
        tensor_keys = [key for key, value in payload.items() if isinstance(value, torch.Tensor)]
        if not tensor_keys:
            raise ValueError("TracePromptDataset expects at least one tensor field.")
        self.size = int(payload[tensor_keys[0]].shape[0])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = {}
        for key, value in self.payload.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[index]
            else:
                item[key] = value[index]
        return item

    def select(self, indices: Sequence[int]) -> "TracePromptDataset":
        selected = {}
        for key, value in self.payload.items():
            if isinstance(value, torch.Tensor):
                selected[key] = value[indices]
            else:
                selected[key] = [value[idx] for idx in indices]
        return TracePromptDataset(selected)


def build_tracing_dataset(
    *,
    vocab_tag: str,
    model_tag: str,
    split: str,
    tokenizer=None,
    trace_type: str,
    num_samples: int,
    prompt_id_start: int,
    query_name: int,
    names_trace_mode: str = "context_only",
    object_trace_mode: str = "target_box",
    query_trace_mode: str = "same_query",
    oversample_factor: int = 4,
    num_entities: int = 3,
    swap_box_a: Optional[int] = None,
    swap_box_b: Optional[int] = None,
):
    trace_category = normalize_trace_category(trace_type, vocab_tag)
    source_swap_pair = None
    if swap_box_a is not None or swap_box_b is not None:
        if vocab_tag != "BOXES":
            raise ValueError(
                "swap_box_a/swap_box_b are only available for vocab_tag='BOXES'."
            )
        source_swap_pair = resolve_source_swap_pair(
            query_name=query_name,
            num_entities=num_entities,
            swap_box_a=swap_box_a,
            swap_box_b=swap_box_b,
        )
    elif vocab_tag == "BOXES":
        source_swap_pair = resolve_source_swap_pair(
            query_name=query_name,
            num_entities=num_entities,
        )
    with _hide_torchvision_imports():
        from datasets.api import load_vocab

    vocab = load_vocab(vocab_tag, model_tag, split=split, tokenizer=tokenizer)
    clean_context, source_context = build_trace_contexts(
        vocab,
        vocab_tag,
        trace_category,
        query_name,
        source_swap_pair=source_swap_pair,
        names_trace_mode=names_trace_mode,
        object_trace_mode=object_trace_mode,
        num_entities=num_entities,
    )
    clean_query_name, source_query_name, query_trace_mode = resolve_prompt_query_names(
        vocab_tag=vocab_tag,
        query_name=query_name,
        num_entities=num_entities,
        query_trace_mode=query_trace_mode,
        source_swap_pair=source_swap_pair,
    )

    candidate_count = max(num_samples, int(num_samples * oversample_factor))
    clean_query_names = [clean_query_name] * candidate_count
    source_query_names = [source_query_name] * candidate_count
    clean_raw_query_names = [
        vocab.fetch_shuffled_name(clean_query_name, prompt_id_start + offset)
        for offset in range(candidate_count)
    ]
    source_raw_query_names = [
        vocab.fetch_shuffled_name(source_query_name, prompt_id_start + offset)
        for offset in range(candidate_count)
    ]
    clean_batch = _build_prompt_batch(
        vocab=vocab,
        contexts=[clean_context for _ in range(candidate_count)],
        num_entities=num_entities,
        batch_size=candidate_count,
        prompt_id_start=prompt_id_start,
        query_names=clean_query_names,
        raw_query_names=clean_raw_query_names,
        vocab_tag=vocab_tag,
    )
    source_batch = _build_prompt_batch(
        vocab=vocab,
        contexts=[source_context for _ in range(candidate_count)],
        num_entities=num_entities,
        batch_size=candidate_count,
        prompt_id_start=prompt_id_start,
        query_names=source_query_names,
        raw_query_names=source_raw_query_names,
        vocab_tag=vocab_tag,
    )

    payload = {
        "prompt_ids": torch.arange(prompt_id_start, prompt_id_start + candidate_count),
        "query_names": source_batch["query_names"],
        "clean_query_names": clean_batch["query_names"],
        "source_query_names": source_batch["query_names"],
        "clean_tokens": clean_batch["tokens"],
        "source_tokens": source_batch["tokens"],
        "clean_answer_tokens": clean_batch["answer_tokens"],
        "source_answer_tokens": source_batch["answer_tokens"],
        "clean_query_positions": clean_batch["query_positions"],
        "source_query_positions": source_batch["query_positions"],
        "clean_prompts": clean_batch["prompts"],
        "source_prompts": source_batch["prompts"],
    }
    dataset = TracePromptDataset(payload)
    metadata = {
        "vocab_tag": vocab_tag,
        "split": split,
        "trace_type": trace_type,
        "trace_category": trace_category,
        "num_entities": num_entities,
        "query_name": query_name,
        "clean_query_name": clean_query_name,
        "source_query_name": source_query_name,
        "names_trace_mode": names_trace_mode,
        "object_trace_mode": object_trace_mode,
        "query_trace_mode": query_trace_mode,
        "prompt_id_start": prompt_id_start,
        "num_candidates": candidate_count,
        "source_swap_pair": None if source_swap_pair is None else list(source_swap_pair),
        "explicit_swap_pair": swap_box_a is not None,
        "clean_context": [tuple(stmt) for stmt in clean_context],
        "source_context": [tuple(stmt) for stmt in source_context],
        "position_debug": {
            "clean": clean_batch["position_debug"],
            "source": source_batch["position_debug"],
        },
    }
    return vocab, dataset, metadata
