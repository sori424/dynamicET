import json
import contextlib
import importlib.util
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# bitsandbytes calls torch.compile at import time through TransformerLens.
# This environment has a mismatched Torch Inductor install, so disable Dynamo
# before importing anything that can pull in transformer_lens.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")


@contextlib.contextmanager
def _hide_torchvision_imports():
    original_find_spec = importlib.util.find_spec

    def find_spec_without_torchvision(name, package=None):
        if name == "torchvision" or name.startswith("torchvision."):
            return None
        return original_find_spec(name, package)

    importlib.util.find_spec = find_spec_without_torchvision
    try:
        yield
    finally:
        importlib.util.find_spec = original_find_spec

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.common import lookup_answer_logits


TRACE_CATEGORY_ALIASES = {
    "character": "names",
    "characters": "names",
    "context": "names",
    "name": "names",
    "names": "names",
    "object": "countries",
    "objects": "countries",
    "country": "countries",
    "countries": "countries",
    "query": "question_query_box",
    "question": "question_query_box",
    "question_query_box": "question_query_box",
    "swap": "swap_boxes",
    "swap_box": "swap_boxes",
    "swap_boxes": "swap_boxes",
    "swap_sentence": "swap_boxes",
    "state": "swap_boxes",
}


VALID_NAMES_TRACE_MODES = {
    "context_only",
    "context_target_box",
    "context_target_box_and_swap_query_box",
    "context_and_swap",
    "context_and_swap_target_box",
    "swap_only",
    "swap_target_box",
    "swap_target_box_to_remaining_context_box",
    "context_target_object_and_swap_target_box_to_remaining_context_box",
}
VALID_OBJECT_TRACE_MODES = {
    "target_box",
    "query_box",
    "context_target_object_and_swap_query_box",
    "swap_target_object_with_remaining_context_object",
}
QUERY_TRACE_MODE_ALIASES = {
    "default": "same_query",
    "same": "same_query",
    "same_query": "same_query",
    "swap_partner": "swap_partner",
    "partner": "swap_partner",
    "swap_partner_box": "swap_partner",
}
VALID_QUERY_TRACE_MODES = set(QUERY_TRACE_MODE_ALIASES.values())


def _get_act_name(component: str, layer_idx: int) -> str:
    return f"blocks.{layer_idx}.hook_{component}"


def _default_binding_selector_extractor(token_maps):
    def flatten(items):
        return [value for group in items for value in group]

    binding_statements = [
        statement
        for statement in token_maps["context"]
        if "subject" in statement and ("country" in statement or "object" in statement)
    ]

    def get_binding_positions(statement, primary_key, fallback_key=None):
        key = primary_key if primary_key in statement else fallback_key
        if key is None:
            raise KeyError(primary_key)
        value = statement[key]
        if isinstance(value, torch.Tensor):
            return [value[:, 0]]
        return [part[:, 0] for part in value]

    countries = flatten(
        [
            get_binding_positions(statement, "country", "object")
            for statement in binding_statements
        ]
    )
    swap_boxes = flatten(
        [
            get_binding_positions(statement, "left_box")
            + get_binding_positions(statement, "right_box")
            for statement in token_maps["context"]
            if "left_box" in statement and "right_box" in statement
        ]
    )
    selector = {
        "names": flatten(
            [get_binding_positions(statement, "subject") for statement in binding_statements]
        ),
        "countries": countries,
        "object": countries,
        "swap_boxes": swap_boxes,
    }
    if "qn_subject" in token_maps:
        selector["question_query_box"] = [token_maps["qn_subject"][:, 0]]
    return selector


def _stack_tokens(vocab, tokens_list):
    longest_length = max(len(tokens) for tokens in tokens_list)
    pad_token_id = vocab.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = vocab.tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = getattr(vocab.tokenizer, "pad_token_type_id", 0)
    return torch.tensor(
        [
            tokens + [pad_token_id] * (longest_length - len(tokens))
            for tokens in tokens_list
        ]
    )


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_model(model_name: str, num_devices: int = 1):
    with _hide_torchvision_imports():
        import coref.models as models

    return models.fetch_model(model_name, num_devices=num_devices)


def normalize_trace_category(trace_type: str, vocab_tag: str) -> str:
    normalized = TRACE_CATEGORY_ALIASES.get(trace_type.lower())
    if normalized is None:
        raise ValueError(
            f"Unknown trace type {trace_type!r}. "
            f"Expected one of {sorted(TRACE_CATEGORY_ALIASES)}."
        )
    if normalized == "swap_boxes" and vocab_tag != "BOXES":
        raise ValueError("Swap tracing is only available for vocab_tag='BOXES'.")
    return normalized


def resolve_trace_modes(
    trace_category: str,
    *,
    names_trace_mode: str = "context_only",
    object_trace_mode: str = "target_box",
) -> Tuple[str, str]:
    resolved_names_trace_mode = names_trace_mode
    resolved_object_trace_mode = object_trace_mode

    if trace_category == "names":
        if (
            names_trace_mode == "context_only"
            and object_trace_mode in VALID_NAMES_TRACE_MODES
        ):
            resolved_names_trace_mode = object_trace_mode
            resolved_object_trace_mode = "target_box"
        elif names_trace_mode not in VALID_NAMES_TRACE_MODES:
            raise ValueError(
                f"Unknown names_trace_mode {names_trace_mode!r}. "
                f"Expected one of {sorted(VALID_NAMES_TRACE_MODES)}."
            )
    elif trace_category == "countries":
        if (
            object_trace_mode == "target_box"
            and names_trace_mode in VALID_OBJECT_TRACE_MODES
        ):
            resolved_object_trace_mode = names_trace_mode
            resolved_names_trace_mode = "context_only"
        elif object_trace_mode not in VALID_OBJECT_TRACE_MODES:
            raise ValueError(
                f"Unknown object_trace_mode {object_trace_mode!r}. "
                f"Expected one of {sorted(VALID_OBJECT_TRACE_MODES)}."
            )

    return resolved_names_trace_mode, resolved_object_trace_mode


def resolve_query_trace_mode(query_trace_mode: str = "same_query") -> str:
    normalized = QUERY_TRACE_MODE_ALIASES.get(
        query_trace_mode.lower(),
        query_trace_mode.lower(),
    )
    if normalized not in VALID_QUERY_TRACE_MODES:
        raise ValueError(
            f"Unknown query_trace_mode {query_trace_mode!r}. "
            f"Expected one of {sorted(VALID_QUERY_TRACE_MODES)}."
        )
    return normalized


def _validate_entity_index(*, field_name: str, value: int, num_entities: int) -> None:
    if not 0 <= value < num_entities:
        raise ValueError(
            f"Tracing expects {field_name} in [0, {num_entities}), got {value}."
        )


def resolve_swap_partner(*, query_name: int, num_entities: int) -> int:
    if num_entities < 2:
        raise ValueError("Tracing expects at least 2 entities.")
    _validate_entity_index(
        field_name="query_name",
        value=query_name,
        num_entities=num_entities,
    )
    return next(entity for entity in reversed(range(num_entities)) if entity != query_name)


def resolve_source_swap_pair(
    *,
    query_name: int,
    num_entities: int,
    swap_box_a: Optional[int] = None,
    swap_box_b: Optional[int] = None,
) -> Tuple[int, int]:
    if num_entities < 2:
        raise ValueError("Tracing expects at least 2 entities.")
    _validate_entity_index(
        field_name="query_name",
        value=query_name,
        num_entities=num_entities,
    )

    if (swap_box_a is None) != (swap_box_b is None):
        raise ValueError(
            "Provide both swap_box_a and swap_box_b together, or leave both unset."
        )

    if swap_box_a is None:
        return (query_name, resolve_swap_partner(query_name=query_name, num_entities=num_entities))

    _validate_entity_index(
        field_name="swap_box_a",
        value=swap_box_a,
        num_entities=num_entities,
    )
    _validate_entity_index(
        field_name="swap_box_b",
        value=swap_box_b,
        num_entities=num_entities,
    )
    if swap_box_a == swap_box_b:
        raise ValueError(
            f"swap_box_a and swap_box_b must be different, got {(swap_box_a, swap_box_b)}."
        )
    return (swap_box_a, swap_box_b)


def resolve_swap_partner_from_pair(
    *,
    query_name: int,
    source_swap_pair: Tuple[int, int],
) -> Optional[int]:
    left_box, right_box = source_swap_pair
    if left_box == query_name:
        return right_box
    if right_box == query_name:
        return left_box
    return None


def resolve_remaining_context_box(
    *,
    source_swap_pair: Tuple[int, int],
    num_entities: int,
) -> int:
    remaining_boxes = [
        entity for entity in range(num_entities) if entity not in source_swap_pair
    ]
    if not remaining_boxes:
        raise ValueError(
            "swap_target_box_to_remaining_context_box requires at least one "
            f"non-swapped context box. Got num_entities={num_entities} and "
            f"swap_pair={source_swap_pair}."
        )
    return remaining_boxes[0]


def resolve_prompt_query_names(
    *,
    vocab_tag: str,
    query_name: int,
    num_entities: int,
    query_trace_mode: str = "same_query",
    source_swap_pair: Optional[Tuple[int, int]] = None,
) -> Tuple[int, int, str]:
    resolved_query_trace_mode = resolve_query_trace_mode(query_trace_mode)
    source_query_name = query_name
    clean_query_name = query_name

    if resolved_query_trace_mode == "swap_partner":
        if vocab_tag != "BOXES":
            raise ValueError(
                "query_trace_mode='swap_partner' is only available for vocab_tag='BOXES'."
            )
        if source_swap_pair is None:
            clean_query_name = resolve_swap_partner(
                query_name=query_name,
                num_entities=num_entities,
            )
        else:
            clean_query_name = resolve_swap_partner_from_pair(
                query_name=query_name,
                source_swap_pair=source_swap_pair,
            )
            if clean_query_name is None:
                raise ValueError(
                    "query_trace_mode='swap_partner' requires query_name to be one "
                    f"of the swapped boxes. Got query_name={query_name} and "
                    f"swap_pair={source_swap_pair}."
                )

    return clean_query_name, source_query_name, resolved_query_trace_mode


def default_template_context(vocab_tag: str, query_name: int) -> Dict[str, object]:
    context = {"query_name": query_name, "raw_query_name": None}
    if vocab_tag == "ONEHOP":
        context["num_qn_context"] = 3
    return context


def _replace_swap_box_mention(statement, *, original_box: int, replacement_box: int):
    if statement.type != "swap":
        raise ValueError("Expected a swap statement when replacing a swap box mention.")

    updates = {}
    if statement.name == original_box:
        updates["name"] = replacement_box
    if statement.attr == original_box:
        updates["attr"] = replacement_box
    if not updates:
        raise ValueError(
            f"Swap statement {statement!r} does not mention box id {original_box}."
        )
    return statement._replace(**updates)


def build_trace_contexts(
    vocab,
    vocab_tag: str,
    trace_category: str,
    query_name: int,
    source_swap_pair: Optional[Tuple[int, int]] = None,
    names_trace_mode: str = "context_only",
    object_trace_mode: str = "target_box",
    num_entities: int = 3,
):
    _validate_entity_index(
        field_name="query_name",
        value=query_name,
        num_entities=num_entities,
    )
    if source_swap_pair is None:
        source_swap_pair = resolve_source_swap_pair(
            query_name=query_name,
            num_entities=num_entities,
        )
    swap_partner = resolve_swap_partner_from_pair(
        query_name=query_name,
        source_swap_pair=source_swap_pair,
    )
    names_trace_mode, object_trace_mode = resolve_trace_modes(
        trace_category,
        names_trace_mode=names_trace_mode,
        object_trace_mode=object_trace_mode,
    )

    source_entities = list(range(num_entities))
    source_attributes = list(range(num_entities))
    clean_entities = list(source_entities)
    clean_attributes = list(source_attributes)
    donor_entity = num_entities
    alternate_donor_entity = num_entities + 1
    # Unless the caller overrides the swap pair explicitly, BOXES tracing
    # pairs the queried box with the last non-query box, which in the
    # 3-entity case yields (0, 2), (1, 2), or (2, 1).
    donor_box = swap_partner
    remaining_context_box = resolve_remaining_context_box(
        source_swap_pair=source_swap_pair,
        num_entities=num_entities,
    )

    def require_swap_partner(reason: str) -> int:
        if donor_box is None:
            raise ValueError(
                f"{reason} requires query_name to be one of the swapped boxes. "
                f"Got query_name={query_name} and swap_pair={source_swap_pair}."
            )
        return donor_box

    def require_query_box_in_swap_pair(reason: str) -> None:
        if query_name not in source_swap_pair:
            raise ValueError(
                f"{reason} requires query_name to be one of the swapped boxes. "
                f"Got query_name={query_name} and swap_pair={source_swap_pair}."
            )

    if vocab_tag == "BOXES":
        if trace_category == "question_query_box":
            source_context = vocab.default_context(
                num_entities=num_entities,
                entities=source_entities,
                attributes=source_attributes,
                include_swap=True,
                swap_pair=source_swap_pair,
            )
            clean_context = list(source_context)
        elif trace_category == "names":
            source_context = vocab.default_context(
                num_entities=num_entities,
                entities=source_entities,
                attributes=source_attributes,
                include_swap=True,
                swap_pair=source_swap_pair,
            )
            if names_trace_mode == "context_only":
                # Start from the original prompt so the swap sentence stays
                # identical to the source prompt, then ablate only the queried
                # box mention inside the context.
                clean_context = list(source_context)
                clean_context[query_name] = clean_context[query_name]._replace(
                    name=donor_entity
                )
            elif names_trace_mode == "context_target_box":
                donor_box = require_swap_partner(
                    "names_trace_mode='context_target_box'"
                )
                # Start from the original prompt so the swap sentence and
                # question stay identical to the source prompt, then ablate
                # only the target/partner box mention inside the context.
                clean_context = list(source_context)
                clean_context[donor_box] = clean_context[donor_box]._replace(
                    name=donor_entity
                )
            elif names_trace_mode in {"swap_only", "swap_target_box"}:
                if names_trace_mode == "swap_only":
                    require_query_box_in_swap_pair("names_trace_mode='swap_only'")
                replaced_box = (
                    query_name
                    if names_trace_mode == "swap_only"
                    else require_swap_partner("names_trace_mode='swap_target_box'")
                )
                # Keep the context and question fixed, and corrupt only one
                # swap-argument box mention inside the swap sentence.
                clean_context = list(source_context)
                swap_statement_index = next(
                    idx
                    for idx, statement in enumerate(clean_context)
                    if statement.type == "swap"
                )
                clean_context[swap_statement_index] = _replace_swap_box_mention(
                    clean_context[swap_statement_index],
                    original_box=replaced_box,
                    replacement_box=donor_entity,
                )
            elif names_trace_mode == "swap_target_box_to_remaining_context_box":
                donor_box = require_swap_partner(
                    "names_trace_mode='swap_target_box_to_remaining_context_box'"
                )
                # Keep the context and question fixed, and rewrite only the
                # target/partner swap-box mention to the remaining in-context
                # box (for 3-entity prompts, this is the third box).
                clean_context = list(source_context)
                swap_statement_index = next(
                    idx
                    for idx, statement in enumerate(clean_context)
                    if statement.type == "swap"
                )
                clean_context[swap_statement_index] = _replace_swap_box_mention(
                    clean_context[swap_statement_index],
                    original_box=donor_box,
                    replacement_box=remaining_context_box,
                )
            elif (
                names_trace_mode
                == "context_target_object_and_swap_target_box_to_remaining_context_box"
            ):
                donor_box = require_swap_partner(
                    "names_trace_mode="
                    "'context_target_object_and_swap_target_box_to_remaining_context_box'"
                )
                # Corrupt the source swap target's context object token while
                # rewriting the target-side swap mention to the remaining
                # in-context box. This yields prompt pairs like:
                # source: "... Box P contains the cup.  ... Swap X and P."
                # clean:  "... Box P contains the apple. ... Swap X and U."
                clean_context = list(source_context)
                clean_context[donor_box] = clean_context[donor_box]._replace(
                    attr=donor_entity
                )
                swap_statement_index = next(
                    idx
                    for idx, statement in enumerate(clean_context)
                    if statement.type == "swap"
                )
                clean_context[swap_statement_index] = _replace_swap_box_mention(
                    clean_context[swap_statement_index],
                    original_box=donor_box,
                    replacement_box=remaining_context_box,
                )
            elif names_trace_mode == "context_target_box_and_swap_query_box":
                donor_box = require_swap_partner(
                    "names_trace_mode='context_target_box_and_swap_query_box'"
                )
                # Corrupt the target/partner box in the context and the
                # queried box mention inside the swap sentence, using
                # different donor box labels for the two sites.
                clean_context = list(source_context)
                clean_context[donor_box] = clean_context[donor_box]._replace(
                    name=alternate_donor_entity
                )
                swap_statement_index = next(
                    idx
                    for idx, statement in enumerate(clean_context)
                    if statement.type == "swap"
                )
                clean_context[swap_statement_index] = _replace_swap_box_mention(
                    clean_context[swap_statement_index],
                    original_box=query_name,
                    replacement_box=donor_entity,
                )
            elif names_trace_mode == "context_and_swap":
                require_swap_partner("names_trace_mode='context_and_swap'")
                # Corrupt the queried box in both the leading context and
                # the queried-side mention inside the swap sentence.
                clean_entities[query_name] = donor_entity
                clean_entity_by_source = dict(zip(source_entities, clean_entities))
                clean_swap_pair = tuple(
                    clean_entity_by_source[box_name] for box_name in source_swap_pair
                )
                clean_context = vocab.default_context(
                    num_entities=num_entities,
                    entities=clean_entities,
                    attributes=source_attributes,
                    include_swap=True,
                    swap_pair=clean_swap_pair,
                )
            elif names_trace_mode == "context_and_swap_target_box":
                donor_box = require_swap_partner(
                    "names_trace_mode='context_and_swap_target_box'"
                )
                # Corrupt the target/partner box in both the leading context
                # and the target-side mention inside the swap sentence.
                clean_context = list(source_context)
                clean_context[donor_box] = clean_context[donor_box]._replace(
                    name=donor_entity
                )
                swap_statement_index = next(
                    idx
                    for idx, statement in enumerate(clean_context)
                    if statement.type == "swap"
                )
                clean_context[swap_statement_index] = _replace_swap_box_mention(
                    clean_context[swap_statement_index],
                    original_box=donor_box,
                    replacement_box=donor_entity,
                )
            else:
                raise ValueError(
                    f"Unsupported BOXES names_trace_mode {names_trace_mode!r}."
                )
        elif trace_category == "countries":
            # BOXES object tracing can corrupt the post-swap donor object's
            # token (default), the queried box's own context object, or the
            # donor object plus the queried box mention inside the swap.
            if object_trace_mode == "target_box":
                donor_box = require_swap_partner(
                    "object_trace_mode='target_box'"
                )
                clean_attributes[donor_box] = donor_entity
                clean_context = vocab.default_context(
                    num_entities=num_entities,
                    entities=source_entities,
                    attributes=clean_attributes,
                    include_swap=True,
                    swap_pair=source_swap_pair,
                )
            elif object_trace_mode == "query_box":
                clean_attributes[query_name] = donor_entity
                clean_context = vocab.default_context(
                    num_entities=num_entities,
                    entities=source_entities,
                    attributes=clean_attributes,
                    include_swap=True,
                    swap_pair=source_swap_pair,
                )
            elif object_trace_mode == "context_target_object_and_swap_query_box":
                donor_box = require_swap_partner(
                    "object_trace_mode='context_target_object_and_swap_query_box'"
                )
                clean_attributes[donor_box] = donor_entity
                clean_context = vocab.default_context(
                    num_entities=num_entities,
                    entities=source_entities,
                    attributes=clean_attributes,
                    include_swap=True,
                    swap_pair=source_swap_pair,
                )
                swap_statement_index = next(
                    idx
                    for idx, statement in enumerate(clean_context)
                    if statement.type == "swap"
                )
                clean_context[swap_statement_index] = _replace_swap_box_mention(
                    clean_context[swap_statement_index],
                    original_box=query_name,
                    replacement_box=donor_entity,
                )
            elif object_trace_mode == "swap_target_object_with_remaining_context_object":
                donor_box = require_swap_partner(
                    "object_trace_mode='swap_target_object_with_remaining_context_object'"
                )
                # Swap the target/partner box's context object with the
                # remaining in-context object's token while keeping the swap
                # sentence fixed. For 3-entity prompts, this changes e.g.
                # source: "... Box X contains the milk. Box U contains the drug ..."
                # clean:  "... Box X contains the drug. Box U contains the milk ..."
                clean_attributes[donor_box] = source_attributes[remaining_context_box]
                clean_attributes[remaining_context_box] = source_attributes[donor_box]
                clean_context = vocab.default_context(
                    num_entities=num_entities,
                    entities=source_entities,
                    attributes=clean_attributes,
                    include_swap=True,
                    swap_pair=source_swap_pair,
                )
            else:
                raise ValueError(
                    f"Unsupported BOXES object_trace_mode {object_trace_mode!r}."
                )
            source_context = vocab.default_context(
                num_entities=num_entities,
                entities=source_entities,
                attributes=source_attributes,
                include_swap=True,
                swap_pair=source_swap_pair,
            )
        elif trace_category == "swap_boxes":
            clean_context = vocab.default_context(
                num_entities=num_entities,
                entities=source_entities,
                attributes=source_attributes,
                include_swap=False,
            )
            source_context = vocab.default_context(
                num_entities=num_entities,
                entities=source_entities,
                attributes=source_attributes,
                include_swap=True,
                swap_pair=source_swap_pair,
            )
        else:
            raise ValueError(f"Unsupported BOXES trace category: {trace_category}")
        return clean_context, source_context

    if trace_category == "question_query_box":
        clean_context = vocab.default_context(
            num_entities=num_entities,
            entities=source_entities,
            attributes=source_attributes,
        )
        source_context = vocab.default_context(
            num_entities=num_entities,
            entities=source_entities,
            attributes=source_attributes,
        )
    elif trace_category == "names":
        clean_entities[query_name] = donor_entity
        clean_context = vocab.default_context(
            num_entities=num_entities,
            entities=clean_entities,
            attributes=source_attributes,
        )
        source_context = vocab.default_context(
            num_entities=num_entities,
            entities=source_entities,
            attributes=source_attributes,
        )
    elif trace_category == "countries":
        clean_attributes[query_name] = donor_entity
        clean_context = vocab.default_context(
            num_entities=num_entities,
            entities=source_entities,
            attributes=clean_attributes,
        )
        source_context = vocab.default_context(
            num_entities=num_entities,
            entities=source_entities,
            attributes=source_attributes,
        )
    else:
        raise ValueError(f"Unsupported {vocab_tag} trace category: {trace_category}")
    return clean_context, source_context


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


def _candidate_prediction_summary(answer_logits: torch.Tensor, answer_tokens: torch.Tensor, target_idx: torch.Tensor):
    pred_idx = answer_logits.argmax(dim=-1)
    batch_indices = torch.arange(answer_logits.shape[0], device=answer_logits.device)
    pred_tokens = answer_tokens[batch_indices, pred_idx]
    target_tokens = answer_tokens[batch_indices, target_idx]
    target_log_probs = answer_logits[batch_indices, target_idx]
    return pred_idx, pred_tokens, target_tokens, target_log_probs


def find_correct_samples(
    model,
    dataset: TracePromptDataset,
    batch_size: int,
    num_samples: int,
    verbose: bool = False,
) -> List[int]:
    device = _model_device(model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tokenizer = model.tokenizer

    correct_indices: List[int] = []
    seen = 0
    for batch in tqdm(dataloader, desc="Finding correct samples"):
        batch = _move_batch_to_device(batch, device)
        with torch.no_grad():
            clean_logits = model(batch["clean_tokens"], return_type="logits")
            source_logits = model(batch["source_tokens"], return_type="logits")

        source_answer_logits = lookup_answer_logits(
            source_logits,
            batch["source_answer_tokens"],
            query_position=batch["source_query_positions"],
        )
        clean_answer_logits = lookup_answer_logits(
            clean_logits,
            batch["source_answer_tokens"],
            query_position=batch["clean_query_positions"],
        )

        query_names = batch.get("source_query_names", batch["query_names"])
        clean_pred_idx, clean_pred_tokens, clean_target_tokens, _ = _candidate_prediction_summary(
            clean_answer_logits,
            batch["source_answer_tokens"],
            query_names,
        )
        source_pred_idx, source_pred_tokens, source_target_tokens, _ = _candidate_prediction_summary(
            source_answer_logits,
            batch["source_answer_tokens"],
            query_names,
        )

        source_ok = source_pred_idx.eq(query_names)

        if verbose:
            for i in range(int(query_names.shape[0])):
                clean_pred = tokenizer.decode([int(clean_pred_tokens[i].item())]).strip()
                clean_target = tokenizer.decode([int(clean_target_tokens[i].item())]).strip()
                source_pred = tokenizer.decode([int(source_pred_tokens[i].item())]).strip()
                source_target = tokenizer.decode([int(source_target_tokens[i].item())]).strip()
                print(
                    f"clean pred={clean_pred!r} target={clean_target!r} | "
                    f"source pred={source_pred!r} target={source_target!r}"
                )

        batch_size_actual = int(query_names.shape[0])
        for i in range(batch_size_actual):
            if bool(source_ok[i].item()):
                correct_indices.append(seen + i)
                if len(correct_indices) >= num_samples:
                    return correct_indices[:num_samples]
        seen += batch_size_actual

    return correct_indices[:num_samples]


def _patch_resid_pre_at_token(
    resid_pre: torch.Tensor,
    hook,
    *,
    source_cache,
    token_index: int,
    valid_mask: torch.Tensor,
):
    valid_mask = valid_mask.to(resid_pre.device)
    if valid_mask.any():
        resid_pre = resid_pre.clone()
        resid_pre[valid_mask, token_index, :] = source_cache[hook.name][valid_mask, token_index, :].to(
            resid_pre.device
        )
    return resid_pre


def _init_nested_scores(token_axis: Iterable[int], layer_axis: Iterable[int]):
    accuracy_sum = {int(token): {int(layer): 0.0 for layer in layer_axis} for token in token_axis}
    prob_sum = {int(token): {int(layer): 0.0 for layer in layer_axis} for token in token_axis}
    count_sum = {int(token): {int(layer): 0 for layer in layer_axis} for token in token_axis}
    return accuracy_sum, prob_sum, count_sum


def _build_result_payload(
    *,
    metadata: Dict[str, object],
    token_axis: List[int],
    layer_axis: List[int],
    accuracy_sum,
    prob_sum,
    count_sum,
    reference_batch: Optional[Dict[str, object]],
):
    scores = {}
    for token in token_axis:
        token_key = str(int(token))
        scores[token_key] = {}
        for layer in layer_axis:
            count = int(count_sum[int(token)][int(layer)])
            accuracy = (
                float(accuracy_sum[int(token)][int(layer)] / count) if count > 0 else None
            )
            mean_target_prob = (
                float(prob_sum[int(token)][int(layer)] / count) if count > 0 else None
            )
            scores[token_key][str(int(layer))] = {
                "accuracy": accuracy,
                "mean_target_prob": mean_target_prob,
                "count": count,
            }

    token_reference = {}
    if reference_batch is not None:
        clean_tokens = reference_batch["clean_tokens"]
        source_tokens = reference_batch["source_tokens"]
        clean_query_positions = reference_batch["clean_query_positions"]
        source_query_positions = reference_batch["source_query_positions"]
        tokenizer = reference_batch["tokenizer"]
        for token in token_axis:
            token_idx = int(token)
            valid = (
                (clean_query_positions >= token_idx)
                & (source_query_positions >= token_idx)
            ).nonzero(as_tuple=False)
            if valid.numel() == 0:
                continue
            sample_idx = int(valid[0].item())
            token_reference[str(token_idx)] = {
                "clean_token": tokenizer.decode(
                    [int(clean_tokens[sample_idx, token_idx].item())],
                    clean_up_tokenization_spaces=False,
                ),
                "source_token": tokenizer.decode(
                    [int(source_tokens[sample_idx, token_idx].item())],
                    clean_up_tokenization_spaces=False,
                ),
            }

    payload = {
        "metadata": metadata,
        "token_axis": token_axis,
        "layer_axis": layer_axis,
        "scores": scores,
        "token_reference": token_reference,
    }
    if reference_batch is not None:
        payload["examples"] = {
            "clean_prompt": reference_batch["clean_prompts"][0],
            "source_prompt": reference_batch["source_prompts"][0],
        }
    return payload


def save_tracing_results(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_tracing_experiment(
    *,
    model,
    dataset: TracePromptDataset,
    tracing_batch_size: int,
    results_path: Path,
    metadata: Dict[str, object],
    start_token: Optional[int] = None,
    end_token: int = 0,
    start_layer: int = 0,
    layer_step: int = 1,
    verbose: bool = False,
):
    device = _model_device(model)
    dataloader = DataLoader(dataset, batch_size=tracing_batch_size, shuffle=False)
    layer_axis = list(range(start_layer, model.cfg.n_layers, layer_step))

    valid_token_limits = torch.minimum(
        dataset.payload["clean_query_positions"],
        dataset.payload["source_query_positions"],
    )
    if start_token is None:
        start_token = int(valid_token_limits.max().item())
    token_axis = list(range(int(start_token), int(end_token) - 1, -1))

    accuracy_sum, prob_sum, count_sum = _init_nested_scores(token_axis, layer_axis)
    reference_batch = None

    for batch in tqdm(dataloader, desc="Tracing batches"):
        batch = _move_batch_to_device(batch, device)
        if reference_batch is None:
            reference_batch = {
                "clean_tokens": batch["clean_tokens"].detach().cpu(),
                "source_tokens": batch["source_tokens"].detach().cpu(),
                "clean_query_positions": batch["clean_query_positions"].detach().cpu(),
                "source_query_positions": batch["source_query_positions"].detach().cpu(),
                "clean_prompts": list(batch["clean_prompts"]),
                "source_prompts": list(batch["source_prompts"]),
                "tokenizer": model.tokenizer,
            }

        with torch.no_grad():
            _, source_cache = model.run_with_cache(
                batch["source_tokens"],
                names_filter=lambda name: "hook_resid_pre" in name,
                return_type="logits",
            )

            for token_idx in token_axis:
                valid_mask = (
                    (batch["clean_query_positions"] >= token_idx)
                    & (batch["source_query_positions"] >= token_idx)
                )
                valid_count = int(valid_mask.sum().item())
                if valid_count == 0:
                    continue

                for layer_idx in layer_axis:
                    hook_name = _get_act_name("resid_pre", layer_idx)
                    patched_logits = model.run_with_hooks(
                        batch["clean_tokens"],
                        fwd_hooks=[
                            (
                                hook_name,
                                lambda resid_pre, hook, source_cache=source_cache, token_idx=token_idx, valid_mask=valid_mask: _patch_resid_pre_at_token(
                                    resid_pre,
                                    hook,
                                    source_cache=source_cache,
                                    token_index=token_idx,
                                    valid_mask=valid_mask,
                                ),
                            )
                        ],
                        return_type="logits",
                    )

                    answer_logits = lookup_answer_logits(
                        patched_logits,
                        batch["source_answer_tokens"],
                        query_position=batch["clean_query_positions"],
                    )
                    query_names = batch.get("source_query_names", batch["query_names"])
                    pred_idx, _, _, target_log_probs = _candidate_prediction_summary(
                        answer_logits,
                        batch["source_answer_tokens"],
                        query_names,
                    )
                    correct_mask = pred_idx.eq(query_names) & valid_mask
                    accuracy_sum[token_idx][layer_idx] += float(correct_mask.sum().item())
                    prob_sum[token_idx][layer_idx] += float(
                        target_log_probs.exp()[valid_mask].sum().item()
                    )
                    count_sum[token_idx][layer_idx] += valid_count

                    if verbose:
                        acc = accuracy_sum[token_idx][layer_idx] / count_sum[token_idx][layer_idx]
                        print(
                            f"token={token_idx:>3} layer={layer_idx:>2} "
                            f"batch_valid={valid_count:>2} running_acc={acc:.4f}"
                        )

                payload = _build_result_payload(
                    metadata=metadata,
                    token_axis=token_axis,
                    layer_axis=layer_axis,
                    accuracy_sum=accuracy_sum,
                    prob_sum=prob_sum,
                    count_sum=count_sum,
                    reference_batch=reference_batch,
                )
                save_tracing_results(results_path, payload)

    final_payload = _build_result_payload(
        metadata=metadata,
        token_axis=token_axis,
        layer_axis=layer_axis,
        accuracy_sum=accuracy_sum,
        prob_sum=prob_sum,
        count_sum=count_sum,
        reference_batch=reference_batch,
    )
    save_tracing_results(results_path, final_payload)
    return final_payload
