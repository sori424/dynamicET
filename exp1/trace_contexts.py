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
