from __future__ import annotations

import itertools
import inspect
from importlib import import_module
from .common import (
    tokenize_prompt,
    Substring,
    recursive_add_offset,
    TrackFormatter,
)


VOCAB_MODULES = {
    "BOXES": "datasets.box",
}
TEMPLATE_MODULES = {
    "BOXES": "datasets.box",
}


def prepend(first_fn):
    def decorator(second_fn):
        def wrapped(*args, **kwargs):
            first_result = first_fn(*args, **kwargs)
            return second_fn(*args, **{**kwargs, **first_result})

        return wrapped

    return decorator


def get_vocab_class(vocab_tag):
    return import_module(VOCAB_MODULES[vocab_tag]).Vocab


def get_templates(vocab_tag):
    return import_module(TEMPLATE_MODULES[vocab_tag]).TEMPLATES


def load_vocab(vocab_tag, model_tag, tokenizer=None, **vocab_kwargs):
    vocab_tag = vocab_tag.upper()
    model_tag_lower = model_tag.lower()
    if "llama" in model_tag_lower or "tulu" in model_tag_lower:
        tokenizer_type = "llama"
    elif "gemma" in model_tag_lower:
        tokenizer_type = "gemma"
    else:
        tokenizer_type = "pythia"
    if vocab_tag in VOCAB_MODULES:
        vocab_class = get_vocab_class(vocab_tag)
        if tokenizer is not None:
            if "tokenizer" not in inspect.signature(vocab_class).parameters:
                raise TypeError(
                    f"{vocab_class.__name__} does not accept an explicit tokenizer."
                )
            vocab_kwargs = {**vocab_kwargs, "tokenizer": tokenizer}
        return vocab_class(tokenizer_type, **vocab_kwargs)
    raise ValueError(
        f"Unknown vocab tag {vocab_tag!r}. Available: {sorted(VOCAB_MODULES)}."
    )


def generate_prompts(
    *,
    vocab,
    context,
    query_name=None,
    prompt_id=None,
    template_type="normal",
    template_context=None,
    raw_query_name=None,
    **kwargs,
):
    del kwargs
    """
    Using setting raw_query_name overrides query_name
    Args:
        tokenizer
        context: List[Statement]
        query_name: Name
        raw_query_name: None | str
    """
    if prompt_id is None:
        prompt_id = vocab.global_rng.integers(int(1e18))
    if template_context is None:
        template_context = dict(query_name=query_name, raw_query_name=raw_query_name)

    if vocab.type in VOCAB_MODULES:
        templates = get_templates(vocab.type)
    else:
        raise ValueError(f"Unknown vocab type {vocab.type}")
    template = templates.lookup(template_type)
    template_string, template_substitutions = template.generate_template(
        template_context=template_context,
        prompt_id=prompt_id,
        vocab=vocab,
        context=context,
    )
    context_text = []
    context_indices = []
    for statement in context:
        text, indices = template.instantiate(
            vocab=vocab,
            statement=statement,
            prompt_id=prompt_id,
            template_context=template_context,
        )
        context_text.append(text)
        context_indices.append(indices)
    joined_context_text = "".join(context_text)
    full_output, full_output_indices = TrackFormatter().format(
        template_string, context=joined_context_text, **template_substitutions
    )
    context_start_pos = full_output_indices["context"][0].start
    context_acc = itertools.accumulate(
        [context_start_pos] + [len(x) for x in context_text]
    )
    full_output_indices = {
        "prompt": Substring(0, len(full_output)),
        "context_section": full_output_indices["context"][0],
        "context": [
            recursive_add_offset(ctx_idx_map, offset)
            for offset, ctx_idx_map in zip(context_acc, context_indices)
        ],
        **template.extract_template_indices(full_output_indices),
    }
    return dict(prompt=full_output, indices=full_output_indices)


def generate_options(labels, options):
    assert len(labels) == len(options)
    option_strings = [f"\n{label}: {option}" for label, option in zip(labels, options)]
    return "".join(option_strings)


generate_tokenized_prompts = prepend(generate_prompts)(tokenize_prompt)
