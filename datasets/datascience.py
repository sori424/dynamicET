import torch
import einops
from einops import rearrange, einsum

import transformer_lens.utils as utils

from .common import recursify, rotate


def run_cached_model(*args, **kwargs):
    from coref.representations import run_cached_model as impl

    return impl(*args, **kwargs)


def build_example_refactored(*args, **kwargs):
    from coref.representations import build_example_refactored as impl

    return impl(*args, **kwargs)


def build_selector(*args, **kwargs):
    from coref.representations import build_selector as impl

    return impl(*args, **kwargs)


def _residual_injection_parameters():
    from coref.representations import ResidualInjectionParameters

    return ResidualInjectionParameters


def _residual_injection_subspace():
    from coref.representations import ResidualInjectionSubspace

    return ResidualInjectionSubspace


def cosine_sim(mean_1, mean_2):
    """
    mean_1 : [layer, dim]
    mean_2 : [layer, dim]
    returns [layer]
    """
    return (
        einsum(mean_1, mean_2, "l d, l d-> l")
        / (
            einsum(mean_1, mean_1, "l d, l d->l")
            * einsum(mean_2, mean_2, "l d, l d->l")
        )
        ** 0.5
    )


def parallel_component(base_subspace, test_subspace):
    """
    base_subspace: [subspace, layer, dim]
    test_subspace: [subspace, layer, dim]
    """

    parallel_component = einsum(
        test_subspace,
        base_subspace,
        einsum(
            base_subspace,
            base_subspace,
            "subspace layer i, subspace layer i -> subspace layer",
        )
        ** -1,
        "subspace layer i, subspace layer i, subspace layer -> subspace layer",
    )

    return parallel_component


def collect_data(
    *,
    model,
    selector=None,
    num_entities=2,
    num_samples=500,
    batch_size=50,
    vocab,
    query_name=0,
    context=None,
    template_context=None,
    execution_fn=None,
    selector_extractor=None,
    index_start=0,
    hook_name="resid_pre",
    template_type="normal"
):
    if template_context is not None:
        assert query_name == 0
    if context is None:
        context = vocab.default_context(num_entities)
    if execution_fn is None:
        model_runner = lambda input_tokens: model.run_with_cache(
            input_tokens, names_filter=lambda x: hook_name in x
        )
    else:
        model_runner = lambda input_tokens: run_cached_model(
            execution_fn=lambda: execution_fn(input_tokens),
            model=model,
            names_filter=lambda x: hook_name in x,
        )
    all_acts = []
    all_tokens = []
    all_answers = []
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        input_tokens, answer_tokens, _, token_maps = build_example_refactored(
            batch_size=end - start,
            vocab=vocab,
            num_entities=num_entities,
            fixed_query_name=query_name,
            context=context,
            prompt_id_start=start + index_start,
            template_type=template_type,
            template_context=template_context,
        )
        logits, cache = model_runner(input_tokens)

        if selector is not None:
            assert selector_extractor is None
            actual_selector_extractor = lambda token_maps: recursify(
                lambda pos: torch.tensor([pos] * (end - start)), dtype=int
            )(selector)
        else:
            actual_selector_extractor = selector_extractor
        # actual selector is selector stacked into batches
        actual_selector = actual_selector_extractor(token_maps)
        if hook_name == "resid_pre":
            acts = recursify(
                lambda positions: torch.stack(
                    [
                        (
                            lambda activations: activations.cpu().gather(
                                dim=1,
                                index=einops.repeat(
                                    positions,
                                    "batch -> batch dummy dim",
                                    dummy=1,
                                    dim=activations.shape[2],
                                ),
                            )[:, 0, :]
                        )(cache[utils.get_act_name(hook_name, layer)])
                        for layer in range(model.cfg.n_layers)
                    ]
                ),
                dtype=torch.Tensor,
            )(actual_selector)
        elif hook_name == "pattern":  # [batch, head_index, query_pos, key_pos]
            assert selector is not None, "Selector extractor not implemented"
            acts = recursify(
                lambda x: torch.stack(
                    [
                        cache[utils.get_act_name(hook_name, layer)][:, :, x, :].cpu()
                        for layer in range(model.cfg.n_layers)
                    ]
                ),
                dtype=int,
            )(selector)
        elif hook_name == "z":  # [batch, pos, head_index, d_head]
            assert selector is not None, "Selector extractor not implemented"
            acts = recursify(
                lambda x: torch.stack(
                    [
                        cache[utils.get_act_name(hook_name, layer)][:, x, :, :].cpu()
                        for layer in range(model.cfg.n_layers)
                    ]
                ),
                dtype=int,
            )(selector)

        tokens = recursify(
            lambda positions: input_tokens.cpu().gather(
                dim=1, index=positions[:, None]
            )[:, 0],
            dtype=torch.Tensor,
        )(actual_selector)
        all_acts.append(acts)
        all_tokens.append(tokens)
        all_answers.append(answer_tokens)
    if hook_name == "resid_pre":
        all_acts = rotate(
            lambda x: rearrange(
                torch.stack(x), "batch layer minibatch dim->(batch minibatch) layer dim"
            ),
            dtype=torch.Tensor,
        )(all_acts)
    elif hook_name == "pattern":
        all_acts = rotate(
            lambda x: rearrange(
                torch.stack(x),
                "batch layer minibatch headidx dim->(batch minibatch) layer headidx dim",
            ),
            dtype=torch.Tensor,
        )(all_acts)
    elif hook_name == "z":
        all_acts = rotate(
            lambda x: rearrange(
                torch.stack(x),
                "batch layer minibatch headidx dim->(batch minibatch) layer headidx dim",
            ),
            dtype=torch.Tensor,
        )(all_acts)

    all_tokens = rotate(lambda x: torch.concatenate(x), dtype=torch.Tensor)(all_tokens)
    all_answers = rotate(lambda x: torch.concatenate(x), dtype=torch.Tensor)(
        all_answers
    )
    return all_acts, all_tokens, all_answers


def vector_to_residual_injection(model, vector, layer_start=1, layer_end=32):
    assert len(vector.shape) == 2
    ret = _residual_injection_parameters()(model, layer_start, layer_end)
    for layer in range(layer_start, layer_end):
        if layer == layer_start:
            ret.parameters[layer][0, :] = vector[layer]
        else:
            ret.parameters[layer][0, :] = vector[layer] - vector[layer - 1]
    return ret


def vector_to_residual_subspace(model, vector, layer_start=1, layer_end=32):
    assert len(vector.shape) == 2
    ret = _residual_injection_subspace()(model, layer_start, layer_end)
    for layer in range(layer_start, layer_end):
        if layer == layer_start:
            ret.parameters[layer][0, 0, :] = vector[layer]
        else:
            ret.parameters[layer][0, 0, :] = vector[layer] - vector[layer - 1]
    return ret


def extract_means(
    *,
    model,
    vocab,
    entity_width=None,
    country_width=None,
    widths=None,
    base_selector_extractor=None,
    num_samples=500
):
    """
    base_selector_extractor: token_maps -> Recursive[pos: Tensor[batch]]
    """
    if base_selector_extractor is None:
        three_selector = build_selector(vocab, 3)
        long_selector = {
            "countries": [
                [p + i for i in range(country_width)]
                for p in three_selector["countries"]
            ],
            "names": [
                [p + i for i in range(entity_width)] for p in three_selector["names"]
            ],
        }
        selector_options = {"selector": long_selector}
    else:

        def data_selector_extractor(token_maps):
            sel = base_selector_extractor(token_maps)
            return {
                key: [[positions + i for i in range(width)] for positions in sel[key]]
                for key, width in widths.items()
            }

        selector_options = {"selector_extractor": data_selector_extractor}

    default_acts, _, _ = collect_data(
        model=model,
        num_entities=3,
        num_samples=num_samples,
        batch_size=50,
        vocab=vocab,
        query_name=0,
        context=None,
        execution_fn=None,
        index_start=0,
        hook_name="resid_pre",
        **selector_options
    )
    contrast_acts, _, _ = collect_data(
        model=model,
        num_entities=3,
        num_samples=num_samples,
        batch_size=50,
        vocab=vocab,
        query_name=0,
        context=vocab.default_context(3, [1, 2, 0], [1, 2, 0]),
        execution_fn=None,
        index_start=0,
        hook_name="resid_pre",
        **selector_options
    )
    if widths is None:
        country_means = torch.stack(
            [
                (
                    default_acts["countries"][1][pos]
                    - contrast_acts["countries"][0][pos]
                ).mean(dim=0)
                for pos in range(country_width)
            ]
        )

        three_country_means = torch.stack(
            [
                (
                    contrast_acts["countries"][2][pos]
                    - default_acts["countries"][0][pos]
                ).mean(dim=0)
                for pos in range(country_width)
            ]
        )
        entity_means = torch.stack(
            [
                (default_acts["names"][1][pos] - contrast_acts["names"][0][pos]).mean(
                    dim=0
                )
                for pos in range(entity_width)
            ]
        )
        three_entity_means = torch.stack(
            [
                (contrast_acts["names"][2][pos] - default_acts["names"][0][pos]).mean(
                    dim=0
                )
                for pos in range(entity_width)
            ]
        )
        return (country_means, three_country_means, entity_means, three_entity_means)
    else:
        return {
            category: [
                torch.stack(
                    [
                        (
                            default_acts[category][1][pos]
                            - contrast_acts[category][0][pos]
                        ).mean(dim=0)
                        for pos in range(width)
                    ]
                ),
                torch.stack(
                    [
                        (
                            contrast_acts["names"][2][pos]
                            - default_acts["names"][0][pos]
                        ).mean(dim=0)
                        for pos in range(width)
                    ]
                ),
            ]
            for category, width in widths.items()
        }


def extract_k_means(
    *,
    model,
    vocab,
    widths,
    base_selector_extractor,
    num_samples=500,
    batch_size=50,
    num_entities,
    prompt_id_start=0,
    spread=False,
    additional_template_context={}
):
    """
    base_selector_extractor: token_maps -> Recursive[pos: Tensor[batch]]
    """
    if spread:
        true_widths = {k: w for k, w in widths.items()}
        widths = {k: 1 for k, _ in widths.items()}

    def data_selector_extractor(token_maps):
        sel = base_selector_extractor(token_maps)
        return {
            key: [[positions + i for i in range(width)] for positions in sel[key]]
            for key, width in widths.items()
        }

    selector_options = {"selector_extractor": data_selector_extractor}


    template_context = {**additional_template_context, 'query_name': 0, 'raw_query_name': None}
    default_acts, _, _ = collect_data(
        model=model,
        num_entities=num_entities,
        num_samples=num_samples,
        batch_size=batch_size,
        vocab=vocab,
        template_context=template_context,
        context=None,
        execution_fn=None,
        index_start=prompt_id_start,
        hook_name="resid_pre",
        **selector_options
    )
    shifted = list(range(num_entities))
    shifted = shifted[1:] + shifted[:1]
    contrast_acts, _, _ = collect_data(
        model=model,
        num_entities=num_entities,
        num_samples=num_samples,
        batch_size=batch_size,
        vocab=vocab,
        template_context=template_context,
        context=vocab.default_context(num_entities, shifted, shifted),
        execution_fn=None,
        index_start=prompt_id_start,
        hook_name="resid_pre",
        **selector_options
    )
    if num_entities == 3:
        ret = {
            category: [
                torch.stack(
                    [
                        (
                            default_acts[category][1][pos]
                            - contrast_acts[category][0][pos]
                        ).mean(dim=0)
                        for pos in range(width)
                    ]
                ),
                torch.stack(
                    [
                        (
                            contrast_acts[category][2][pos]
                            - default_acts[category][0][pos]
                        ).mean(dim=0)
                        for pos in range(width)
                    ]
                ),
            ]
            for category, width in widths.items()
        }
    else:
        ret = {
            category: [
                torch.stack(
                    [
                        (
                            default_acts[category][side][pos]
                            - contrast_acts[category][0][pos]
                        ).mean(dim=0)
                        for pos in range(width)
                    ]
                )
                for side in range(1, num_entities)
            ]
            for category, width in widths.items()
        }
    if spread:
        ret = {
            category: [
                einops.repeat(means, "1 layer dim -> width layer dim", width=width)
                for means in ret[category]
            ]
            for category, width in true_widths.items()
        }
    return ret
