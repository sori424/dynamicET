from __future__ import annotations

from exp2.eval_core import (
    CircuitComponent,
    DictTensorDataset,
    STAGE_FILES,
    all_filtered_components,
    hook_name,
    tensor_topk_components,
    torch_load_cpu,
)

from exp2.eval_metrics import (
    move_batch_to_device,
)

from collections import defaultdict
from dataclasses import asdict, dataclass

def selected_stage_heads(
    tensor,
    *,
    k: int,
    circuit_source: str,
    largest: bool,
    include_zero_contribution_heads: bool,
    zero_tolerance: float,
) -> Tuple[List[List[int]], List[float]]:
    if circuit_source == "all-negative":
        return all_filtered_components(
            tensor,
            largest=False,
            zero_tolerance=zero_tolerance,
        )
    if circuit_source == "all-positive":
        return all_filtered_components(
            tensor,
            largest=True,
            zero_tolerance=zero_tolerance,
        )
    heads, values = tensor_topk_components(tensor, k, largest=largest)
    if include_zero_contribution_heads:
        return heads, values
    filtered = [
        (head, value)
        for head, value in zip(heads, values)
        if abs(value) > zero_tolerance
    ]
    return [head for head, _ in filtered], [value for _, value in filtered]


def add_components(
    *,
    components: List[CircuitComponent],
    stage: str,
    heads: Sequence[Sequence[int]],
    values: Sequence[float],
    position_mode: str,
    hook_component: str,
    utils,
) -> None:
    for (layer, head), score in zip(heads, values):
        components.append(
            CircuitComponent(
                stage=stage,
                position_mode=position_mode,
                hook_name=hook_name(utils, hook_component, int(layer)),
                layer=int(layer),
                head=int(head),
                score=float(score),
            )
        )


def append_attention_pattern_context_reader_components(
    *,
    components: List[CircuitComponent],
    selected_heads: Sequence[Sequence[int]],
    selected_scores: Sequence[float],
    position_mode: str,
    hook_component: str,
    utils,
) -> List[CircuitComponent]:
    updated = list(components)
    add_components(
        components=updated,
        stage="struct_reader_context_target_box_to_target_box",
        heads=selected_heads,
        values=selected_scores,
        position_mode=position_mode,
        hook_component=hook_component,
        utils=utils,
    )
    return dedupe_components(updated)


def dedupe_components(components: Sequence[CircuitComponent]) -> List[CircuitComponent]:
    deduped: Dict[Tuple[str, str, int], CircuitComponent] = {}
    for component in components:
        key = (component.position_mode, component.hook_name, component.head)
        previous = deduped.get(key)
        if previous is None or abs(component.score) > abs(previous.score):
            deduped[key] = component
    return list(deduped.values())


def circuit_by_position_from_components(
    components: Sequence[CircuitComponent],
) -> Dict[str, Dict[str, List[int]]]:
    circuit: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for component in components:
        heads = circuit[component.position_mode][component.hook_name]
        if component.head not in heads:
            heads.append(component.head)
    return {
        mode: {hook: sorted(heads) for hook, heads in by_hook.items()}
        for mode, by_hook in circuit.items()
    }


def component_count_by_position(components: Sequence[CircuitComponent]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for component in components:
        counts[component.position_mode] += 1
    return dict(counts)


def slice_dict_tensor_dataset(dataset: DictTensorDataset, limit: int) -> DictTensorDataset:
    limit = max(0, min(int(limit), len(dataset)))
    if limit == 0:
        raise ValueError("Dataset slice limit must keep at least one example.")
    return DictTensorDataset(
        {
            key: value[:limit] if hasattr(value, "__getitem__") else value
            for key, value in dataset.data.items()
        }
    )


def ranges_overlap(start_a: int, len_a: int, start_b: int, len_b: int) -> bool:
    end_a = start_a + len_a
    end_b = start_b + len_b
    return start_a < end_b and start_b < end_a


def compute_context_reader_attention_scores(
    *,
    model,
    dataloader,
    device: Any,
    utils,
    torch_module,
):
    pattern_hook_names = [
        utils.get_act_name("pattern", layer)
        for layer in range(int(model.cfg.n_layers))
    ]
    pattern_hook_set = set(pattern_hook_names)
    score_sums = torch_module.zeros(
        (int(model.cfg.n_layers), int(model.cfg.n_heads)),
        dtype=torch_module.float64,
    )
    total = 0

    from tqdm import tqdm

    with torch_module.no_grad():
        for batch in tqdm(dataloader, desc="Attention pattern scores"):
            batch = move_batch_to_device(batch, device)
            _, cache = model.run_with_cache(
                batch["base_tokens"],
                names_filter=lambda name: name in pattern_hook_set,
                return_type="logits",
            )
            query_positions = batch["base_swap_target_box_positions"].to(
                device=device, dtype=torch_module.long
            )
            key_positions = batch["base_context_target_box_positions"].to(
                device=device, dtype=torch_module.long
            )
            for layer, name in enumerate(pattern_hook_names):
                pattern = cache[name]
                if pattern.ndim != 4:
                    raise ValueError(
                        f"Expected attention pattern [batch, head, query, key], got "
                        f"{tuple(pattern.shape)} at {name}."
                    )
                query_index = query_positions[:, None, None, None].expand(
                    -1, int(pattern.shape[1]), 1, int(pattern.shape[3])
                )
                key_index = key_positions[:, None, None].expand(
                    -1, int(pattern.shape[1]), 1
                )
                at_query = pattern.gather(dim=2, index=query_index).squeeze(dim=2)
                selected = at_query.gather(dim=2, index=key_index).squeeze(dim=2)
                score_sums[layer] += selected.detach().to(
                    "cpu", dtype=torch_module.float64
                ).sum(dim=0)
            total += int(batch["base_tokens"].shape[0])
            del cache
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

    if total == 0:
        raise ValueError("Cannot compute attention pattern scores from an empty dataloader.")
    return (score_sums / total).to(dtype=torch_module.float32)


def update_output_circuit_fields(output: Dict[str, Any], components: Sequence[CircuitComponent]) -> None:
    output["component_count"] = len(components)
    output["component_count_by_position"] = component_count_by_position(components)
    output["components"] = [asdict(component) for component in components]
    output["circuit_by_position"] = circuit_by_position_from_components(components)


def tensor_topk_from_candidates(
    tensor,
    candidates: Sequence[Sequence[int]],
    k: int,
    *,
    largest: bool,
) -> Tuple[List[List[int]], List[float]]:
    if k <= 0 or not candidates:
        return [], []
    scored = []
    for layer, head in candidates:
        scored.append(([int(layer), int(head)], float(tensor[int(layer), int(head)].item())))
    scored.sort(key=lambda item: item[1], reverse=largest)
    selected = scored[: min(k, len(scored))]
    return [head for head, _ in selected], [score for _, score in selected]


def load_circuit_components(
    *,
    results_dir: Path,
    utils,
    hook_component: str,
    circuit_source: str,
    largest: bool,
    include_zero_contribution_heads: bool,
    zero_tolerance: float,
    n_value_fetcher: int,
    n_pos_trans: int,
    n_pos_detect: int,
    n_struct_read: int,
    n_context_reader: int,
    pos_detector_position_mode: str,
    struct_reader_position_mode: str,
    include_context_target_reader_heads: bool,
) -> Tuple[List[CircuitComponent], Dict[str, Any]]:
    tensors = {}
    tensor_summaries = {}
    for stage, filename in STAGE_FILES.items():
        path = results_dir / filename
        if not path.exists():
            if stage == "struct_reader_context_target_box_to_target_box":
                continue
            raise FileNotFoundError(
                f"Missing path-patching score tensor: {path}\n"
                "This means circuit_eval.py is running in circuit-eval mode. "
                "For a full-model-only accuracy check, rerun with --model-only. "
                "For circuit metrics, run run_path_patching.py for this model first."
            )
        tensor = torch_load_cpu(path)
        if tensor.ndim != 2:
            raise ValueError(f"Expected [layer, head] tensor in {path}, got {tuple(tensor.shape)}.")
        tensors[stage] = tensor
        tensor_summaries[stage] = {
            "file": str(path),
            "shape": list(tensor.shape),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "nonzero": int((tensor != 0).sum().item()),
        }

    stage_ks = {
        "value_fetcher": n_value_fetcher,
        "pos_transmitter": n_pos_trans,
        "pos_detector": n_pos_detect,
        "struct_reader_swap_target": n_struct_read,
        "struct_reader_context_target_box_to_target_box": n_context_reader,
    }

    selected: Dict[str, Tuple[List[List[int]], List[float]]] = {}
    for stage, tensor in tensors.items():
        if stage == "struct_reader_context_target_box_to_target_box" and not include_context_target_reader_heads:
            continue
        selected[stage] = selected_stage_heads(
            tensor,
            k=stage_ks[stage],
            circuit_source=circuit_source,
            largest=largest,
            include_zero_contribution_heads=include_zero_contribution_heads,
            zero_tolerance=zero_tolerance,
        )

    if "value_fetcher" in selected and "pos_transmitter" in selected:
        value_heads, value_values = selected["value_fetcher"]
        pos_heads, _ = selected["pos_transmitter"]
        pos_set = {tuple(head) for head in pos_heads}
        filtered = [
            (head, value)
            for head, value in zip(value_heads, value_values)
            if tuple(head) not in pos_set
        ]
        selected["value_fetcher"] = (
            [head for head, _ in filtered],
            [value for _, value in filtered],
        )

    components: List[CircuitComponent] = []
    stage_positions = {
        "value_fetcher": "last_token",
        "pos_transmitter": "last_token",
        "pos_detector": pos_detector_position_mode,
        "struct_reader_swap_target": "swap_target_box",
        "struct_reader_context_target_box_to_target_box": struct_reader_position_mode,
    }
    for stage, (heads, values) in selected.items():
        add_components(
            components=components,
            stage=stage,
            heads=heads,
            values=values,
            position_mode=stage_positions[stage],
            hook_component=hook_component,
            utils=utils,
        )

    components = dedupe_components(components)
    return components, {
        "score_tensors": tensor_summaries,
        "selected_heads": {
            stage: heads for stage, (heads, _) in selected.items()
        },
        "selected_scores": {
            stage: values for stage, (_, values) in selected.items()
        },
    }


def random_components_like(
    *,
    components: Sequence[CircuitComponent],
    model,
    utils,
    hook_component: str,
    rng,
) -> List[CircuitComponent]:
    n_layers = int(model.cfg.n_layers)
    n_heads = int(model.cfg.n_heads)
    all_pairs = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    by_position: Dict[str, int] = defaultdict(int)
    for component in components:
        by_position[component.position_mode] += 1

    random_components: List[CircuitComponent] = []
    for position_mode, count in by_position.items():
        if count > len(all_pairs):
            raise ValueError("Cannot sample more random heads than exist in the model.")
        sampled = rng.sample(all_pairs, count)
        for layer, head in sampled:
            random_components.append(
                CircuitComponent(
                    stage="random",
                    position_mode=position_mode,
                    hook_name=hook_name(utils, hook_component, layer),
                    layer=layer,
                    head=head,
                    score=float("nan"),
                )
            )
    return random_components


def metric_value(metrics: Mapping[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        raise KeyError(f"Metric {metric_name!r} is not available in {metrics}.")
    return float(value)


def remove_one_component(
    components: Sequence[CircuitComponent], remove_idx: int
) -> List[CircuitComponent]:
    return [component for idx, component in enumerate(components) if idx != remove_idx]


def component_key(component: CircuitComponent) -> Tuple[str, str, int]:
    return (component.position_mode, component.hook_name, component.head)


def remove_component_set(
    components: Sequence[CircuitComponent],
    removal_set: Sequence[CircuitComponent],
) -> List[CircuitComponent]:
    removal_keys = {component_key(component) for component in removal_set}
    return [
        component
        for component in components
        if component_key(component) not in removal_keys
    ]


def full_model_components_for_positions(
    *,
    components: Sequence[CircuitComponent],
    model,
    utils,
    hook_component: str,
) -> List[CircuitComponent]:
    position_modes = sorted({component.position_mode for component in components})
    full_components: List[CircuitComponent] = []
    for position_mode in position_modes:
        for layer in range(int(model.cfg.n_layers)):
            for head in range(int(model.cfg.n_heads)):
                full_components.append(
                    CircuitComponent(
                        stage="full_model",
                        position_mode=position_mode,
                        hook_name=hook_name(utils, hook_component, layer),
                        layer=layer,
                        head=head,
                        score=float("nan"),
                    )
                )
    return full_components


def sample_component_subset(
    *,
    components: Sequence[CircuitComponent],
    rng,
    removal_probability: float,
) -> List[CircuitComponent]:
    return [
        component
        for component in components
        if rng.random() < removal_probability
    ]
