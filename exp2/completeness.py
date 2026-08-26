from __future__ import annotations

from exp2.eval_core import (
    MINIMALITY_RAW_DROP_METRICS,
)

from exp2.circuit_components import (
    circuit_by_position_from_components,
    full_model_components_for_positions,
    metric_value,
    remove_component_set,
    remove_one_component,
    sample_component_subset,
)

from exp2.eval_metrics import (
    evaluate_metrics,
)

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import random
import statistics
import sys

def completeness_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> Dict[str, Optional[float]]:
    selected = [record for record in records if record["kind"] == prefix]
    scores = [float(record["incompleteness_score"]) for record in selected]
    circuit_scores = [float(record["circuit_score"]) for record in selected]
    model_scores = [float(record["model_minus_k_score"]) for record in selected]
    return {
        f"{prefix}_count": len(selected),
        f"{prefix}_incompleteness_mean": (
            float(statistics.mean(scores)) if scores else None
        ),
        f"{prefix}_incompleteness_std": (
            float(statistics.pstdev(scores)) if scores else None
        ),
        f"{prefix}_circuit_score_mean": (
            float(statistics.mean(circuit_scores)) if circuit_scores else None
        ),
        f"{prefix}_model_minus_k_score_mean": (
            float(statistics.mean(model_scores)) if model_scores else None
        ),
    }


def plot_completeness_records(
    *,
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
    figure_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random_records = [record for record in records if record["kind"] == "random"]
    stage_records = [record for record in records if record["kind"] == "stage"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    if random_records:
        ax.scatter(
            [record["circuit_score"] for record in random_records],
            [record["model_minus_k_score"] for record in random_records],
            c="tab:red",
            label="Random set",
            alpha=0.8,
        )

    colors = [
        "tab:green",
        "gold",
        "tab:purple",
        "tab:blue",
        "tab:cyan",
        "tab:brown",
        "tab:pink",
    ]
    for record, color in zip(stage_records, colors):
        ax.scatter(
            record["circuit_score"],
            record["model_minus_k_score"],
            c=color,
            label=record["name"],
            marker="D",
            s=70,
            edgecolor="black",
            linewidth=0.4,
        )

    all_scores = [
        float(record[key])
        for record in records
        for key in ("circuit_score", "model_minus_k_score")
        if record.get(key) is not None
    ]
    bounded_metric = metric_name in {
        "accuracy",
        "full_vocab_accuracy",
        "candidate_accuracy",
        "mean_label_prob",
    }
    if bounded_metric:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    elif all_scores:
        lo, hi = min(all_scores), max(all_scores)
        pad = max((hi - lo) * 0.08, 1e-6)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    ax.axline((0, 0), slope=1, color="gainsboro", linestyle="--", label="y = x")
    ax.set_xlabel(f"F(C \\ K), {metric_name}")
    ax.set_ylabel(f"F(M \\ K), {metric_name}")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def resolve_completeness_figure_path(
    *,
    requested_path: Optional[Path],
    results_dir: Path,
    model_folder: str,
    metric_name: str,
) -> Path:
    filename = f"{model_folder}_completeness_{metric_name}.pdf"
    if requested_path is None:
        return results_dir / filename
    path = requested_path if requested_path.is_absolute() else Path.cwd() / requested_path
    if path.suffix:
        return path
    return path / filename


def evaluate_completeness_subset(
    *,
    name: str,
    kind: str,
    removal_set: Sequence[CircuitComponent],
    model,
    dataloader,
    device: Any,
    circuit_components: Sequence[CircuitComponent],
    full_model_components: Sequence[CircuitComponent],
    mean_activations: Mapping[str, Any],
    hook_names: Sequence[str],
    primary_metric: str,
    ablate_non_circuit_positions: bool,
) -> Dict[str, Any]:
    circuit_minus_k = remove_component_set(circuit_components, removal_set)
    model_minus_k = remove_component_set(full_model_components, removal_set)

    circuit_metrics = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position_from_components(circuit_minus_k),
        mean_activations=mean_activations,
        ablate_non_circuit_positions=ablate_non_circuit_positions,
        preserve_all_heads_at_circuit_positions=False,
    )
    model_minus_k_metrics = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position_from_components(model_minus_k),
        mean_activations=mean_activations,
        ablate_non_circuit_positions=False,
        preserve_all_heads_at_circuit_positions=False,
    )
    circuit_score = metric_value(circuit_metrics, primary_metric)
    model_minus_k_score = metric_value(model_minus_k_metrics, primary_metric)
    gap = abs(model_minus_k_score - circuit_score)

    return {
        "name": name,
        "kind": kind,
        "removed_component_count": len(removal_set),
        "removed_components": [asdict(component) for component in removal_set],
        "circuit_component_count": len(circuit_minus_k),
        "full_model_component_count": len(model_minus_k),
        "circuit_metrics": circuit_metrics,
        "model_minus_k_metrics": model_minus_k_metrics,
        "metric_name": primary_metric,
        "circuit_score": circuit_score,
        "model_minus_k_score": model_minus_k_score,
        "incompleteness_score": gap,
        "completeness_score": 1.0 - gap,
    }


def run_completeness_evaluation(
    *,
    model,
    dataloader,
    device: Any,
    components: Sequence[CircuitComponent],
    mean_activations: Mapping[str, Any],
    hook_names: Sequence[str],
    primary_metric: str,
    random_iters: int,
    random_removal_probability: float,
    seed: int,
    ablate_non_circuit_positions: bool,
    utils,
    hook_component: str,
    figure_path: Optional[Path],
    plot_title: str,
) -> Dict[str, Any]:
    if not 0.0 <= random_removal_probability <= 1.0:
        raise ValueError("--completeness-random-removal-probability must be in [0, 1].")

    circuit_components = list(components)
    full_components = full_model_components_for_positions(
        components=circuit_components,
        model=model,
        utils=utils,
        hook_component=hook_component,
    )
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []

    for idx in range(random_iters):
        removal_set = sample_component_subset(
            components=circuit_components,
            rng=rng,
            removal_probability=random_removal_probability,
        )
        print(
            f"Evaluating completeness random set {idx + 1}/{random_iters}...",
            file=sys.stderr,
        )
        records.append(
            evaluate_completeness_subset(
                name=f"random_{idx}",
                kind="random",
                removal_set=removal_set,
                model=model,
                dataloader=dataloader,
                device=device,
                circuit_components=circuit_components,
                full_model_components=full_components,
                mean_activations=mean_activations,
                hook_names=hook_names,
                primary_metric=primary_metric,
                ablate_non_circuit_positions=ablate_non_circuit_positions,
            )
        )

    components_by_stage: Dict[str, List[CircuitComponent]] = defaultdict(list)
    for component in circuit_components:
        components_by_stage[component.stage].append(component)
    for stage in sorted(components_by_stage):
        print(f"Evaluating completeness stage {stage}...", file=sys.stderr)
        records.append(
            evaluate_completeness_subset(
                name=stage,
                kind="stage",
                removal_set=components_by_stage[stage],
                model=model,
                dataloader=dataloader,
                device=device,
                circuit_components=circuit_components,
                full_model_components=full_components,
                mean_activations=mean_activations,
                hook_names=hook_names,
                primary_metric=primary_metric,
                ablate_non_circuit_positions=ablate_non_circuit_positions,
            )
        )

    result: Dict[str, Any] = {
        "metric_name": primary_metric,
        "random_iters": random_iters,
        "random_removal_probability": random_removal_probability,
        "component_count": len(circuit_components),
        "full_model_component_count": len(full_components),
        "ablate_non_circuit_positions_for_circuit": ablate_non_circuit_positions,
        "ablate_non_circuit_positions_for_model_minus_k": False,
        "summary": {
            **completeness_summary(records, prefix="random"),
            **completeness_summary(records, prefix="stage"),
        },
        "records": records,
    }

    if figure_path is not None:
        plot_completeness_records(
            records=records,
            metric_name=primary_metric,
            figure_path=figure_path,
            title=plot_title,
        )
        result["figure_path"] = str(figure_path)

    return result


def run_minimality_pruning(
    *,
    model,
    dataloader,
    device: Any,
    components: Sequence[CircuitComponent],
    mean_activations: Mapping[str, Any],
    hook_names: Sequence[str],
    primary_metric: str,
    minimality_threshold: float,
    minimality_criterion: str,
    ablate_non_circuit_positions: bool,
    preserve_all_heads_at_circuit_positions: bool,
) -> Tuple[List[CircuitComponent], Dict[str, Any]]:
    remaining = list(components)
    removed = []
    initial_metrics = evaluate_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position_from_components(remaining),
        mean_activations=mean_activations,
        ablate_non_circuit_positions=ablate_non_circuit_positions,
        preserve_all_heads_at_circuit_positions=preserve_all_heads_at_circuit_positions,
    )
    current_metrics = initial_metrics
    current_score = metric_value(current_metrics, primary_metric)

    idx = 0
    step = 0
    while idx < len(remaining):
        component = remaining[idx]
        trial_components = remove_one_component(remaining, idx)
        trial_metrics = evaluate_metrics(
            model=model,
            dataloader=dataloader,
            device=device,
            hook_names=hook_names,
            circuit_by_position=circuit_by_position_from_components(trial_components),
            mean_activations=mean_activations,
            ablate_non_circuit_positions=ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=preserve_all_heads_at_circuit_positions,
        )
        trial_score = metric_value(trial_metrics, primary_metric)
        contribution = current_score - trial_score
        if minimality_criterion == "retention":
            remove_component = trial_score >= current_score * minimality_threshold
            normalized = (
                contribution / current_score if current_score not in (0.0, -0.0) else math.inf
            )
        elif minimality_criterion == "minimality-py":
            if primary_metric in MINIMALITY_RAW_DROP_METRICS:
                normalized = contribution
            else:
                normalized = (
                    current_score / trial_score - 1.0
                    if trial_score > 0
                    else math.inf
                )
            remove_component = normalized < minimality_threshold
        else:
            raise ValueError(f"Unknown minimality criterion {minimality_criterion!r}.")

        if remove_component:
            removed_component = remaining.pop(idx)
            removed.append(
                {
                    **asdict(removed_component),
                    "step": step,
                    "score_before": current_score,
                    "score_after": trial_score,
                    "contribution": contribution,
                    "normalized_contribution": normalized,
                }
            )
            current_metrics = trial_metrics
            current_score = trial_score
            step += 1
        else:
            idx += 1

    final_metrics = current_metrics
    return remaining, {
        "criterion": minimality_criterion,
        "metric_name": primary_metric,
        "threshold": minimality_threshold,
        "initial_component_count": len(components),
        "removed_component_count": len(removed),
        "final_component_count": len(remaining),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "removed_components": removed,
        "kept_components": [asdict(component) for component in remaining],
    }
