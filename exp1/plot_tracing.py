import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_ALIASES = {
    "accuracy": "accuracy",
    "iia": "accuracy",
    "mean_target_prob": "mean_target_prob",
    "target_prob": "mean_target_prob",
    "prob": "mean_target_prob",
}


def canonical_metric(metric_name: str) -> str:
    canonical = METRIC_ALIASES.get(metric_name.lower())
    if canonical is None:
        raise ValueError(
            f"Unknown metric {metric_name!r}. Expected one of {sorted(METRIC_ALIASES)}."
        )
    return canonical


def load_results(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ordered_axes(payload: Dict[str, object]) -> Tuple[List[int], List[int]]:
    token_axis = sorted(int(token_idx) for token_idx in payload["token_axis"])
    layer_axis = [int(layer_idx) for layer_idx in payload["layer_axis"]]
    return token_axis, layer_axis


def build_metric_matrix(
    payload: Dict[str, object],
    metric_name: str,
) -> Tuple[List[int], List[int], np.ndarray]:
    token_axis, layer_axis = ordered_axes(payload)
    matrix = np.full((len(layer_axis), len(token_axis)), np.nan, dtype=float)
    scores = payload["scores"]

    for row_idx, layer_idx in enumerate(layer_axis):
        for col_idx, token_idx in enumerate(token_axis):
            value = scores.get(str(token_idx), {}).get(str(layer_idx), {}).get(metric_name)
            if value is not None:
                matrix[row_idx, col_idx] = float(value)
    return token_axis, layer_axis, matrix


def get_token_labels(
    payload: Dict[str, object],
    token_axis: List[int],
    token_source: str,
) -> List[str]:
    labels = []
    for token_idx in token_axis:
        token_info = payload.get("token_reference", {}).get(str(token_idx), {})
        label = token_info.get(f"{token_source}_token")
        if label is None:
            label = token_info.get("clean_token")
        if label is None:
            label = token_info.get("source_token")
        labels.append(label if label is not None else "")
    return labels


def find_context_start_column(token_labels: List[str]) -> int:
    for col_idx, token_text in enumerate(token_labels):
        if token_text.strip() == "Context":
            return col_idx

    prompt_prefix = ""
    for col_idx, token_text in enumerate(token_labels):
        prompt_prefix += token_text
        if "Context:" in prompt_prefix:
            return col_idx
    return 0


def build_display_labels(token_labels: List[str]) -> List[str]:
    display_labels: List[str] = []
    for idx, token_text in enumerate(token_labels):
        if token_text.strip() == "":
            display_labels.append("")
            continue

        if (
            token_text.strip() == "Context"
            and idx + 1 < len(token_labels)
            and token_labels[idx + 1] == ":"
        ):
            display_labels.append("Context:")
            continue

        if idx > 0 and token_text == ":" and token_labels[idx - 1].strip() == "Context":
            display_labels.append("")
            continue

        display_labels.append(repr(token_text))
    return display_labels


def filter_blank_columns(
    token_axis: List[int],
    token_labels: List[str],
    display_labels: List[str],
    matrix: np.ndarray,
) -> Tuple[List[int], List[str], List[str], np.ndarray]:
    keep_indices = [idx for idx, label in enumerate(display_labels) if label != ""]
    if not keep_indices:
        return [], [], [], matrix[:, :0]

    filtered_token_axis = [token_axis[idx] for idx in keep_indices]
    filtered_token_labels = [token_labels[idx] for idx in keep_indices]
    filtered_display_labels = [display_labels[idx] for idx in keep_indices]
    filtered_matrix = matrix[:, keep_indices]
    return filtered_token_axis, filtered_token_labels, filtered_display_labels, filtered_matrix


def metric_display_name(metric_name: str) -> str:
    if metric_name == "accuracy":
        return "IIA"
    if metric_name == "mean_target_prob":
        return "Mean target probability"
    return metric_name


def infer_output_path(results_path: Path, metric_name: str) -> Path:
    suffix = "iia" if metric_name == "accuracy" else metric_name
    return results_path.with_name(f"{results_path.stem}_{suffix}_heatmap.png")


def plot_heatmap(
    *,
    payload: Dict[str, object],
    results_path: Path,
    output_path: Path,
    metric_name: str,
    token_source: str,
    include_prefix: bool,
    annotate: bool,
    dpi: int,
) -> Path:
    token_axis, layer_axis, matrix = build_metric_matrix(payload, metric_name)
    token_labels = get_token_labels(payload, token_axis, token_source)

    if not include_prefix:
        start_col = find_context_start_column(token_labels)
        token_axis = token_axis[start_col:]
        token_labels = token_labels[start_col:]
        matrix = matrix[:, start_col:]

    plot_label_tokens = build_display_labels(token_labels)
    token_axis, token_labels, plot_label_tokens, matrix = filter_blank_columns(
        token_axis,
        token_labels,
        plot_label_tokens,
        matrix,
    )

    if matrix.size == 0:
        raise ValueError("No tracing values available to plot.")

    width = max(12.0, 0.42 * len(plot_label_tokens) + 3.0)
    height = max(6.0, 0.34 * len(layer_axis) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    if metric_name == "accuracy":
        cmap = "viridis"
        vmin = 0.0
        vmax = 1.0
    else:
        cmap = "magma"
        finite_values = matrix[np.isfinite(matrix)]
        if finite_values.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(finite_values.min())
            vmax = float(finite_values.max())
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(plot_label_tokens)))
    ax.set_xticklabels(plot_label_tokens, rotation=90, fontsize=13, family="monospace")
    ax.set_yticks(np.arange(len(layer_axis)))
    ax.set_yticklabels([str(layer_idx) for layer_idx in layer_axis], fontsize=11)
    ax.set_xlabel(f"{token_source} prompt tokens", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    trace_category = payload.get("metadata", {}).get("trace_category", "unknown")
    trace_type = payload.get("metadata", {}).get("trace_type", trace_category)
    title_metric = metric_display_name(metric_name)
    context_note = "full prompt" if include_prefix else "from Context:"
    ax.set_title(
        f"{results_path.stem}\n{title_metric} heatmap ({trace_type} -> {trace_category}, {context_note})",
        fontsize=13,
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(title_metric, fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(-0.5, len(plot_label_tokens), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layer_axis), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.4, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        threshold = (vmin + vmax) / 2.0
        if matrix.shape[0] * matrix.shape[1] <= 250:
            font_size = 9
        elif matrix.shape[0] * matrix.shape[1] <= 700:
            font_size = 7
        else:
            font_size = 6

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    text = "-"
                    color = "black"
                else:
                    text = f"{value:.2f}"
                    color = "white" if value < threshold else "black"
                ax.text(
                    col_idx,
                    row_idx,
                    text,
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color=color,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a layer-by-token tracing heatmap from a tracing JSON result."
    )
    parser.add_argument("--results_path", type=Path, help="Path to a tracing JSON file.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Where to save the heatmap PNG. Defaults next to the JSON file.",
    )
    parser.add_argument(
        "--metric",
        default="iia",
        help="Metric to plot: iia/accuracy or mean_target_prob.",
    )
    parser.add_argument(
        "--token_source",
        choices=["clean", "source"],
        default="clean",
        help="Which token labels to use on the x-axis.",
    )
    parser.add_argument(
        "--include_prefix",
        action="store_true",
        help="Include tokens before 'Context:' instead of cropping the x-axis from the context section.",
    )
    parser.add_argument(
        "--no_annotate",
        action="store_true",
        help="Disable per-cell numeric annotations.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Saved figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_name = canonical_metric(args.metric)
    results_path = args.results_path.resolve()
    output_path = args.output_path.resolve() if args.output_path else infer_output_path(
        results_path,
        metric_name,
    )

    payload = load_results(results_path)
    saved_path = plot_heatmap(
        payload=payload,
        results_path=results_path,
        output_path=output_path,
        metric_name=metric_name,
        token_source=args.token_source,
        include_prefix=args.include_prefix,
        annotate=not args.no_annotate,
        dpi=args.dpi,
    )
    print(f"Saved heatmap to {saved_path}")


if __name__ == "__main__":
    main()
