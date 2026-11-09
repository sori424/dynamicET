"""Overlay exp1 tracing results as a layer-by-token SVG heatmap.

Run from the dynamicET root with ``python -m exp1.plot_tracing_overlay_svg``
or run this file directly. Inputs are JSON files produced by exp1/tracing.py;
all inputs must have matching token and layer axes. Only the Python standard
library is required. See the repository README for a complete example.

Adapted from binding-iclr/plot_tracing_overlay_svg.py, preserving its rendering
and command-line options.
"""

import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


METRIC_ALIASES = {
    "accuracy": "accuracy",
    "iia": "accuracy",
    "mean_target_prob": "mean_target_prob",
    "target_prob": "mean_target_prob",
    "prob": "mean_target_prob",
}

DEFAULT_SERIES_COLORS = ["#d8a34a", "#cf7f97", "#78a9d4"]
LOW_SCORE_FLOOR = 0.2
SCORE_GAMMA = 1.25
ALPHA_BLEND_SCALE = 0.62
ALPHA_BLEND_CAP = 0.82


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
) -> Tuple[List[int], List[int], List[List[float]]]:
    token_axis, layer_axis = ordered_axes(payload)
    matrix = [[math.nan for _ in token_axis] for _ in layer_axis]
    scores = payload["scores"]

    for row_idx, layer_idx in enumerate(layer_axis):
        for col_idx, token_idx in enumerate(token_axis):
            value = scores.get(str(token_idx), {}).get(str(layer_idx), {}).get(metric_name)
            if value is not None:
                matrix[row_idx][col_idx] = float(value)
    return token_axis, layer_axis, matrix


def get_token_text(
    payload: Dict[str, object],
    token_idx: int,
    field: str,
) -> str:
    token_info = payload.get("token_reference", {}).get(str(token_idx), {})
    value = token_info.get(field)
    if value is not None:
        return value
    fallback = token_info.get("clean_token")
    if fallback is not None:
        return fallback
    fallback = token_info.get("source_token")
    return fallback if fallback is not None else ""


def merged_token_labels(payloads: Sequence[Dict[str, object]], token_axis: List[int]) -> List[str]:
    labels: List[str] = []
    for token_idx in token_axis:
        base_token = get_token_text(payloads[0], token_idx, "source_token")
        alternatives: List[str] = []
        for payload in payloads:
            clean_token = get_token_text(payload, token_idx, "clean_token")
            if clean_token != base_token:
                if clean_token not in alternatives:
                    alternatives.append(clean_token)

        if alternatives:
            base_display = base_token.strip()
            alt_display = "/".join(token.strip() for token in alternatives)
            labels.append(f"{base_display} → {alt_display}")
        else:
            labels.append(base_token)
    return labels


def blend_hex_colors(hex_colors: Sequence[str]) -> str:
    if not hex_colors:
        return "#222222"

    rgb_values = [hex_to_rgb(hex_color) for hex_color in hex_colors]
    blended = tuple(
        sum(rgb[channel_idx] for rgb in rgb_values) / len(rgb_values)
        for channel_idx in range(3)
    )
    return rgb_to_hex(blended)


def build_token_label_colors(
    payloads: Sequence[Dict[str, object]],
    token_axis: Sequence[int],
    series_colors: Sequence[str],
    default_color: str = "#222222",
) -> List[str]:
    label_colors: List[str] = []

    for token_idx in token_axis:
        base_token = get_token_text(payloads[0], token_idx, "source_token")
        active_colors: List[str] = []

        for payload, series_color in zip(payloads, series_colors):
            clean_token = get_token_text(payload, token_idx, "clean_token")
            if clean_token != base_token:
                active_colors.append(series_color)

        if not active_colors:
            label_colors.append(default_color)
            continue

        label_colors.append(blend_hex_colors(active_colors))

    return label_colors


def find_context_start_column(token_labels: Sequence[str]) -> int:
    for col_idx, token_text in enumerate(token_labels):
        if token_text.strip() == "Context":
            return col_idx

    prompt_prefix = ""
    for col_idx, token_text in enumerate(token_labels):
        prompt_prefix += token_text
        if "Context:" in prompt_prefix:
            return col_idx
    return 0


def build_display_labels(token_labels: Sequence[str]) -> List[str]:
    display_labels: List[str] = []

    for idx, token_text in enumerate(token_labels):
        normalized = token_text.replace("\n", " ").strip()
        if normalized == "":
            display_labels.append("")
            continue

        display_labels.append(normalized)
    return display_labels


def parse_axis_label_prefixes(prefix_specs: Optional[Sequence[str]]) -> Dict[str, str]:
    if not prefix_specs:
        return {}

    prefix_map: Dict[str, str] = {}
    for spec in prefix_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --axis_label_prefix value {spec!r}. Expected LABEL=PREFIX."
            )
        label, prefix = spec.split("=", 1)
        normalized_label = label.strip()
        if normalized_label == "":
            raise ValueError("Axis label prefix mappings must include a non-empty label.")
        prefix_map[normalized_label] = prefix
    return prefix_map


def apply_axis_label_prefixes(
    display_labels: Sequence[str],
    axis_label_prefixes: Dict[str, str],
) -> List[str]:
    if not axis_label_prefixes:
        return list(display_labels)

    prefixed_labels: List[str] = []
    for label in display_labels:
        prefix = axis_label_prefixes.get(label)
        if prefix is None:
            prefixed_labels.append(label)
        else:
            prefixed_labels.append(f"{prefix}{label}")
    return prefixed_labels


def filter_blank_columns(
    token_axis: List[int],
    display_labels: List[str],
    label_colors: List[str],
    matrices: List[List[List[float]]],
    excluded_labels: Optional[Sequence[str]] = None,
) -> Tuple[List[int], List[str], List[str], List[List[List[float]]]]:
    excluded_label_set = set(excluded_labels or [])
    keep_indices = [
        idx
        for idx, label in enumerate(display_labels)
        if label != "" and label not in excluded_label_set
    ]
    filtered_axis = [token_axis[idx] for idx in keep_indices]
    filtered_labels = [display_labels[idx] for idx in keep_indices]
    filtered_label_colors = [label_colors[idx] for idx in keep_indices]
    filtered_matrices: List[List[List[float]]] = []

    for matrix in matrices:
        filtered_matrices.append(
            [[row[idx] for idx in keep_indices] for row in matrix]
        )

    return filtered_axis, filtered_labels, filtered_label_colors, filtered_matrices


def score_to_strength(score: float) -> float:
    if math.isnan(score) or score <= LOW_SCORE_FLOOR:
        return 0.0
    normalized = (score - LOW_SCORE_FLOOR) / (1.0 - LOW_SCORE_FLOOR)
    return max(0.0, min(1.0, normalized)) ** SCORE_GAMMA


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    stripped = hex_color.lstrip("#")
    if len(stripped) != 6:
        raise ValueError(f"Expected a 6-digit hex color, got {hex_color!r}.")
    return (
        int(stripped[0:2], 16) / 255.0,
        int(stripped[2:4], 16) / 255.0,
        int(stripped[4:6], 16) / 255.0,
    )


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    channels = []
    for channel in rgb:
        clamped = max(0.0, min(1.0, channel))
        channels.append(f"{int(round(clamped * 255)):02x}")
    return "#" + "".join(channels)


def combine_scores_into_color(
    scores: Sequence[float],
    colors: Sequence[Tuple[float, float, float]],
    visibility_thresholds: Sequence[float],
    blend_mode: str,
) -> str:
    if blend_mode == "max":
        best_strength = 0.0
        best_color = (1.0, 1.0, 1.0)

        for score, color, visibility_threshold in zip(
            scores,
            colors,
            visibility_thresholds,
        ):
            if math.isnan(score) or score <= visibility_threshold:
                continue
            strength = score_to_strength(score)
            if strength > best_strength:
                best_strength = strength
                best_color = color

        rgb = [
            1.0 - best_strength * (1.0 - best_color[channel_idx])
            for channel_idx in range(3)
        ]
        return rgb_to_hex((rgb[0], rgb[1], rgb[2]))

    if blend_mode != "alpha":
        raise ValueError(f"Unknown blend mode {blend_mode!r}.")

    weighted_sum = [0.0, 0.0, 0.0]
    alpha_sum = 0.0

    for score, color, visibility_threshold in zip(
        scores,
        colors,
        visibility_thresholds,
    ):
        if math.isnan(score) or score <= visibility_threshold:
            continue
        strength = score_to_strength(score)
        alpha = ALPHA_BLEND_SCALE * strength
        if alpha <= 0.0:
            continue
        alpha_sum += alpha
        for channel_idx in range(3):
            weighted_sum[channel_idx] += alpha * color[channel_idx]

    if alpha_sum <= 0.0:
        return "#ffffff"

    total_alpha = min(ALPHA_BLEND_CAP, alpha_sum)
    mixed_rgb = [weighted_sum[channel_idx] / alpha_sum for channel_idx in range(3)]
    rgb = [
        (1.0 - total_alpha) + total_alpha * mixed_rgb[channel_idx]
        for channel_idx in range(3)
    ]
    return rgb_to_hex((rgb[0], rgb[1], rgb[2]))


def metric_display_name(metric_name: str) -> str:
    if metric_name == "accuracy":
        return "IIA"
    if metric_name == "mean_target_prob":
        return "Mean target probability"
    return metric_name


def estimate_text_width(label: str, font_size: float) -> float:
    width_units = 0.0
    for char in label:
        if char == " ":
            width_units += 0.33
        elif char.isupper():
            width_units += 0.62
        elif char.islower():
            width_units += 0.52
        elif char.isdigit():
            width_units += 0.5
        else:
            width_units += 0.4
    return width_units * font_size * 0.82


def render_svg(
    *,
    matrices: List[List[List[float]]],
    layer_axis: List[int],
    display_labels: List[str],
    x_label_colors: List[str],
    legend_labels: List[str],
    legend_colors: List[str],
    visibility_thresholds: List[float],
    metric_label: str,
    blend_mode: str,
    output_path: Path,
) -> Path:
    n_rows = len(layer_axis)
    n_cols = len(display_labels)

    cell_width = 22
    cell_height = 8
    x_label_font_size = 25
    y_tick_font_size = 25
    axis_label_font_size = 25
    legend_font_size = 25
    colorbar_tick_font_size = 17
    colorbar_label_font_size = 20
    plot_width = n_cols * cell_width
    plot_height = n_rows * cell_height

    left_margin = 90
    top_margin = 72
    right_margin = 128
    bottom_margin = 210
    width = left_margin + plot_width + right_margin
    height = top_margin + plot_height + bottom_margin

    plot_x = left_margin
    plot_y = top_margin
    plot_frame_x = plot_x + 0.5
    plot_frame_y = plot_y + 0.5
    plot_frame_width = plot_width - 1
    plot_frame_height = plot_height - 1
    colorbar_x = plot_x + plot_width + 22
    colorbar_y = plot_y
    colorbar_width = 14
    colorbar_height = plot_height

    series_rgb = [hex_to_rgb(color) for color in legend_colors]

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<linearGradient id="metricScale" x1="0%" y1="100%" x2="0%" y2="0%">',
        '<stop offset="0%" stop-color="#ffffff"/>',
        '<stop offset="20%" stop-color="#ffffff"/>',
        '<stop offset="50%" stop-color="#d8d8d8"/>',
        '<stop offset="100%" stop-color="#5f5f5f"/>',
        "</linearGradient>",
        "</defs>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" height="{plot_height}" fill="#ffffff"/>',
    ]

    for row_idx, layer_idx in enumerate(layer_axis):
        y = plot_y + (n_rows - 1 - row_idx) * cell_height
        for col_idx in range(n_cols):
            scores = [matrix[row_idx][col_idx] for matrix in matrices]
            fill = combine_scores_into_color(
                scores,
                series_rgb,
                visibility_thresholds,
                blend_mode,
            )
            x = plot_x + col_idx * cell_width
            svg_lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}" stroke="none"/>'
            )

    svg_lines.append(
        f'<rect x="{plot_frame_x}" y="{plot_frame_y}" width="{plot_frame_width}" height="{plot_frame_height}" fill="none" stroke="#222222" stroke-width="1" shape-rendering="crispEdges"/>'
    )

    tick_step = 10
    for row_idx, layer_idx in enumerate(layer_axis):
        if layer_idx % tick_step != 0:
            continue
        y = plot_y + (n_rows - 1 - row_idx + 0.5) * cell_height
        svg_lines.append(
            f'<line x1="{plot_x - 5}" y1="{y}" x2="{plot_x}" y2="{y}" stroke="#222222" stroke-width="1"/>'
        )
        svg_lines.append(
            f'<text x="{plot_x - 10}" y="{y + 5}" font-size="{y_tick_font_size}" text-anchor="end" fill="#222222">{escape(str(layer_idx))}</text>'
        )

    svg_lines.append(
        f'<text x="28" y="{plot_y + plot_height / 2}" font-size="{axis_label_font_size}" text-anchor="middle" fill="#222222" transform="rotate(-90 28 {plot_y + plot_height / 2})">Layers</text>'
    )

    for col_idx, (label, label_color) in enumerate(zip(display_labels, x_label_colors)):
        x = plot_x + col_idx * cell_width + cell_width / 2
        y = plot_y + plot_height + 10
        svg_lines.append(
            f'<text x="{x}" y="{y}" font-size="{x_label_font_size}" text-anchor="end" fill="{label_color}" transform="rotate(-90 {x} {y})">{escape(label)}</text>'
        )

    legend_row_height = 22
    legend_box = 12
    legend_text_offset = 6
    legend_item_gap = 18
    legend_padding_x = 30
    legend_padding_y = 15
    legend_item_widths = [
        legend_box + legend_text_offset + estimate_text_width(label, legend_font_size)
        for label in legend_labels
    ]
    legend_width = int(
        math.ceil(
            2 * legend_padding_x
            + sum(legend_item_widths)
            + legend_item_gap * max(0, len(legend_labels) - 1)
        )
    )
    legend_height = legend_box + 2 * legend_padding_y
    legend_x = plot_x + 8
    legend_gap = 10
    legend_y = plot_y - legend_height - legend_gap
    svg_lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" fill="#ffffff" stroke="#c7c7c7" stroke-width="1" rx="3" ry="3"/>'
    )
    current_x = legend_x + legend_padding_x
    legend_item_y = legend_y + legend_padding_y
    for idx, (label, color) in enumerate(zip(legend_labels, legend_colors)):
        svg_lines.append(
            f'<rect x="{current_x}" y="{legend_item_y}" width="{legend_box}" height="{legend_box}" fill="{color}" stroke="none"/>'
        )
        svg_lines.append(
            f'<text x="{current_x + legend_box + legend_text_offset}" y="{legend_item_y + legend_box - 1}" font-size="{legend_font_size}" fill="#222222">{escape(label)}</text>'
        )
        current_x += legend_item_widths[idx] + legend_item_gap

    svg_lines.append(
        f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_width}" height="{colorbar_height}" fill="url(#metricScale)" stroke="#222222" stroke-width="1"/>'
    )
    svg_lines.append(
        f'<text x="{colorbar_x + colorbar_width + 12}" y="{colorbar_y + 6}" font-size="{colorbar_tick_font_size}" fill="#222222">1</text>'
    )
    svg_lines.append(
        f'<text x="{colorbar_x + colorbar_width + 12}" y="{colorbar_y + colorbar_height}" font-size="{colorbar_tick_font_size}" fill="#222222">0</text>'
    )
    svg_lines.append(
        f'<text x="{colorbar_x + colorbar_width + 40}" y="{colorbar_y + colorbar_height / 2}" font-size="{colorbar_label_font_size}" text-anchor="middle" fill="#222222" transform="rotate(-90 {colorbar_x + colorbar_width + 40} {colorbar_y + colorbar_height / 2})">{escape(metric_label)}</text>'
    )

    svg_lines.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay multiple tracing JSON heatmaps into a single SVG."
    )
    parser.add_argument(
        "--results_paths",
        nargs="+",
        type=Path,
        required=True,
        help="Tracing JSON files to overlay. All files must share the same token and layer axes.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Where to save the combined SVG.",
    )
    parser.add_argument(
        "--metric",
        default="iia",
        help="Metric to plot: iia/accuracy or mean_target_prob.",
    )
    parser.add_argument(
        "--series_labels",
        nargs="*",
        default=None,
        help="Legend labels in the same order as --results_paths.",
    )
    parser.add_argument(
        "--series_colors",
        nargs="*",
        default=DEFAULT_SERIES_COLORS,
        help="Hex colors in the same order as --results_paths.",
    )
    parser.add_argument(
        "--series_visibility_thresholds",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Hide scores at or below this value for each series, in the same order as "
            "--results_paths. Defaults to 0.2 for every series."
        ),
    )
    parser.add_argument(
        "--include_prefix",
        action="store_true",
        help="Include the instruction prefix instead of cropping labels from Context: onward.",
    )
    parser.add_argument(
        "--blend_mode",
        choices=["alpha", "max"],
        default="max",
        help="How to combine overlapping series in one cell.",
    )
    parser.add_argument(
        "--axis_label_prefix",
        action="append",
        default=None,
        help="Prefix a specific x-axis label using LABEL=PREFIX. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude_axis_label",
        action="append",
        default=None,
        help="Remove every x-axis column with this exact display label. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_name = canonical_metric(args.metric)
    payloads = [load_results(path.resolve()) for path in args.results_paths]

    matrices: List[List[List[float]]] = []
    token_axis: List[int] = []
    layer_axis: List[int] = []

    for idx, payload in enumerate(payloads):
        payload_token_axis, payload_layer_axis, matrix = build_metric_matrix(payload, metric_name)
        if idx == 0:
            token_axis = payload_token_axis
            layer_axis = payload_layer_axis
        else:
            if payload_token_axis != token_axis:
                raise ValueError("All inputs must share the same token axis.")
            if payload_layer_axis != layer_axis:
                raise ValueError("All inputs must share the same layer axis.")
        matrices.append(matrix)

    if args.series_labels is None or len(args.series_labels) == 0:
        series_labels = [path.stem for path in args.results_paths]
    else:
        if len(args.series_labels) != len(args.results_paths):
            raise ValueError("Expected one --series_labels entry per results file.")
        series_labels = list(args.series_labels)

    if len(args.series_colors) != len(args.results_paths):
        raise ValueError("Expected one --series_colors entry per results file.")

    series_colors = list(args.series_colors)
    if (
        args.series_visibility_thresholds is None
        or len(args.series_visibility_thresholds) == 0
    ):
        series_visibility_thresholds = [LOW_SCORE_FLOOR] * len(args.results_paths)
    else:
        if len(args.series_visibility_thresholds) != len(args.results_paths):
            raise ValueError(
                "Expected one --series_visibility_thresholds entry per results file."
            )
        series_visibility_thresholds = list(args.series_visibility_thresholds)

    for visibility_threshold in series_visibility_thresholds:
        if not 0.0 <= visibility_threshold < 1.0:
            raise ValueError("Series visibility thresholds must be in the range [0, 1).")

    token_labels = merged_token_labels(payloads, token_axis)
    token_label_colors = build_token_label_colors(payloads, token_axis, series_colors)
    if not args.include_prefix:
        start_col = find_context_start_column(token_labels)
        token_axis = token_axis[start_col:]
        token_labels = token_labels[start_col:]
        token_label_colors = token_label_colors[start_col:]
        matrices = [[row[start_col:] for row in matrix] for matrix in matrices]

    display_labels = build_display_labels(token_labels)
    axis_label_prefixes = parse_axis_label_prefixes(args.axis_label_prefix)
    display_labels = apply_axis_label_prefixes(display_labels, axis_label_prefixes)
    token_axis, display_labels, token_label_colors, matrices = filter_blank_columns(
        token_axis,
        display_labels,
        token_label_colors,
        matrices,
        args.exclude_axis_label,
    )

    metric_label = metric_display_name(metric_name)
    saved_path = render_svg(
        matrices=matrices,
        layer_axis=layer_axis,
        display_labels=display_labels,
        x_label_colors=token_label_colors,
        legend_labels=series_labels,
        legend_colors=series_colors,
        visibility_thresholds=series_visibility_thresholds,
        metric_label=metric_label,
        blend_mode=args.blend_mode,
        output_path=args.output_path.resolve(),
    )
    print(f"Saved overlay heatmap to {saved_path}")


if __name__ == "__main__":
    main()
