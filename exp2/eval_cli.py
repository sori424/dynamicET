from __future__ import annotations

from exp2.eval_core import (
    METRIC_CHOICES,
    PROMPT_FORMATS,
    SYSTEM_PROMPT,
    find_binding_root,
)

from exp2.eval_runner import (
    evaluate_one_model,
    write_json,
)

from pathlib import Path
import argparse
import sys

def parse_swap_pair(value: str) -> Tuple[int, int]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected a pair like 1,0.")
    try:
        first, second = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Swap pair entries must be integers.") from exc
    if first == second:
        raise argparse.ArgumentTypeError("Swap pair entries must differ.")
    return first, second


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BOXES circuits recovered from path-patching score tensors."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemma9b"],
        help="Result subfolders to evaluate, e.g. gemma9b llama8b qwen7b.",
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        default=None,
        help="Override model name when evaluating exactly one --models entry.",
    )
    parser.add_argument(
        "--prompt-format",
        "--prompt_format",
        choices=PROMPT_FORMATS,
        default="raw",
        help=(
            "Prompt wrapper for generated BOXES eval data. Use the same value "
            "as run_path_patching.py. `auto` uses assistant-prefill chat prompts "
            "for instruct models, keeping `Answer:` in the assistant message so "
            "the next token is the object."
        ),
    )
    parser.add_argument(
        "--system-prompt",
        "--system_prompt",
        default=SYSTEM_PROMPT,
        help=(
            "System instruction used for chat-formatted prompts. Pass an empty "
            "string to omit the system role when the tokenizer chat template is used."
        ),
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        type=Path,
        default=None,
        help="Path to a model result folder or to the parent results/path_patching folder.",
    )
    parser.add_argument(
        "--output-json",
        "--output_json",
        type=Path,
        default=None,
        help="Where to write metrics JSON. Defaults inside each model results folder.",
    )
    parser.add_argument("--num-samples", "--num_samples", type=int, default=300)
    parser.add_argument("--mean-samples", "--mean_samples", type=int, default=100)
    parser.add_argument(
        "--run-heldout",
        "--run_heldout",
        action="store_true",
        help=(
            "After circuit selection/pruning on the main prompt range, evaluate "
            "the frozen circuit on a disjoint held-out prompt range."
        ),
    )
    parser.add_argument(
        "--heldout-samples",
        "--heldout_samples",
        type=int,
        default=None,
        help="Held-out eval sample count. Defaults to --num-samples.",
    )
    parser.add_argument(
        "--heldout-prompt-id-start",
        "--heldout_prompt_id_start",
        type=int,
        default=None,
        help=(
            "First prompt id for held-out eval. Defaults to "
            "--prompt-id-start + --num-samples."
        ),
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument(
        "--random-circuit-iters",
        "--random_circuit_iters",
        type=int,
        default=10,
    )
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--vocab-tag", "--vocab_tag", default="BOXES")
    parser.add_argument("--vocab-split", "--vocab_split", default="train")
    parser.add_argument("--num-entities", "--num_entities", type=int, default=3)
    parser.add_argument("--query-name", "--query_name", type=int, default=0)
    parser.add_argument("--prompt-id-start", "--prompt_id_start", type=int, default=0)
    parser.add_argument("--swap-pair", "--swap_pair", type=parse_swap_pair, default=(1, 0))
    parser.add_argument("--template-type", "--template_type", default="normal")
    parser.add_argument(
        "--counterfactual-mode",
        "--counterfactual_mode",
        choices=["path_patching_shifted", "same_objects", "new_swap_objects"],
        default="path_patching_shifted",
        help=(
            "How to build source prompts. The default matches "
            "run_path_patching.py: vocab.default_context with shifted "
            "entities/attributes."
        ),
    )
    parser.add_argument(
        "--hook-component",
        "--hook_component",
        choices=["z", "result", "o_proj"],
        default="o_proj",
        help=(
            "Activation hook to mean-ablate. `o_proj` means the attention "
            "output-projection input, which maps to TransformerLens `hook_z`."
        ),
    )
    parser.add_argument(
        "--circuit-source",
        "--circuit_source",
        choices=["topk", "all-negative", "all-positive"],
        default="topk",
    )
    parser.add_argument(
        "--context-reader-source",
        "--context_reader_source",
        choices=["path-patching", "attention-pattern"],
        default="path-patching",
        help=(
            "How to select struct_reader_context_target_box_to_target_box heads. "
            "attention-pattern ranks heads by average attention from the swap-target "
            "box token to the context-target box token."
        ),
    )
    parser.add_argument(
        "--attention-pattern-samples",
        "--attention_pattern_samples",
        type=int,
        default=100,
        help=(
            "Number of eval examples used when --context-reader-source "
            "attention-pattern."
        ),
    )
    parser.add_argument(
        "--attention-pattern-position-mode",
        "--attention_pattern_position_mode",
        default="swap_target_box",
        help=(
            "Position where attention-pattern context-reader heads are preserved. "
            "For swap_target_box -> context_target_box attention, the useful head "
            "output is at the query position, so the default is swap_target_box."
        ),
    )
    parser.add_argument(
        "--attention-pattern-candidate-source",
        "--attention_pattern_candidate_source",
        choices=["all", "struct-reader-swap-target"],
        default="all",
        help=(
            "Candidate heads to rank by attention pattern. Use all for a direct "
            "attention scan, or struct-reader-swap-target to only rank the "
            "path-patched swap-target reader heads."
        ),
    )
    parser.add_argument("--largest", action="store_true", help="Use largest top-k scores.")
    parser.add_argument("--n-value-fetcher", "--n_value_fetcher", type=int, default=50)
    parser.add_argument("--n-pos-trans", "--n_pos_trans", type=int, default=50)
    parser.add_argument("--n-pos-detect", "--n_pos_detect", type=int, default=25)
    parser.add_argument("--n-struct-read", "--n_struct_read", type=int, default=10)
    parser.add_argument("--n-context-reader", "--n_context_reader", type=int, default=10)
    parser.add_argument(
        "--pos-detector-position-mode",
        "--pos_detector_position_mode",
        default="question_query_box",
    )
    parser.add_argument(
        "--struct-reader-position-mode",
        "--struct_reader_position_mode",
        default="context_target_box",
    )
    parser.add_argument(
        "--include-context-target-reader-heads",
        "--include_context_target_reader_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-zero-contribution-heads",
        "--include_zero_contribution_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--zero-contribution-tolerance",
        "--zero_contribution_tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--ablate-non-circuit-positions",
        "--ablate_non_circuit_positions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preserve-all-heads-at-circuit-positions",
        "--preserve_all_heads_at_circuit_positions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--primary-metric",
        "--primary_metric",
        default="candidate_accuracy",
        choices=METRIC_CHOICES,
    )
    parser.add_argument("--run-minimality", "--run_minimality", action="store_true")
    parser.add_argument("--minimality-samples", "--minimality_samples", type=int, default=20)
    parser.add_argument(
        "--minimality-metric",
        "--minimality_metric",
        default="candidate_accuracy",
        choices=METRIC_CHOICES,
        help=(
            "Metric used for minimality pruning. Defaults to mean_label_logit "
            "so pruning is based on a continuous score instead of candidate_accuracy."
        ),
    )
    parser.add_argument(
        "--minimality-threshold",
        "--minimality_threshold",
        type=float,
        default=0.01,
        help=(
            "For minimality-py, remove a head when the metric drop is below "
            "this value. Accuracy/probability metrics use before/after - 1; "
            "logit/margin metrics use raw before - after. For retention, "
            "remove a head if the metric stays above threshold * current_metric."
        ),
    )
    parser.add_argument(
        "--minimality-criterion",
        "--minimality_criterion",
        choices=["minimality-py", "retention"],
        default="minimality-py",
    )
    parser.add_argument(
        "--run-completeness",
        "--run_completeness",
        action="store_true",
        help=(
            "Run completeness tests: remove K from the circuit and compare "
            "F(C \\ K) against F(M \\ K)."
        ),
    )
    parser.add_argument(
        "--completeness-samples",
        "--completeness_samples",
        type=int,
        default=20,
        help="Number of eval examples used for completeness tests.",
    )
    parser.add_argument(
        "--completeness-random-iters",
        "--completeness_random_iters",
        type=int,
        default=20,
        help="Number of random K subsets for completeness tests.",
    )
    parser.add_argument(
        "--completeness-random-removal-probability",
        "--completeness_random_removal_probability",
        type=float,
        default=0.5,
        help="Probability of including each circuit component in random K.",
    )
    parser.add_argument(
        "--completeness-plot",
        "--completeness_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a completeness scatter plot when --run-completeness is set.",
    )
    parser.add_argument(
        "--completeness-figure",
        "--completeness_figure",
        type=Path,
        default=None,
        help=(
            "Completeness plot path. With multiple models, pass a directory; "
            "defaults inside each model results folder."
        ),
    )
    parser.add_argument(
        "--no-model-eval",
        action="store_true",
        help="Only recover circuit heads and write JSON; do not load the model.",
    )
    parser.add_argument(
        "--model-only",
        "--model_only",
        action="store_true",
        help=(
            "Evaluate only the full model on generated BOXES prompts; skip all "
            "path-patching score tensors, mean activations, and circuit metrics."
        ),
    )
    return parser


def main() -> None:
    global SYSTEM_PROMPT
    args = build_argparser().parse_args()
    SYSTEM_PROMPT = args.system_prompt
    binding_root = find_binding_root()
    if args.model_name and len(args.models) != 1:
        raise ValueError("--model-name can only be used with a single --models entry.")
    if args.model_only and args.no_model_eval:
        raise ValueError("--model-only cannot be combined with --no-model-eval.")
    if args.run_heldout and args.no_model_eval:
        raise ValueError("--run-heldout requires model evaluation.")
    if args.model_only and (args.run_minimality or args.run_completeness):
        raise ValueError(
            "--model-only cannot be combined with --run-minimality or --run-completeness."
        )
    if args.completeness_samples <= 0:
        raise ValueError("--completeness-samples must be positive.")
    if args.completeness_random_iters < 0:
        raise ValueError("--completeness-random-iters cannot be negative.")
    if not 0.0 <= args.completeness_random_removal_probability <= 1.0:
        raise ValueError(
            "--completeness-random-removal-probability must be in [0, 1]."
        )
    if (
        args.completeness_figure is not None
        and args.completeness_figure.suffix
        and len(args.models) > 1
    ):
        raise ValueError(
            "--completeness-figure as a file can only be used with one model; "
            "pass a directory when evaluating multiple models."
        )

    results = [
        evaluate_one_model(args, binding_root=binding_root, model_folder=model_folder)
        for model_folder in args.models
    ]
    payload: Any = results[0] if len(results) == 1 else results

    if args.output_json is not None:
        output_path = args.output_json
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        write_json(output_path, payload)
        print(f"Wrote {output_path}", file=sys.stderr)
    else:
        for result in results:
            output_path = Path(result["results_dir"]) / "circuit_eval_metrics.json"
            write_json(output_path, result)
            print(f"Wrote {output_path}", file=sys.stderr)
