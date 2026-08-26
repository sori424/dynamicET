from __future__ import annotations

from exp4.bid_data import (
    CONDITION_CHOICES,
    DEFAULT_COMPONENTS_JSON,
    DEFAULT_CONDITIONS,
    key_value_head_for_attention_head,
    parse_conditions,
    parse_source_position_kinds,
)

from exp3.head_role import (  # noqa: E402
    PROMPT_FORMATS,
    DictTensorDataset,
    decode_prompts_from_tokens,
    encode_prompt_texts,
    first_answer_token_id,
    format_prompt_for_model,
    last_token_indices,
    move_batch_to_device,
    resolve_model_name,
    resolve_prompt_format,
    set_seed,
)
from pathlib import Path
import argparse
import torch

def serialize_heads(heads: Sequence[Head]) -> List[Dict[str, int]]:
    return [{"layer": int(layer), "head": int(head)} for layer, head in heads]


def serialize_delta_norms(deltas: Mapping[Head, torch.Tensor]) -> Dict[str, float]:
    return {
        f"L{layer}H{head}": float(torch.linalg.vector_norm(delta.float()).item())
        for (layer, head), delta in sorted(deltas.items())
    }


def serialize_kv_head_mapping(
    heads: Sequence[Head],
    *,
    logical_n_heads: int,
    kv_n_heads: int,
) -> Dict[str, int]:
    return {
        f"L{layer}H{head}": key_value_head_for_attention_head(
            int(head),
            logical_n_heads=logical_n_heads,
            activation_n_heads=kv_n_heads,
        )
        for layer, head in sorted(heads)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run genuine binding-ID Q/K interventions on the kept group-B "
            "attention heads from minimality_pruning.kept_components."
        )
    )
    parser.add_argument("--model-name", "--model_name", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--components-json",
        "--components_json",
        type=Path,
        default=DEFAULT_COMPONENTS_JSON,
        help="Circuit JSON containing minimality_pruning.kept_components.",
    )
    parser.add_argument(
        "--group-b-stage",
        "--group_b_stage",
        default="pos_transmitter",
        help="Expected kept-component stage for group B.",
    )
    parser.add_argument(
        "--group-b-position-modes",
        "--group_b_position_modes",
        default="answer_colon,last_token",
        help="Comma-separated accepted position modes for group B.",
    )
    parser.add_argument("--vocab-tag", "--vocab_tag", default="BOXES")
    parser.add_argument("--vocab-split", "--vocab_split", default="train")
    parser.add_argument(
        "--prompt-format",
        "--prompt_format",
        choices=PROMPT_FORMATS,
        default="auto",
    )
    parser.add_argument("--template-type", "--template_type", default="normal")
    parser.add_argument("--prompt-id-start", "--prompt_id_start", type=int, default=0)
    parser.add_argument("--num-samples", "--num_samples", type=int, default=100)
    parser.add_argument(
        "--direction-prompt-id-start",
        "--direction_prompt_id_start",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--direction-num-samples",
        "--direction_num_samples",
        type=int,
        default=100,
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--source-position-kinds",
        "--source_position_kinds",
        type=parse_source_position_kinds,
        default=parse_source_position_kinds("box,object"),
        help="Comma-separated K source/readout token kinds: box, object, or box,object.",
    )
    parser.add_argument(
        "--conditions",
        type=parse_conditions,
        default=DEFAULT_CONDITIONS,
        help=f"Comma-separated conditions. Choices: {', '.join(CONDITION_CHOICES)}.",
    )
    parser.add_argument("--query-entity", "--query_entity", type=int, default=0)
    parser.add_argument(
        "--target-source-entity",
        "--target_source_entity",
        type=int,
        default=0,
        help="Source slot/entity to shift attention toward.",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--include-swap",
        "--include_swap",
        action="store_true",
        default=True,
        help="Include the BOXES swap sentence. Enabled by default.",
    )
    parser.add_argument(
        "--no-include-swap",
        "--no_include_swap",
        dest="include_swap",
        action="store_false",
        help="Disable the BOXES swap sentence.",
    )
    parser.add_argument(
        "--swap-pair",
        "--swap_pair",
        default="0,1",
        help="Swap sentence pair. Default gives Query 0 original source slot 1.",
    )
    parser.add_argument(
        "--attention-eps",
        "--attention_eps",
        type=float,
        default=1e-9,
    )
    parser.add_argument("--output-json", "--output_json", type=Path, default=None)
    parser.add_argument(
        "--no-per-sample",
        "--no_per_sample",
        action="store_true",
        help="Omit optional per-sample results from the JSON payload.",
    )
    return parser.parse_args()
