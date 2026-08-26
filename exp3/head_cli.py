from __future__ import annotations

from exp3.head_config import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_SCORE_SITES,
    PROMPT_FORMATS,
    SITE_CHOICES,
    STAGE_CHOICES,
    parse_swap_pair,
)

from pathlib import Path
import argparse

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run interchange interventions by patching selected attention-head "
            "activations from paired counterfactual BOXES prompts into original prompts."
        )
    )
    parser.add_argument("--model_name", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--prompt-format",
        "--prompt_format",
        choices=PROMPT_FORMATS,
        default="raw",
        help=(
            "Prompt wrapper for generated BOXES data. Use the same value as "
            "run_path_patching.py and circuit_eval.py. `raw` preserves the old "
            "completion prompt. `auto` uses assistant-prefill chat prompts for "
            "instruct models, keeping `Answer:` in the assistant message so the "
            "next token is the object."
        ),
    )
    parser.add_argument("--vocab_tag", default="BOXES")
    parser.add_argument("--vocab_split", default="train")
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--score_file",
        default="value_fetcher.pt",
        help=(
            "Score tensor to select heads from. May be an absolute path, a path "
            "relative to cwd, or a filename inside --results_dir."
        ),
    )
    parser.add_argument(
        "--components-json",
        "--components_json",
        type=Path,
        default=None,
        help=(
            "Optional circuit_eval metrics JSON. When provided, patch the "
            "filtered component heads from that file at their recorded positions "
            "instead of selecting top-k heads from --score_file."
        ),
    )
    parser.add_argument(
        "--component-set",
        "--component_set",
        choices=("auto", "components", "minimality_kept"),
        default="auto",
        help=(
            "Which component list to read from --components_json. `components` "
            "uses the filtered circuit; `minimality_kept` uses pruning survivors "
            "when present; `auto` prefers pruning survivors if available."
        ),
    )
    parser.add_argument(
        "--component-hook-mode",
        "--component_hook_mode",
        choices=("role_aligned", "recorded"),
        default="role_aligned",
        help=(
            "When --components_json is provided, choose which activation hook "
            "to patch for component heads. `role_aligned` patches value_fetcher "
            "components at o_proj/result hooks and pointer/address roles at "
            "attention pattern hooks. `recorded` preserves the hook_name stored "
            "in the component JSON."
        ),
    )
    parser.add_argument(
        "--stages",
        "--roles",
        nargs="+",
        choices=STAGE_CHOICES,
        default=None,
        help=(
            "When --components_json is provided, keep only components with these "
            "stage/role names before patching. For example: --roles pos_transmitter."
        ),
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--largest",
        action="store_true",
        help="Select largest scores. Default selects smallest/most-negative scores.",
    )
    parser.add_argument(
        "--hook_component",
        choices=("v", "z", "result", "o_proj", "pattern"),
        default="o_proj",
        help=(
            "Attention-head activation to patch in score-file mode. "
            "`pattern` patches where heads look. `v` patches value vectors. "
            "`z` patches pre-W_O head outputs. `o_proj`/`result` patch "
            "post-W_O per-head outputs."
        ),
    )
    parser.add_argument("--num_entities", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--query_name", type=int, default=0)
    parser.add_argument(
        "--swap_pair",
        type=parse_swap_pair,
        default=(0,1),
        help=(
            "Original swap order as entity ids, e.g. '0,1' gives original "
            "`Swap X P` and counterfactual `Swap P X` under the same prompt id."
        ),
    )
    parser.add_argument("--prompt_id_start", type=int, default=0)
    parser.add_argument("--template_type", default="normal")
    parser.add_argument(
        "--counterfactual_mode",
        choices=("new_swap_objects", "same_objects", "value_fetcher_objects"),
        default="new_swap_objects",
        help=(
            "`new_swap_objects` reverses the swapped boxes and gives those boxes "
            "fresh object attributes in the counterfactual, while leaving non-swap "
            "boxes unchanged. `same_objects` keeps the older same-object behavior. "
            "`value_fetcher_objects` keeps the same box slots, swap order, and query "
            "in the counterfactual, but assigns fresh objects to the swapped slots."
        ),
    )
    parser.add_argument(
        "--sites",
        nargs="+",
        choices=SITE_CHOICES,
        default=None,
        help=(
            "Sites to evaluate. For --components_json, the default is the sites "
            "present in the component file; otherwise the default is "
            f"{', '.join(DEFAULT_SCORE_SITES)}. By default each site is run separately. "
            "`swap_query_box` is the X/query token; `swap_target_box` is the "
            "P/source token; `context_target_box` is the earlier context mention "
            "of the target/source box. Each site patches the selected site token."
        ),
    )
    parser.add_argument(
        "--combine_sites",
        action="store_true",
        help="Patch all requested --sites in one combined intervention.",
    )
    parser.add_argument(
        "--role-prompt-scores",
        "--role_prompt_scores",
        action="store_true",
        help=(
            "With --components_json, additionally save per-role scores for both "
            "prompt structures: payload_prompt uses --counterfactual_mode "
            "value_fetcher_objects, pointer_prompt uses new_swap_objects. "
            "Unless --sites is passed, each role uses its canonical score site: "
            "value_fetcher/pos_transmitter answer_colon, pos_detector "
            "question_query_box, struct_reader_swap_target swap_target_box."
        ),
    )
    parser.add_argument(
        "--restore-answer-colon",
        "--restore_answer_colon",
        dest="restore_answer_colon",
        action="store_true",
        default=False,
        help=(
            "For interventions whose requested sites do not include answer_colon, "
            "restore the patched heads at answer_colon from the original/base "
            "activation cache. Disabled by default."
        ),
    )
    parser.add_argument(
        "--no-restore-answer-colon",
        "--no_restore_answer_colon",
        dest="restore_answer_colon",
        action="store_false",
        help="Keep answer_colon activation restoration disabled.",
    )
    parser.add_argument(
        "--c-to-a-milk-experiment",
        "--c_to_a_milk_experiment",
        dest="c_to_a_milk_experiment",
        action="store_true",
        default=True,
        help=(
            "Also run the fixed C -> A diagnostic: source prompt `Box P cup. "
            "Box X milk. ... Swap P and X.` patched into base prompt `Box X "
            "milk. Box P cup. ... Swap X and P.`, measuring the increase of "
            "the `milk` logit on the base prompt."
        ),
    )
    parser.add_argument(
        "--no-c-to-a-milk-experiment",
        "--no_c_to_a_milk_experiment",
        dest="c_to_a_milk_experiment",
        action="store_false",
        help="Skip the fixed C -> A milk-logit diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser
