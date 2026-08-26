from __future__ import annotations

from exp3.head_config import (
    C_TO_A_MILK_ANSWER,
    C_TO_A_MILK_BASE_PROMPT,
    C_TO_A_MILK_MEASURED_OBJECT,
    C_TO_A_MILK_SOURCE_PROMPT,
    DEFAULT_SCORE_SITES,
    PATCH_POINT_INTERPRETATION,
    ROLE_DEFAULT_SCORE_SITES,
    ROLE_KIND_BY_STAGE,
    ROLE_PROMPT_SCORE_MODES,
    SCRIPT_DIR,
    STAGE_CHOICES,
    all_hook_names,
    component_count_by_site,
    component_count_by_stage,
    component_heads_by_site_hook,
    default_output_path_for_score,
    filter_components_by_stage,
    load_component_dicts,
    normalize_device,
    normalize_hook_component,
    resolve_score_path,
    score_heads_by_site_hook,
    select_heads,
    select_site_heads,
    set_seed,
    unique_layer_heads,
)

from exp3.head_cli import (
    build_argparser,
)

from exp3.head_data import (
    build_c_to_a_milk_dataset,
    build_paired_dataset,
    resolve_model_name,
    resolve_prompt_format,
)

from exp3.role_scores import (
    ensure_attn_result_enabled,
    evaluate_role_prompt_scores,
    evaluation_plan_for_sites,
    preview_dataset,
)

from exp3.head_interventions import (
    evaluate_interchange,
    patch_offsets_for_site,
)

from exp3.head_metrics import (
    format_optional_float,
)

from collections import defaultdict
from datasets.api import load_vocab
from exp2.pp_utils import compute_topk_components, get_model_and_tokenizer
from torch.utils.data import DataLoader, Dataset
import json
import sys

def main() -> None:
    args = build_argparser().parse_args()
    args.results_dir = args.results_dir.resolve()
    resolved_model_name = resolve_model_name(args.model_name)
    resolved_prompt_format = resolve_prompt_format(
        resolved_model_name,
        args.prompt_format,
    )
    hook_component = normalize_hook_component(args.hook_component)
    score_path: Optional[Path] = None
    components_path: Optional[Path] = None
    components: Optional[List[Dict[str, object]]] = None
    unfiltered_components: Optional[List[Dict[str, object]]] = None
    component_set_used: Optional[str] = None
    component_payload: Dict[str, object] = {}

    if args.components_json is not None:
        components_path = args.components_json.expanduser().resolve()
        if not components_path.exists():
            raise FileNotFoundError(f"Could not find components JSON: {components_path}")
        unfiltered_components, component_set_used, component_payload = load_component_dicts(
            components_path,
            args.component_set,
        )
        components = filter_components_by_stage(unfiltered_components, args.stages)
        if not components:
            available = ", ".join(sorted(component_count_by_stage(unfiltered_components)))
            requested = ", ".join(args.stages or ())
            raise ValueError(
                f"No components remain after --roles/--stages filter ({requested}). "
                f"Available stages in {components_path}: {available}."
            )
        all_heads_by_site_hook = component_heads_by_site_hook(
            components,
            component_hook_mode=args.component_hook_mode,
        )
        active_sites, heads_by_site_hook, _ = select_site_heads(
            all_heads_by_site_hook,
            args.sites,
        )
    else:
        if args.role_prompt_scores:
            raise ValueError("--role-prompt-scores requires --components_json.")
        if args.stages is not None:
            raise ValueError(
                "--roles/--stages requires --components_json because raw score "
                "tensors do not carry component role metadata. For score-file "
                "mode, choose a role by setting --score_file, e.g. "
                "--score_file pos_transmitter.pt."
            )
        score_path = resolve_score_path(args.results_dir, args.score_file)
        if not score_path.exists():
            raise FileNotFoundError(f"Could not find score file: {score_path}")
        active_sites = list(args.sites) if args.sites is not None else list(DEFAULT_SCORE_SITES)
        heads = select_heads(score_path, args.top_k, args.largest)
        heads_by_layer: Dict[int, List[int]] = defaultdict(list)
        for layer_idx, head_idx in heads:
            heads_by_layer[layer_idx].append(head_idx)
        heads_by_site_hook = score_heads_by_site_hook(
            heads_by_layer=heads_by_layer,
            hook_component=hook_component,
            sites=active_sites,
        )

    hook_names = all_hook_names(heads_by_site_hook)
    heads = unique_layer_heads(heads_by_site_hook)

    selected_device = normalize_device(args.device)
    set_seed(args.seed)

    print(f"Model: {resolved_model_name}", file=sys.stderr)
    if args.model_name != resolved_model_name:
        print(f"Requested model alias: {args.model_name}", file=sys.stderr)
    print(f"Prompt format: {resolved_prompt_format}", file=sys.stderr)
    if args.prompt_format != resolved_prompt_format:
        print(f"Requested prompt format: {args.prompt_format}", file=sys.stderr)
    if components_path is not None:
        print(f"Components JSON: {components_path}", file=sys.stderr)
        print(f"Component set: {component_set_used}", file=sys.stderr)
        print(f"Component hook mode: {args.component_hook_mode}", file=sys.stderr)
        if args.stages is not None:
            print(f"Stage/role filter: {', '.join(args.stages)}", file=sys.stderr)
    else:
        print(f"Score file: {score_path}", file=sys.stderr)
        print(f"Hook component: {args.hook_component} ({hook_component})", file=sys.stderr)
    print(
        "Restore answer_colon for non-answer_colon interventions: "
        f"{args.restore_answer_colon}",
        file=sys.stderr,
    )
    print(f"Device: {selected_device}", file=sys.stderr)

    model, tokenizer = get_model_and_tokenizer(resolved_model_name)
    ensure_attn_result_enabled(model, hook_names)

    vocab = load_vocab(
        args.vocab_tag,
        resolved_model_name,
        split=args.vocab_split,
        tokenizer=tokenizer,
    )
    dataset = build_paired_dataset(
        tokenizer=tokenizer,
        model_name=resolved_model_name,
        vocab=vocab,
        num_entities=args.num_entities,
        num_samples=args.num_samples,
        query_name=args.query_name,
        prompt_id_start=args.prompt_id_start,
        swap_pair=args.swap_pair,
        template_type=args.template_type,
        counterfactual_mode=args.counterfactual_mode,
        prompt_format=resolved_prompt_format,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Selected {len(heads)} unique heads:", heads, file=sys.stderr)
    print("Component count by site:", component_count_by_site(heads_by_site_hook), file=sys.stderr)
    print("Dataset preview:", file=sys.stderr)
    preview = preview_dataset(dataset, tokenizer, active_sites)
    print(json.dumps(preview, indent=2), file=sys.stderr)

    eval_plan = evaluation_plan_for_sites(
        active_sites=active_sites,
        heads_by_site_hook=heads_by_site_hook,
        combine_sites=args.combine_sites,
    )

    results = {}
    for name, sites, site_heads_by_hook in eval_plan:
        results[name] = evaluate_interchange(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            hook_names=hook_names,
            heads_by_site_hook=site_heads_by_hook,
            sites=sites,
            device=selected_device,
            restore_answer_colon=args.restore_answer_colon,
        )
        patched = results[name]["patched"]
        delta = results[name]["deltas"]
        print(
            f"{name}: patched acc={format_optional_float(patched['accuracy'])} "
            f"candidate_acc={format_optional_float(patched['candidate_accuracy'])} "
            "delta_label_logit="
            f"{format_optional_float(delta['patched_minus_original_mean_label_logit'])} "
            "delta_query_obj_logit="
            f"{format_optional_float(delta['patched_minus_original_mean_query_object_logit'])} "
            "delta_source_minus_original_margin="
            f"{format_optional_float(delta['patched_minus_original_mean_source_label_minus_original_label_logit_margin'])} "
            "delta_query_minus_target_margin="
            f"{format_optional_float(delta['patched_minus_original_mean_query_label_minus_target_label_logit_margin'])}",
            file=sys.stderr,
        )

    custom_experiments = {}
    if args.c_to_a_milk_experiment:
        custom_dataset = build_c_to_a_milk_dataset(
            tokenizer=tokenizer,
            model_name=resolved_model_name,
            prompt_format=resolved_prompt_format,
        )
        custom_dataloader = DataLoader(
            custom_dataset,
            batch_size=min(args.batch_size, len(custom_dataset)),
            shuffle=False,
        )
        custom_preview = preview_dataset(custom_dataset, tokenizer, active_sites)
        print("C -> A milk-logit experiment preview:", file=sys.stderr)
        print(json.dumps(custom_preview, indent=2), file=sys.stderr)
        custom_results = {}
        for name, sites, site_heads_by_hook in eval_plan:
            custom_results[name] = evaluate_interchange(
                model=model,
                tokenizer=tokenizer,
                dataloader=custom_dataloader,
                hook_names=hook_names,
                heads_by_site_hook=site_heads_by_hook,
                sites=sites,
                device=selected_device,
                restore_answer_colon=args.restore_answer_colon,
            )
            delta = custom_results[name]["deltas"]
            print(
                f"c_to_a_milk/{name}: delta_milk_logit="
                f"{format_optional_float(delta['patched_minus_original_mean_query_object_logit'])}",
                file=sys.stderr,
            )
        custom_experiments["c_to_a_milk"] = {
            "direction": "C_to_A",
            "base_prompt_name": "A",
            "source_prompt_name": "C",
            "base_prompt": C_TO_A_MILK_BASE_PROMPT,
            "source_prompt": C_TO_A_MILK_SOURCE_PROMPT,
            "answer": C_TO_A_MILK_ANSWER,
            "measured_object": C_TO_A_MILK_MEASURED_OBJECT,
            "measurement_delta_key": (
                "patched_minus_original_mean_query_object_logit"
            ),
            "preview": custom_preview,
            "results": custom_results,
        }

    role_prompt_scores = None
    if args.role_prompt_scores:
        if components is None:
            raise ValueError("--role-prompt-scores requires component data.")
        component_stage_counts = component_count_by_stage(components)
        roles_for_prompt_scores = (
            list(args.stages)
            if args.stages is not None
            else [
                stage
                for stage in STAGE_CHOICES
                if stage in component_stage_counts
            ]
        )
        print(
            "Evaluating per-role payload/pointer prompt scores for roles: "
            + ", ".join(roles_for_prompt_scores),
            file=sys.stderr,
        )
        role_prompt_scores = evaluate_role_prompt_scores(
            model=model,
            tokenizer=tokenizer,
            vocab=vocab,
            components=components,
            roles=roles_for_prompt_scores,
            requested_sites=args.sites,
            component_hook_mode=args.component_hook_mode,
            model_name=resolved_model_name,
            num_entities=args.num_entities,
            num_samples=args.num_samples,
            query_name=args.query_name,
            prompt_id_start=args.prompt_id_start,
            swap_pair=args.swap_pair,
            template_type=args.template_type,
            prompt_format=resolved_prompt_format,
            batch_size=args.batch_size,
            combine_sites=args.combine_sites,
            device=selected_device,
            restore_answer_colon=args.restore_answer_colon,
            include_c_to_a_milk_experiment=args.c_to_a_milk_experiment,
        )

    payload = {
        "model_name": resolved_model_name,
        "requested_model_name": args.model_name,
        "vocab_tag": args.vocab_tag,
        "vocab_split": args.vocab_split,
        "results_dir": str(args.results_dir),
        "score_file": str(score_path) if score_path is not None else None,
        "components_json": str(components_path) if components_path is not None else None,
        "component_set": component_set_used,
        "component_hook_mode": (
            args.component_hook_mode if components_path is not None else None
        ),
        "stage_filter": list(args.stages) if args.stages is not None else None,
        "role_kind_by_stage": ROLE_KIND_BY_STAGE,
        "patch_point_interpretation": PATCH_POINT_INTERPRETATION,
        "role_prompt_score_modes": ROLE_PROMPT_SCORE_MODES,
        "role_default_score_sites": ROLE_DEFAULT_SCORE_SITES,
        "role_prompt_scores_enabled": args.role_prompt_scores,
        "patch_offsets_by_site": {
            site: list(patch_offsets_for_site(site)) for site in active_sites
        },
        "restore_answer_colon": args.restore_answer_colon,
        "restore_answer_colon_applied_by_result": {
            name: bool(args.restore_answer_colon and "answer_colon" not in sites)
            for name, sites, _ in eval_plan
        },
        "source_component_count": (
            len(unfiltered_components) if unfiltered_components is not None else None
        ),
        "source_component_count_after_stage_filter": (
            len(components) if components is not None else None
        ),
        "source_component_count_by_stage": (
            component_count_by_stage(unfiltered_components)
            if unfiltered_components is not None
            else None
        ),
        "filtered_component_count_by_stage": (
            component_count_by_stage(components) if components is not None else None
        ),
        "source_component_count_by_position": component_payload.get(
            "component_count_by_position"
        ),
        "component_count_by_site": component_count_by_site(heads_by_site_hook),
        "heads_by_site_hook": heads_by_site_hook,
        "hook_names": hook_names,
        "top_k": args.top_k,
        "largest": args.largest,
        "hook_component": args.hook_component,
        "source_resolved_hook_component": (
            component_payload.get("resolved_hook_component")
            if components_path is not None
            else None
        ),
        "resolved_hook_component": (
            args.component_hook_mode
            if components_path is not None
            else hook_component
        ),
        "heads": [[layer, head] for layer, head in heads],
        "num_entities": args.num_entities,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "query_name": args.query_name,
        "swap_pair": list(args.swap_pair),
        "prompt_id_start": args.prompt_id_start,
        "template_type": args.template_type,
        "counterfactual_mode": args.counterfactual_mode,
        "prompt_format": resolved_prompt_format,
        "requested_prompt_format": args.prompt_format,
        "sites": list(active_sites),
        "combine_sites": args.combine_sites,
        "preview": preview,
        "results": results,
        "custom_experiments": custom_experiments,
        "role_prompt_scores": role_prompt_scores,
    }

    output_path = (
        default_output_path_for_score(
            components_path.stem if components_path is not None else args.score_file
        )
        if args.output_json is None
        else args.output_json
    )
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}", file=sys.stderr)
