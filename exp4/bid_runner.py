from __future__ import annotations

from exp4.bid_data import (
    BASE_OBJECT_IDS_BY_ENTITY,
    CONTROL_CONDITIONS,
    NUM_ENTITIES,
    SCRIPT_DIR,
    build_binding_id_dataset,
    k_hook_name,
    load_group_b_components,
    normalize_device,
    normalize_position_mode,
    parse_csv,
    parse_int_tuple,
    pattern_hook_name,
    print_sanity_table,
    q_hook_name,
    query_entity_for_source_slot,
    sanity_table,
    select_wrong_heads,
    source_slot_for_query,
)

from exp4.bid_evaluation import (
    build_condition_specs,
    evaluate_conditions,
)

from exp4.bid_directions import (
    delta_between,
    estimate_binding_id_directions,
    hook_names_for_heads,
    matched_norm_directions,
    negate_deltas,
    random_matched_deltas,
)

from exp4.bid_cli import (
    parse_args,
    serialize_delta_norms,
    serialize_heads,
    serialize_kv_head_mapping,
)

from datasets.api import load_vocab
from exp2.pp_utils import get_model_and_tokenizer
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
from torch.utils.data import DataLoader
import json
import sys

def main() -> None:
    args = parse_args()
    components_path = args.components_json.expanduser().resolve()
    if not components_path.exists():
        raise FileNotFoundError(components_path)

    swap_pair = parse_int_tuple(args.swap_pair, expected_len=2)
    if len(set(swap_pair)) != 2:
        raise ValueError(f"--swap-pair entries must be different, got {swap_pair}.")
    if not (0 <= args.query_entity < NUM_ENTITIES):
        raise ValueError(f"--query-entity must be in [0, {NUM_ENTITIES}), got {args.query_entity}.")
    if not (0 <= args.target_source_entity < NUM_ENTITIES):
        raise ValueError(
            f"--target-source-entity must be in [0, {NUM_ENTITIES}), "
            f"got {args.target_source_entity}."
        )

    resolved_model_name = resolve_model_name(args.model_name)
    resolved_prompt_format = resolve_prompt_format(
        resolved_model_name,
        args.prompt_format,
    )
    selected_device = normalize_device(args.device)
    set_seed(args.seed)

    group_b_components, b_heads, component_payload, kept_components, nonmatching = (
        load_group_b_components(
            components_path,
            expected_stage=args.group_b_stage,
            expected_position_modes=parse_csv(args.group_b_position_modes),
        )
    )
    sanity_rows = sanity_table(group_b_components)
    print_sanity_table(sanity_rows)

    original_source_entity = source_slot_for_query(
        args.query_entity,
        include_swap=bool(args.include_swap),
        swap_pair=swap_pair,
    )
    target_source_entity = int(args.target_source_entity)
    if original_source_entity == target_source_entity:
        raise ValueError(
            "Target source entity equals the original source entity. Choose a "
            "different --target-source-entity for an intervention."
        )
    distractor_entity = next(
        entity
        for entity in range(NUM_ENTITIES)
        if entity not in (original_source_entity, target_source_entity)
    )

    print(f"Model: {resolved_model_name}", file=sys.stderr)
    print(f"Prompt format: {resolved_prompt_format}", file=sys.stderr)
    print(f"Components JSON: {components_path}", file=sys.stderr)
    print(
        "Loaded minimality_pruning.kept_components: "
        f"{len(kept_components)}; selected group-B entries: {len(group_b_components)}",
        file=sys.stderr,
    )
    if nonmatching:
        print(
            "Kept components outside the configured group-B stage/site were "
            f"loaded for provenance but not used as group B: {len(nonmatching)}",
            file=sys.stderr,
        )
    print(f"Selected B heads: {serialize_heads(b_heads)}", file=sys.stderr)
    print(
        f"Query entity={args.query_entity}; original source={original_source_entity}; "
        f"target source={target_source_entity}; distractor={distractor_entity}",
        file=sys.stderr,
    )
    print(f"Source position kinds: {args.source_position_kinds}", file=sys.stderr)
    print(f"Conditions: {args.conditions}", file=sys.stderr)
    print(f"Device: {selected_device}", file=sys.stderr)

    model, tokenizer = get_model_and_tokenizer(resolved_model_name)
    logical_n_heads = int(getattr(model.cfg, "n_heads", 0))
    if logical_n_heads <= 0:
        logical_n_heads = max(int(head) for _, head in b_heads) + 1
    kv_n_heads = int(
        getattr(
            model.cfg,
            "n_key_value_heads",
            getattr(model.cfg, "n_kv_heads", logical_n_heads),
        )
    )
    vocab = load_vocab(
        args.vocab_tag,
        resolved_model_name,
        split=args.vocab_split,
        tokenizer=tokenizer,
    )

    direction_dataloaders = {
        source_slot: DataLoader(
            build_binding_id_dataset(
                tokenizer=tokenizer,
                model_name=resolved_model_name,
                vocab=vocab,
                query_entity=query_entity_for_source_slot(
                    source_slot,
                    include_swap=bool(args.include_swap),
                    swap_pair=swap_pair,
                ),
                num_samples=args.direction_num_samples,
                prompt_id_start=args.direction_prompt_id_start,
                prompt_format=resolved_prompt_format,
                include_swap=bool(args.include_swap),
                swap_pair=swap_pair,
                template_type=args.template_type,
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        for source_slot in range(NUM_ENTITIES)
    }

    b_direction_payload = estimate_binding_id_directions(
        model=model,
        dataloaders_by_source_slot=direction_dataloaders,
        heads=b_heads,
        source_position_kinds=args.source_position_kinds,
        device=selected_device,
        label="B heads",
    )
    b_q_original_to_target = delta_between(
        b_direction_payload["q_means"],
        original_source_entity,
        target_source_entity,
    )
    b_k_target_to_original = delta_between(
        b_direction_payload["k_means"],
        target_source_entity,
        original_source_entity,
    )
    b_k_original_to_target = delta_between(
        b_direction_payload["k_means"],
        original_source_entity,
        target_source_entity,
    )
    b_q_current = b_direction_payload["q_means"][int(original_source_entity)]
    b_k_question_query_box_current = b_direction_payload["k_position_means"][
        "query_box_positions"
    ][int(original_source_entity)]
    q_select_question_query_box_more = matched_norm_directions(
        b_k_question_query_box_current,
        reference=b_q_original_to_target,
    )
    q_select_question_query_box_less = negate_deltas(q_select_question_query_box_more)
    k_select_question_query_box_more = matched_norm_directions(
        b_q_current,
        reference=b_k_original_to_target,
    )
    k_select_question_query_box_less = negate_deltas(k_select_question_query_box_more)

    random_q_original_to_target = random_matched_deltas(
        b_q_original_to_target,
        seed=args.seed + 1001,
    )
    random_k_target_to_original = random_matched_deltas(
        b_k_target_to_original,
        seed=args.seed + 1002,
    )
    random_k_original_to_target = random_matched_deltas(
        b_k_original_to_target,
        seed=args.seed + 1003,
    )

    wrong_heads: List[Head] = []
    wrong_q_original_to_target = None
    wrong_k_target_to_original = None
    wrong_k_original_to_target = None
    if any(condition.startswith("wrong_head") for condition in args.conditions):
        wrong_heads = select_wrong_heads(model, b_heads)
        wrong_direction_payload = estimate_binding_id_directions(
            model=model,
            dataloaders_by_source_slot=direction_dataloaders,
            heads=wrong_heads,
            source_position_kinds=args.source_position_kinds,
            device=selected_device,
            label="wrong heads",
        )
        wrong_q_original_to_target = delta_between(
            wrong_direction_payload["q_means"],
            original_source_entity,
            target_source_entity,
        )
        wrong_k_target_to_original = delta_between(
            wrong_direction_payload["k_means"],
            target_source_entity,
            original_source_entity,
        )
        wrong_k_original_to_target = delta_between(
            wrong_direction_payload["k_means"],
            original_source_entity,
            target_source_entity,
        )
    else:
        wrong_direction_payload = None

    eval_dataloader = DataLoader(
        build_binding_id_dataset(
            tokenizer=tokenizer,
            model_name=resolved_model_name,
            vocab=vocab,
            query_entity=args.query_entity,
            num_samples=args.num_samples,
            prompt_id_start=args.prompt_id_start,
            prompt_format=resolved_prompt_format,
            include_swap=bool(args.include_swap),
            swap_pair=swap_pair,
            template_type=args.template_type,
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    condition_specs = build_condition_specs(
        requested_conditions=args.conditions,
        b_heads=b_heads,
        wrong_heads=wrong_heads,
        original_entity=original_source_entity,
        target_entity=target_source_entity,
        distractor_entity=distractor_entity,
        alpha=float(args.alpha),
        b_q_original_to_target=b_q_original_to_target,
        b_k_target_to_original=b_k_target_to_original,
        b_k_original_to_target=b_k_original_to_target,
        q_select_question_query_box_more=q_select_question_query_box_more,
        q_select_question_query_box_less=q_select_question_query_box_less,
        k_select_question_query_box_more=k_select_question_query_box_more,
        k_select_question_query_box_less=k_select_question_query_box_less,
        random_q_original_to_target=random_q_original_to_target,
        random_k_target_to_original=random_k_target_to_original,
        random_k_original_to_target=random_k_original_to_target,
        wrong_q_original_to_target=wrong_q_original_to_target,
        wrong_k_target_to_original=wrong_k_target_to_original,
        wrong_k_original_to_target=wrong_k_original_to_target,
    )

    condition_results = evaluate_conditions(
        model=model,
        tokenizer=tokenizer,
        dataloader=eval_dataloader,
        condition_specs=condition_specs,
        b_heads=b_heads,
        source_position_kinds=args.source_position_kinds,
        original_entity=original_source_entity,
        target_entity=target_source_entity,
        device=selected_device,
        include_per_sample=not args.no_per_sample,
        attention_eps=float(args.attention_eps),
    )

    control_results = {
        name: result
        for name, result in condition_results.items()
        if name in CONTROL_CONDITIONS
    }
    main_results = {
        name: result
        for name, result in condition_results.items()
        if name not in CONTROL_CONDITIONS
    }

    payload = {
        "model_name": resolved_model_name,
        "requested_model_name": args.model_name,
        "prompt_format": resolved_prompt_format,
        "requested_prompt_format": args.prompt_format,
        "components_json": str(components_path),
        "source_circuit_model": component_payload.get("model_name"),
        "loaded_minimality_pruning_kept_components": kept_components,
        "selected_b_components": sanity_rows,
        "selected_b_heads": serialize_heads(b_heads),
        "non_group_b_kept_components": sanity_table(nonmatching),
        "group_b_selection": {
            "component_set": "minimality_pruning.kept_components",
            "expected_stage": args.group_b_stage,
            "expected_position_modes": list(parse_csv(args.group_b_position_modes)),
            "normalized_expected_position_modes": sorted(
                {
                    normalize_position_mode(position_mode)
                    for position_mode in parse_csv(args.group_b_position_modes)
                }
            ),
            "recorded_hooks_are_discovery_hooks_only": True,
            "intervention_hooks": {
                "q": hook_names_for_heads(b_heads, q_hook_name),
                "k": hook_names_for_heads(b_heads, k_hook_name),
                "pattern_readout": hook_names_for_heads(b_heads, pattern_hook_name),
            },
            "gqa_head_mapping": {
                "logical_attention_heads": int(logical_n_heads),
                "key_value_heads": int(kv_n_heads),
                "selected_b_attention_head_to_kv_head": serialize_kv_head_mapping(
                    b_heads,
                    logical_n_heads=logical_n_heads,
                    kv_n_heads=kv_n_heads,
                ),
                "wrong_attention_head_to_kv_head": serialize_kv_head_mapping(
                    wrong_heads,
                    logical_n_heads=logical_n_heads,
                    kv_n_heads=kv_n_heads,
                )
                if wrong_heads
                else {},
            },
        },
        "direction_estimation": {
            "spaces": "B-local hook_q and hook_k head dimensions",
            "prompt_id_start": args.direction_prompt_id_start,
            "num_samples": args.direction_num_samples,
            "batch_size": args.batch_size,
            "source_position_kinds_for_k": list(args.source_position_kinds),
            "carrier_position_fields_for_k": ["query_box_positions"],
            "source_slot_to_query_entity": {
                str(source_slot): query_entity_for_source_slot(
                    source_slot,
                    include_swap=bool(args.include_swap),
                    swap_pair=swap_pair,
                )
                for source_slot in range(NUM_ENTITIES)
            },
            "b_q_counts": b_direction_payload["q_counts"],
            "b_k_counts": b_direction_payload["k_counts"],
            "b_k_carrier_counts": b_direction_payload["k_position_counts"],
            "delta_norms": {
                "b_q_original_to_target": serialize_delta_norms(
                    b_q_original_to_target
                ),
                "b_k_target_to_original": serialize_delta_norms(
                    b_k_target_to_original
                ),
                "b_k_original_to_target": serialize_delta_norms(
                    b_k_original_to_target
                ),
                "q_select_question_query_box_more": serialize_delta_norms(
                    q_select_question_query_box_more
                ),
                "q_select_question_query_box_less": serialize_delta_norms(
                    q_select_question_query_box_less
                ),
                "k_select_question_query_box_more": serialize_delta_norms(
                    k_select_question_query_box_more
                ),
                "k_select_question_query_box_less": serialize_delta_norms(
                    k_select_question_query_box_less
                ),
                "random_q_original_to_target": serialize_delta_norms(
                    random_q_original_to_target
                ),
                "random_k_target_to_original": serialize_delta_norms(
                    random_k_target_to_original
                ),
                "random_k_original_to_target": serialize_delta_norms(
                    random_k_original_to_target
                ),
            },
            "wrong_head_counts": (
                {
                    "q": wrong_direction_payload["q_counts"],
                    "k": wrong_direction_payload["k_counts"],
                    "k_carriers": wrong_direction_payload["k_position_counts"],
                }
                if wrong_direction_payload is not None
                else None
            ),
        },
        "intervention_setup": {
            "alpha": float(args.alpha),
            "query_entity": int(args.query_entity),
            "include_swap": bool(args.include_swap),
            "swap_pair": list(swap_pair) if args.include_swap else None,
            "original_source_entity": int(original_source_entity),
            "target_source_entity": int(target_source_entity),
            "distractor_entity_for_wrong_position_control": int(distractor_entity),
            "q_intervention_target": "base_answer_colon_positions",
            "q_wrong_position_target": "query_box_positions",
            "k_intervention_source_positions": list(args.source_position_kinds),
            "source_side_k_rule": {
                "target_source_entity": "delta_k_target_to_original",
                "original_source_entity": "delta_k_original_to_target",
            },
            "carrier_selection_rule": {
                "question_query_box_position_field": "query_box_positions",
                "competing_context_carrier_position_field": "box_token_positions",
                "q_side": (
                    "Q(answer_colon) receives +/- matched-norm mean "
                    "K(question_query_box)."
                ),
                "k_side": (
                    "K(question_query_box) and/or context box carriers receive "
                    "+/- matched-norm mean Q(answer_colon)."
                ),
            },
            "both_side_rule": (
                "Q is shifted original->target while K swaps original and target, "
                "so the shifted query should again match the original source slot."
            ),
        },
        "num_entities": NUM_ENTITIES,
        "base_object_ids_by_entity": BASE_OBJECT_IDS_BY_ENTITY,
        "vocab_tag": args.vocab_tag,
        "vocab_split": args.vocab_split,
        "prompt_id_start": args.prompt_id_start,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "conditions_requested": list(args.conditions),
        "wrong_head_control_heads": serialize_heads(wrong_heads),
        "results": main_results,
        "control_results": control_results,
    }

    rendered = json.dumps(payload, indent=2, allow_nan=True)
    if args.output_json is not None:
        output_path = args.output_json.expanduser()
        if not output_path.is_absolute():
            output_path = SCRIPT_DIR / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {output_path}", file=sys.stderr)
    print(rendered)
