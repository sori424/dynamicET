from __future__ import annotations

from exp4.bid_data import (
    NUM_ENTITIES,
    k_hook_name,
    pattern_hook_name,
    q_hook_name,
)

from exp4.bid_attention import (
    attention_masses_for_heads,
    carrier_labels,
    carrier_masses_for_heads,
    finalize_attention_stats,
    finalize_carrier_stats,
    finalize_logit_stats,
    init_attention_stats,
    init_carrier_stats,
    init_logit_stats,
    mean_carrier_mass,
    mean_mass_by_entity,
    tensor_carrier_row_to_floats,
    tensor_row_to_floats,
    update_attention_stats,
    update_carrier_stats,
    update_logit_stats,
)

from exp4.bid_directions import (
    cache_forward,
    hook_names_for_heads,
    make_intervention_hooks,
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
from tqdm import tqdm
import torch

def evaluate_conditions(
    *,
    model,
    tokenizer,
    dataloader: DataLoader,
    condition_specs: Sequence[Mapping[str, object]],
    b_heads: Sequence[Head],
    source_position_kinds: Sequence[str],
    original_entity: int,
    target_entity: int,
    device: torch.device,
    include_per_sample: bool,
    attention_eps: float,
) -> Dict[str, Dict[str, object]]:
    pattern_hook_names = hook_names_for_heads(b_heads, pattern_hook_name)
    logical_n_heads = int(getattr(model.cfg, "n_heads", 0)) or max(
        int(head) for _, head in b_heads
    ) + 1
    results: Dict[str, Dict[str, object]] = {}
    working_stats = {
        str(spec["name"]): {
            "attention": init_attention_stats(b_heads),
            "carrier_attention": init_carrier_stats(b_heads),
            "logits": init_logit_stats(),
            "per_sample": [],
        }
        for spec in condition_specs
    }

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate binding-ID Q/K interventions"):
            batch = move_batch_to_device(batch, device)
            before_logits, before_cache = cache_forward(
                model,
                batch["base_tokens"],
                cache_hook_names=pattern_hook_names,
            )
            before_masses = attention_masses_for_heads(
                cache=before_cache,
                heads=b_heads,
                batch=batch,
                source_position_kinds=source_position_kinds,
            )
            before_carrier_masses = carrier_masses_for_heads(
                cache=before_cache,
                heads=b_heads,
                batch=batch,
            )

            for spec in condition_specs:
                name = str(spec["name"])
                q_deltas = spec.get("q_deltas")
                k_entity_deltas = spec.get("k_entity_deltas")
                k_position_field_deltas = spec.get("k_position_field_deltas")
                if (
                    q_deltas is None
                    and k_entity_deltas is None
                    and k_position_field_deltas is None
                ):
                    after_logits = before_logits
                    after_cache = before_cache
                else:
                    hooks = make_intervention_hooks(
                        heads=spec["intervention_heads"],
                        q_deltas=q_deltas,
                        q_position_field=spec.get("q_position_field"),
                        k_entity_deltas=k_entity_deltas,
                        k_position_field_deltas=k_position_field_deltas,
                        source_position_kinds=source_position_kinds,
                        alpha=float(spec["alpha"]),
                        batch=batch,
                        logical_n_heads=logical_n_heads,
                    )
                    after_logits, after_cache = cache_forward(
                        model,
                        batch["base_tokens"],
                        cache_hook_names=pattern_hook_names,
                        fwd_hooks=hooks,
                    )

                after_masses = attention_masses_for_heads(
                    cache=after_cache,
                    heads=b_heads,
                    batch=batch,
                    source_position_kinds=source_position_kinds,
                )
                after_carrier_masses = carrier_masses_for_heads(
                    cache=after_cache,
                    heads=b_heads,
                    batch=batch,
                )
                update_attention_stats(
                    working_stats[name]["attention"],
                    before_masses=before_masses,
                    after_masses=after_masses,
                    original_entity=original_entity,
                    target_entity=target_entity,
                    eps=attention_eps,
                )
                update_carrier_stats(
                    working_stats[name]["carrier_attention"],
                    before_masses=before_carrier_masses,
                    after_masses=after_carrier_masses,
                    eps=attention_eps,
                )

                batch_size = int(batch["base_tokens"].shape[0])
                before_mean_mass = mean_mass_by_entity(before_masses)
                after_mean_mass = mean_mass_by_entity(after_masses)
                before_mean_carrier_mass = mean_carrier_mass(before_carrier_masses)
                after_mean_carrier_mass = mean_carrier_mass(after_carrier_masses)
                for batch_idx in range(batch_size):
                    answer_pos = int(batch["base_last_token_indices"][batch_idx].item())
                    object_token_ids = batch["object_token_ids"][batch_idx].to(torch.long)
                    sample_logit_payload = update_logit_stats(
                        working_stats[name]["logits"],
                        before_logits_at_answer=before_logits[batch_idx, answer_pos],
                        after_logits_at_answer=after_logits[batch_idx, answer_pos],
                        object_token_ids=object_token_ids,
                        original_entity=original_entity,
                        target_entity=target_entity,
                    )
                    if include_per_sample:
                        before_mass_row = before_mean_mass[batch_idx]
                        after_mass_row = after_mean_mass[batch_idx]
                        before_carrier_row = before_mean_carrier_mass[batch_idx]
                        after_carrier_row = after_mean_carrier_mass[batch_idx]
                        working_stats[name]["per_sample"].append(
                            {
                                "prompt_id": int(batch["prompt_ids"][batch_idx].item()),
                                "query_entity": int(
                                    batch["query_entities"][batch_idx].item()
                                ),
                                "original_source_entity": int(original_entity),
                                "target_source_entity": int(target_entity),
                                "object_tokens": {
                                    str(entity): {
                                        "token_id": int(object_token_ids[entity].item()),
                                        "token": tokenizer.decode(
                                            [int(object_token_ids[entity].item())],
                                            clean_up_tokenization_spaces=False,
                                        ),
                                    }
                                    for entity in range(NUM_ENTITIES)
                                },
                                "mean_b_attention_mass_by_entity": {
                                    "before": tensor_row_to_floats(before_mass_row),
                                    "after": tensor_row_to_floats(after_mass_row),
                                },
                                "mean_b_carrier_attention_mass": {
                                    "before": tensor_carrier_row_to_floats(
                                        before_carrier_row
                                    ),
                                    "after": tensor_carrier_row_to_floats(
                                        after_carrier_row
                                    ),
                                },
                                "mean_b_top_entity": {
                                    "before": int(torch.argmax(before_mass_row).item()),
                                    "after": int(torch.argmax(after_mass_row).item()),
                                },
                                "mean_b_top_carrier": {
                                    "before": carrier_labels()[
                                        int(torch.argmax(before_carrier_row).item())
                                    ],
                                    "after": carrier_labels()[
                                        int(torch.argmax(after_carrier_row).item())
                                    ],
                                },
                                "logits": sample_logit_payload,
                            }
                        )

                if after_cache is not before_cache:
                    del after_cache, after_logits
            del before_cache, before_logits
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for spec in condition_specs:
        name = str(spec["name"])
        stats = working_stats[name]
        public_spec = {
            key: value
            for key, value in spec.items()
            if key not in {"q_deltas", "k_entity_deltas", "k_position_field_deltas"}
        }
        intervention_heads = list(public_spec.get("intervention_heads", []))
        q_intervention_hooks = (
            hook_names_for_heads(intervention_heads, q_hook_name)
            if spec.get("q_deltas") is not None
            else []
        )
        k_intervention_hooks = (
            hook_names_for_heads(intervention_heads, k_hook_name)
            if (
                spec.get("k_entity_deltas") is not None
                or spec.get("k_position_field_deltas") is not None
            )
            else []
        )
        public_spec["intervention_heads"] = [
            {"layer": int(layer), "head": int(head)}
            for layer, head in intervention_heads
        ]
        public_spec["intervention_hook_names"] = {
            "q": q_intervention_hooks,
            "k": k_intervention_hooks,
            "pattern_readout": hook_names_for_heads(b_heads, pattern_hook_name),
        }
        results[name] = {
            "condition": public_spec,
            "attention": finalize_attention_stats(
                stats["attention"],
                original_entity=original_entity,
                target_entity=target_entity,
            ),
            "carrier_attention": finalize_carrier_stats(stats["carrier_attention"]),
            "final_logits": finalize_logit_stats(
                stats["logits"],
                original_entity=original_entity,
                target_entity=target_entity,
            ),
            "per_sample": stats["per_sample"] if include_per_sample else None,
        }
    return results


def build_condition_specs(
    *,
    requested_conditions: Sequence[str],
    b_heads: Sequence[Head],
    wrong_heads: Sequence[Head],
    original_entity: int,
    target_entity: int,
    distractor_entity: int,
    alpha: float,
    b_q_original_to_target: TensorByHead,
    b_k_target_to_original: TensorByHead,
    b_k_original_to_target: TensorByHead,
    q_select_question_query_box_more: TensorByHead,
    q_select_question_query_box_less: TensorByHead,
    k_select_question_query_box_more: TensorByHead,
    k_select_question_query_box_less: TensorByHead,
    random_q_original_to_target: TensorByHead,
    random_k_target_to_original: TensorByHead,
    random_k_original_to_target: TensorByHead,
    wrong_q_original_to_target: Optional[TensorByHead],
    wrong_k_target_to_original: Optional[TensorByHead],
    wrong_k_original_to_target: Optional[TensorByHead],
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []

    def add(
        name: str,
        *,
        category: str,
        description: str,
        intervention_heads: Sequence[Head],
        q_deltas: Optional[TensorByHead] = None,
        q_position_field: Optional[str] = None,
        k_entity_deltas: Optional[Mapping[int, TensorByHead]] = None,
        k_position_field_deltas: Optional[Mapping[str, TensorByHead]] = None,
    ) -> None:
        specs.append(
            {
                "name": name,
                "category": category,
                "description": description,
                "alpha": float(alpha),
                "intervention_heads": list(intervention_heads),
                "q_deltas": q_deltas,
                "q_position_field": q_position_field,
                "k_entity_deltas": dict(k_entity_deltas) if k_entity_deltas else None,
                "k_position_field_deltas": (
                    dict(k_position_field_deltas) if k_position_field_deltas else None
                ),
                "k_delta_entities": (
                    {str(entity): "custom_delta" for entity in k_entity_deltas}
                    if k_entity_deltas
                    else {}
                ),
                "k_delta_position_fields": (
                    {str(field): "custom_delta" for field in k_position_field_deltas}
                    if k_position_field_deltas
                    else {}
                ),
            }
        )

    for condition in requested_conditions:
        if condition == "no_intervention":
            add(
                condition,
                category="baseline",
                description="No Q/K intervention; after metrics equal before metrics.",
                intervention_heads=[],
            )
        elif condition == "query_shift":
            add(
                condition,
                category="main",
                description=(
                    "Add Q delta original_source->target_source to B heads at "
                    "answer_colon."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "source_swap":
            add(
                condition,
                category="main",
                description=(
                    "Swap K-side IDs at original and target context source positions "
                    "for B heads."
                ),
                intervention_heads=b_heads,
                k_entity_deltas={
                    int(target_entity): b_k_target_to_original,
                    int(original_entity): b_k_original_to_target,
                },
            )
        elif condition == "both_restore":
            add(
                condition,
                category="main",
                description=(
                    "Apply Q original->target and the K swap together; the shifted "
                    "query should again match the original source slot."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="base_answer_colon_positions",
                k_entity_deltas={
                    int(target_entity): b_k_target_to_original,
                    int(original_entity): b_k_original_to_target,
                },
            )
        elif condition == "random_query_shift":
            add(
                condition,
                category="control",
                description="Matched-norm random Q direction on B heads at answer_colon.",
                intervention_heads=b_heads,
                q_deltas=random_q_original_to_target,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "random_source_swap":
            add(
                condition,
                category="control",
                description="Matched-norm random K directions on B heads at source slots.",
                intervention_heads=b_heads,
                k_entity_deltas={
                    int(target_entity): random_k_target_to_original,
                    int(original_entity): random_k_original_to_target,
                },
            )
        elif condition == "random_both":
            add(
                condition,
                category="control",
                description="Matched-norm random Q and K directions on B heads.",
                intervention_heads=b_heads,
                q_deltas=random_q_original_to_target,
                q_position_field="base_answer_colon_positions",
                k_entity_deltas={
                    int(target_entity): random_k_target_to_original,
                    int(original_entity): random_k_original_to_target,
                },
            )
        elif condition == "wrong_head_query_shift":
            if wrong_q_original_to_target is None:
                raise ValueError("wrong_head_query_shift requested without wrong-head Q deltas.")
            add(
                condition,
                category="control",
                description="Q original->target shift applied to non-B heads.",
                intervention_heads=wrong_heads,
                q_deltas=wrong_q_original_to_target,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "wrong_head_source_swap":
            if wrong_k_target_to_original is None or wrong_k_original_to_target is None:
                raise ValueError("wrong_head_source_swap requested without wrong-head K deltas.")
            add(
                condition,
                category="control",
                description="K source-slot swap applied to non-B heads.",
                intervention_heads=wrong_heads,
                k_entity_deltas={
                    int(target_entity): wrong_k_target_to_original,
                    int(original_entity): wrong_k_original_to_target,
                },
            )
        elif condition == "wrong_head_both":
            if (
                wrong_q_original_to_target is None
                or wrong_k_target_to_original is None
                or wrong_k_original_to_target is None
            ):
                raise ValueError("wrong_head_both requested without wrong-head deltas.")
            add(
                condition,
                category="control",
                description="Q and K binding-ID shifts applied to matched non-B heads.",
                intervention_heads=wrong_heads,
                q_deltas=wrong_q_original_to_target,
                q_position_field="base_answer_colon_positions",
                k_entity_deltas={
                    int(target_entity): wrong_k_target_to_original,
                    int(original_entity): wrong_k_original_to_target,
                },
            )
        elif condition == "wrong_position_query_shift":
            add(
                condition,
                category="control",
                description=(
                    "B-head Q original->target shift applied at the question box token "
                    "instead of answer_colon."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="query_box_positions",
            )
        elif condition == "wrong_position_source_shift":
            add(
                condition,
                category="control",
                description=(
                    "B-head K target->original shift applied to a distractor source "
                    "slot rather than original/target slots."
                ),
                intervention_heads=b_heads,
                k_entity_deltas={int(distractor_entity): b_k_target_to_original},
            )
        elif condition == "wrong_position_both":
            add(
                condition,
                category="control",
                description=(
                    "B-head Q shift at the question box plus K shift at a distractor "
                    "source slot."
                ),
                intervention_heads=b_heads,
                q_deltas=b_q_original_to_target,
                q_position_field="query_box_positions",
                k_entity_deltas={int(distractor_entity): b_k_target_to_original},
            )
        elif condition == "question_query_box_source_shift":
            add(
                condition,
                category="control",
                description=(
                    "B-head K target->original binding-ID delta applied at the "
                    "question_query_box token as the source position."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": b_k_target_to_original,
                },
            )
        elif condition == "q_select_question_query_box_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Add a matched-norm mean K(question_query_box) direction to "
                    "B-head Q at answer_colon, increasing compatibility with the "
                    "question_query_box carrier."
                ),
                intervention_heads=b_heads,
                q_deltas=q_select_question_query_box_more,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "q_select_question_query_box_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Subtract the matched-norm mean K(question_query_box) direction "
                    "from B-head Q at answer_colon, decreasing compatibility with "
                    "the question_query_box carrier."
                ),
                intervention_heads=b_heads,
                q_deltas=q_select_question_query_box_less,
                q_position_field="base_answer_colon_positions",
            )
        elif condition == "k_question_query_box_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Add a matched-norm mean Q(answer_colon) direction to B-head K "
                    "at the question_query_box token."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_more,
                },
            )
        elif condition == "k_question_query_box_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Subtract the matched-norm mean Q(answer_colon) direction from "
                    "B-head K at the question_query_box token."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_less,
                },
            )
        elif condition == "k_competing_carriers_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Add a matched-norm mean Q(answer_colon) direction to B-head K "
                    "at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "box_token_positions": k_select_question_query_box_more,
                },
            )
        elif condition == "k_competing_carriers_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Subtract the matched-norm mean Q(answer_colon) direction from "
                    "B-head K at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "box_token_positions": k_select_question_query_box_less,
                },
            )
        elif condition == "k_question_query_box_more_competing_less":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Increase B-head K compatibility at question_query_box while "
                    "decreasing compatibility at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_more,
                    "box_token_positions": k_select_question_query_box_less,
                },
            )
        elif condition == "k_question_query_box_less_competing_more":
            add(
                condition,
                category="carrier_selection",
                description=(
                    "Decrease B-head K compatibility at question_query_box while "
                    "increasing compatibility at all context box carrier tokens."
                ),
                intervention_heads=b_heads,
                k_position_field_deltas={
                    "query_box_positions": k_select_question_query_box_less,
                    "box_token_positions": k_select_question_query_box_more,
                },
            )
        else:
            raise ValueError(f"Unhandled condition: {condition}")
    return specs
