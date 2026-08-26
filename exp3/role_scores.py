from __future__ import annotations

from exp3.head_config import (
    C_TO_A_MILK_ANSWER,
    C_TO_A_MILK_BASE_PROMPT,
    C_TO_A_MILK_MEASURED_OBJECT,
    C_TO_A_MILK_SOURCE_PROMPT,
    ROLE_DEFAULT_SCORE_SITES,
    ROLE_PROMPT_SCORE_MODES,
    all_hook_names,
    component_count_by_site,
    component_heads_by_site_hook,
    filter_components_by_stage,
    select_site_heads,
    unique_layer_heads,
)

from exp3.head_data import (
    build_c_to_a_milk_dataset,
    build_paired_dataset,
)

from exp3.head_interventions import (
    evaluate_interchange,
    patch_offsets_for_site,
)

from torch.utils.data import DataLoader, Dataset

def ensure_attn_result_enabled(model, hook_names: Sequence[str]) -> None:
    if not any("hook_result" in hook for hook in hook_names):
        return
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    else:
        model.cfg.use_attn_result = True


def evaluation_plan_for_sites(
    *,
    active_sites: Sequence[str],
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    combine_sites: bool,
) -> List[Tuple[str, List[str], Dict[str, Dict[str, Sequence[int]]]]]:
    if combine_sites:
        return [("combined", list(active_sites), dict(heads_by_site_hook))]
    return [
        (site, [site], {site: heads_by_site_hook[site]})
        for site in active_sites
    ]


def evaluate_role_prompt_scores(
    *,
    model,
    tokenizer,
    vocab,
    components: Sequence[Mapping[str, object]],
    roles: Sequence[str],
    requested_sites: Optional[Sequence[str]],
    component_hook_mode: str,
    model_name: str,
    num_entities: int,
    num_samples: int,
    query_name: int,
    prompt_id_start: int,
    swap_pair: Tuple[int, int],
    template_type: str,
    prompt_format: str,
    batch_size: int,
    combine_sites: bool,
    device: torch.device,
    restore_answer_colon: bool,
    include_c_to_a_milk_experiment: bool,
) -> Dict[str, object]:
    datasets = {
        prompt_label: build_paired_dataset(
            tokenizer=tokenizer,
            model_name=model_name,
            vocab=vocab,
            num_entities=num_entities,
            num_samples=num_samples,
            query_name=query_name,
            prompt_id_start=prompt_id_start,
            swap_pair=swap_pair,
            template_type=template_type,
            counterfactual_mode=counterfactual_mode,
            prompt_format=prompt_format,
        )
        for prompt_label, counterfactual_mode in ROLE_PROMPT_SCORE_MODES.items()
    }
    dataloaders = {
        prompt_label: DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for prompt_label, dataset in datasets.items()
    }
    c_to_a_milk_dataset = (
        build_c_to_a_milk_dataset(
            tokenizer=tokenizer,
            model_name=model_name,
            prompt_format=prompt_format,
        )
        if include_c_to_a_milk_experiment
        else None
    )
    c_to_a_milk_dataloader = (
        DataLoader(
            c_to_a_milk_dataset,
            batch_size=min(batch_size, len(c_to_a_milk_dataset)),
            shuffle=False,
        )
        if c_to_a_milk_dataset is not None
        else None
    )

    role_results: Dict[str, object] = {}
    for role in roles:
        role_components = filter_components_by_stage(components, [role])
        if not role_components:
            role_results[role] = {
                "skipped": True,
                "reason": "no components for role",
            }
            continue

        all_role_heads_by_site_hook = component_heads_by_site_hook(
            role_components,
            component_hook_mode=component_hook_mode,
        )
        role_requested_sites = (
            requested_sites
            if requested_sites is not None
            else ROLE_DEFAULT_SCORE_SITES.get(role)
        )
        active_sites, role_heads_by_site_hook, missing_sites = select_site_heads(
            all_role_heads_by_site_hook,
            role_requested_sites,
            missing_ok=True,
        )
        if not active_sites:
            role_results[role] = {
                "skipped": True,
                "reason": "no components for requested sites",
                "missing_sites": missing_sites,
            }
            continue

        role_hook_names = all_hook_names(role_heads_by_site_hook)
        ensure_attn_result_enabled(model, role_hook_names)
        plan = evaluation_plan_for_sites(
            active_sites=active_sites,
            heads_by_site_hook=role_heads_by_site_hook,
            combine_sites=combine_sites,
        )

        prompt_results: Dict[str, object] = {}
        for prompt_label, counterfactual_mode in ROLE_PROMPT_SCORE_MODES.items():
            prompt_result = {
                "counterfactual_mode": counterfactual_mode,
                "preview": preview_dataset(
                    datasets[prompt_label],
                    tokenizer,
                    active_sites,
                ),
                "results": {},
            }
            for name, sites, site_heads_by_hook in plan:
                prompt_result["results"][name] = evaluate_interchange(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=dataloaders[prompt_label],
                    hook_names=role_hook_names,
                    heads_by_site_hook=site_heads_by_hook,
                    sites=sites,
                    device=device,
                    restore_answer_colon=restore_answer_colon,
                )
            prompt_results[prompt_label] = prompt_result

        custom_experiments = {}
        if c_to_a_milk_dataloader is not None and c_to_a_milk_dataset is not None:
            custom_results = {}
            for name, sites, site_heads_by_hook in plan:
                custom_results[name] = evaluate_interchange(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=c_to_a_milk_dataloader,
                    hook_names=role_hook_names,
                    heads_by_site_hook=site_heads_by_hook,
                    sites=sites,
                    device=device,
                    restore_answer_colon=restore_answer_colon,
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
                "preview": preview_dataset(
                    c_to_a_milk_dataset,
                    tokenizer,
                    active_sites,
                ),
                "results": custom_results,
            }

        role_results[role] = {
            "skipped": False,
            "component_count": len(role_components),
            "component_count_by_site": component_count_by_site(role_heads_by_site_hook),
            "default_sites": list(ROLE_DEFAULT_SCORE_SITES.get(role, ())),
            "requested_sites": (
                list(role_requested_sites)
                if role_requested_sites is not None
                else None
            ),
            "active_sites": list(active_sites),
            "heads_by_site_hook": role_heads_by_site_hook,
            "hook_names": role_hook_names,
            "heads": [[layer, head] for layer, head in unique_layer_heads(role_heads_by_site_hook)],
            "missing_sites": missing_sites,
            "prompt_scores": prompt_results,
            "custom_experiments": custom_experiments,
        }

    return role_results


def decode_prompt(tokenizer, tokens: torch.Tensor, last_idx: int) -> str:
    return tokenizer.decode(
        tokens[: last_idx + 1].tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def preview_dataset(dataset: DictTensorDataset, tokenizer, sites: Sequence[str]) -> Dict[str, object]:
    item = dataset[0]
    base_last = int(item["base_last_token_indices"].item())
    source_last = int(item["source_last_token_indices"].item())
    preview = {
        "original": decode_prompt(tokenizer, item["base_tokens"], base_last),
        "counterfactual": decode_prompt(tokenizer, item["source_tokens"], source_last),
        "label_token_id": int(item["labels"].item()),
        "label_token": tokenizer.decode([int(item["labels"].item())]),
        "source_label_token_id": int(item["source_labels"].item()),
        "source_label_token": tokenizer.decode(
            [int(item["source_labels"].item())],
            clean_up_tokenization_spaces=False,
        ),
        "query_box_object_token_id": int(item["query_box_object_labels"].item()),
        "query_box_object_token": tokenizer.decode(
            [int(item["query_box_object_labels"].item())],
            clean_up_tokenization_spaces=False,
        ),
        "source_query_box_object_token_id": int(
            item["source_query_box_object_labels"].item()
        ),
        "source_query_box_object_token": tokenizer.decode(
            [int(item["source_query_box_object_labels"].item())],
            clean_up_tokenization_spaces=False,
        ),
        "sites": {},
    }
    for site in sites:
        base_pos = int(item[f"base_{site}_positions"].item())
        source_pos = int(item[f"source_{site}_positions"].item())
        patch_offsets = patch_offsets_for_site(site)
        base_patch_positions = [
            base_pos + offset
            for offset in patch_offsets
            if 0 <= base_pos + offset < int(item["base_tokens"].shape[0])
        ]
        source_patch_positions = [
            source_pos + offset
            for offset in patch_offsets
            if 0 <= source_pos + offset < int(item["source_tokens"].shape[0])
        ]
        preview["sites"][site] = {
            "base_pos": base_pos,
            "base_token": tokenizer.decode([int(item["base_tokens"][base_pos].item())]),
            "source_pos": source_pos,
            "source_token": tokenizer.decode([int(item["source_tokens"][source_pos].item())]),
            "patch_offsets": list(patch_offsets),
            "base_patch_positions": [
                {
                    "pos": pos,
                    "token": tokenizer.decode(
                        [int(item["base_tokens"][pos].item())],
                        clean_up_tokenization_spaces=False,
                    ),
                }
                for pos in base_patch_positions
            ],
            "source_patch_positions": [
                {
                    "pos": pos,
                    "token": tokenizer.decode(
                        [int(item["source_tokens"][pos].item())],
                        clean_up_tokenization_spaces=False,
                    ),
                }
                for pos in source_patch_positions
            ],
        }
    return preview
