from __future__ import annotations

from exp3.head_metrics import (
    finalize_metrics,
    logit_margin,
    move_batch_to_device,
    new_metrics,
    optional_metric_delta,
    scalar_delta,
    token_delta,
    token_stats,
    update_metrics,
)

from collections import defaultdict
from tqdm import tqdm
import torch

def positions_for_sites(
    batch: Mapping[str, torch.Tensor],
    sites: Sequence[str],
    *,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    out = {}
    for site in sites:
        field = f"{prefix}_{site}_positions"
        if field not in batch:
            raise KeyError(f"Missing position field {field!r}.")
        out[site] = batch[field]
    return out


def patch_offsets_for_site(site: str) -> Tuple[int, ...]: 
    return (0,)


def patch_selected_attention_patterns(
    pattern: torch.Tensor,
    hook,
    *,
    source_cache: Mapping[str, torch.Tensor],
    original_cache: Optional[Mapping[str, torch.Tensor]] = None,
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    base_positions: Mapping[str, torch.Tensor],
    source_positions: Mapping[str, torch.Tensor],
    restore_heads_by_hook: Optional[Mapping[str, Sequence[int]]] = None,
    restore_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pattern.ndim != 4:
        raise ValueError(
            f"Expected attention pattern [batch, head, dest, src], got "
            f"{tuple(pattern.shape)} at {hook.name}."
        )

    source_pattern = source_cache[hook.name].to(pattern.device)
    if source_pattern.ndim != 4:
        raise ValueError(
            f"Expected source attention pattern [batch, head, dest, src], got "
            f"{tuple(source_pattern.shape)} at {hook.name}."
        )
    if pattern.shape[3] != source_pattern.shape[3]:
        raise ValueError(
            f"Cannot patch attention patterns with mismatched source lengths at "
            f"{hook.name}: base src={pattern.shape[3]}, source src={source_pattern.shape[3]}."
        )

    batch_size = int(pattern.shape[0])
    for site, base_pos_tensor in base_positions.items():
        heads = [
            head
            for head in heads_by_site_hook.get(site, {}).get(hook.name, ())
            if head < pattern.shape[1]
        ]
        if not heads:
            continue
        source_pos_tensor = source_positions[site]
        patch_offsets = patch_offsets_for_site(site)
        for batch_idx in range(batch_size):
            base_start = int(base_pos_tensor[batch_idx].item())
            source_start = int(source_pos_tensor[batch_idx].item())
            for offset in patch_offsets:
                base_pos = base_start + offset
                source_pos = source_start + offset
                if (
                    base_pos < 0
                    or base_pos >= pattern.shape[2]
                    or source_pos < 0
                    or source_pos >= source_pattern.shape[2]
                ):
                    continue
                for head_idx in heads:
                    pattern[batch_idx, head_idx, base_pos, :] = source_pattern[
                        batch_idx,
                        head_idx,
                        source_pos,
                        :,
                    ]
    if (
        original_cache is not None
        and restore_heads_by_hook is not None
        and restore_positions is not None
    ):
        restore_heads = [
            head
            for head in restore_heads_by_hook.get(hook.name, ())
            if head < pattern.shape[1]
        ]
        if restore_heads:
            original_pattern = original_cache[hook.name].to(pattern.device)
            for batch_idx in range(batch_size):
                restore_pos = int(restore_positions[batch_idx].item())
                if restore_pos < 0 or restore_pos >= pattern.shape[2]:
                    continue
                for head_idx in restore_heads:
                    pattern[batch_idx, head_idx, restore_pos, :] = original_pattern[
                        batch_idx,
                        head_idx,
                        restore_pos,
                        :,
                    ]
    return pattern


def patch_selected_heads(
    head_output: torch.Tensor,
    hook,
    *,
    source_cache: Mapping[str, torch.Tensor],
    original_cache: Optional[Mapping[str, torch.Tensor]] = None,
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    base_positions: Mapping[str, torch.Tensor],
    source_positions: Mapping[str, torch.Tensor],
    restore_heads_by_hook: Optional[Mapping[str, Sequence[int]]] = None,
    restore_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if hook.name.endswith("hook_pattern"):
        return patch_selected_attention_patterns(
            head_output,
            hook,
            source_cache=source_cache,
            original_cache=original_cache,
            heads_by_site_hook=heads_by_site_hook,
            base_positions=base_positions,
            source_positions=source_positions,
            restore_heads_by_hook=restore_heads_by_hook,
            restore_positions=restore_positions,
        )

    if head_output.ndim != 4:
        raise ValueError(
            f"Expected head activation [batch, seq, head, dim], got "
            f"{tuple(head_output.shape)} at {hook.name}."
        )

    source_output = source_cache[hook.name].to(head_output.device)
    batch_size = int(head_output.shape[0])
    for site, base_pos_tensor in base_positions.items():
        heads = [
            head
            for head in heads_by_site_hook.get(site, {}).get(hook.name, ())
            if head < head_output.shape[2]
        ]
        if not heads:
            continue
        source_pos_tensor = source_positions[site]
        patch_offsets = patch_offsets_for_site(site)
        for batch_idx in range(batch_size):
            base_start = int(base_pos_tensor[batch_idx].item())
            source_start = int(source_pos_tensor[batch_idx].item())
            for offset in patch_offsets:
                base_pos = base_start + offset
                source_pos = source_start + offset
                if (
                    base_pos < 0
                    or base_pos >= head_output.shape[1]
                    or source_pos < 0
                    or source_pos >= source_output.shape[1]
                ):
                    continue
                for head_idx in heads:
                    head_output[batch_idx, base_pos, head_idx, :] = source_output[
                        batch_idx,
                        source_pos,
                        head_idx,
                        :,
                    ]
    if (
        original_cache is not None
        and restore_heads_by_hook is not None
        and restore_positions is not None
    ):
        restore_heads = [
            head
            for head in restore_heads_by_hook.get(hook.name, ())
            if head < head_output.shape[2]
        ]
        if restore_heads:
            original_output = original_cache[hook.name].to(head_output.device)
            for batch_idx in range(batch_size):
                restore_pos = int(restore_positions[batch_idx].item())
                for head_idx in restore_heads:
                    head_output[batch_idx, restore_pos, head_idx, :] = original_output[
                        batch_idx,
                        restore_pos,
                        head_idx,
                        :,
                    ]
    return head_output


def heads_by_hook_for_sites(
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    sites: Sequence[str],
) -> Dict[str, List[int]]:
    heads_by_hook: Dict[str, set] = defaultdict(set)
    for site in sites:
        for hook, heads in heads_by_site_hook.get(site, {}).items():
            heads_by_hook[hook].update(int(head) for head in heads)
    return {hook: sorted(heads) for hook, heads in heads_by_hook.items()}


def evaluate_interchange(
    *,
    model,
    tokenizer,
    dataloader: DataLoader,
    hook_names: Sequence[str],
    heads_by_site_hook: Mapping[str, Mapping[str, Sequence[int]]],
    sites: Sequence[str],
    device: torch.device,
    restore_answer_colon: bool,
) -> Dict[str, Dict[str, Optional[float]]]:
    metrics = {
        "original": new_metrics(),
        "counterfactual": new_metrics(),
        "patched": new_metrics(),
    }
    query_object_token_effects = []
    source_label_token_effects = []
    sample_index = 0
    hook_name_set = set(hook_names)
    should_restore_answer_colon = restore_answer_colon and "answer_colon" not in sites
    restore_heads_by_hook = (
        heads_by_hook_for_sites(heads_by_site_hook, sites)
        if should_restore_answer_colon
        else None
    )
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Interchange sites={','.join(sites)}"):
            batch = move_batch_to_device(batch, device)
            base_positions = positions_for_sites(batch, sites, prefix="base")
            source_positions = positions_for_sites(batch, sites, prefix="source")

            source_logits, source_cache = model.run_with_cache(
                batch["source_tokens"],
                names_filter=lambda name: name in hook_name_set,
                return_type="logits",
            )
            if should_restore_answer_colon:
                original_logits, original_cache = model.run_with_cache(
                    batch["base_tokens"],
                    names_filter=lambda name: name in hook_name_set,
                    return_type="logits",
                )
            else:
                original_logits = model(batch["base_tokens"], return_type="logits")
                original_cache = None
            hooks = [
                (
                    name,
                    lambda z, hook, name=name: patch_selected_heads(
                        z,
                        hook,
                        source_cache=source_cache,
                        original_cache=original_cache if should_restore_answer_colon else None,
                        heads_by_site_hook=heads_by_site_hook,
                        base_positions=base_positions,
                        source_positions=source_positions,
                        restore_heads_by_hook=restore_heads_by_hook,
                        restore_positions=(
                            batch["base_answer_colon_positions"]
                            if should_restore_answer_colon
                            else None
                        ),
                    ),
                )
                for name in hook_names
            ]
            patched_logits = model.run_with_hooks(
                batch["base_tokens"],
                fwd_hooks=hooks,
                return_type="logits",
            )

            for batch_idx in range(batch["labels"].shape[0]):
                label = int(batch["labels"][batch_idx].item())
                source_label = int(batch["source_labels"][batch_idx].item())
                query_object_label = int(
                    batch["query_box_object_labels"][batch_idx].item()
                )
                source_query_object_label = int(
                    batch["source_query_box_object_labels"][batch_idx].item()
                )
                base_last = int(batch["base_last_token_indices"][batch_idx].item())
                source_last = int(batch["source_last_token_indices"][batch_idx].item())
                candidates = (
                    batch["candidate_labels"][batch_idx]
                    if "candidate_labels" in batch
                    else None
                )
                update_metrics(
                    metrics["original"],
                    original_logits[batch_idx, base_last],
                    label,
                    candidates,
                    source_label=source_label,
                    query_object_label=query_object_label,
                    source_query_object_label=source_query_object_label,
                )
                update_metrics(
                    metrics["counterfactual"],
                    source_logits[batch_idx, source_last],
                    label,
                    candidates,
                    source_label=source_label,
                    query_object_label=query_object_label,
                    source_query_object_label=source_query_object_label,
                )
                update_metrics(
                    metrics["patched"],
                    patched_logits[batch_idx, base_last],
                    label,
                    candidates,
                    source_label=source_label,
                    query_object_label=query_object_label,
                    source_query_object_label=source_query_object_label,
                )
                original_source_label_stats = token_stats(
                    original_logits[batch_idx, base_last],
                    source_label,
                )
                counterfactual_source_label_stats = token_stats(
                    source_logits[batch_idx, source_last],
                    source_label,
                )
                patched_source_label_stats = token_stats(
                    patched_logits[batch_idx, base_last],
                    source_label,
                )
                original_object_stats = token_stats(
                    original_logits[batch_idx, base_last],
                    query_object_label,
                )
                counterfactual_object_stats = token_stats(
                    source_logits[batch_idx, source_last],
                    query_object_label,
                )
                patched_object_stats = token_stats(
                    patched_logits[batch_idx, base_last],
                    query_object_label,
                )
                original_source_object_stats = token_stats(
                    original_logits[batch_idx, base_last],
                    source_query_object_label,
                )
                counterfactual_source_object_stats = token_stats(
                    source_logits[batch_idx, source_last],
                    source_query_object_label,
                )
                patched_source_object_stats = token_stats(
                    patched_logits[batch_idx, base_last],
                    source_query_object_label,
                )
                original_source_label_margin = logit_margin(
                    original_logits[batch_idx, base_last],
                    source_label,
                    label,
                )
                counterfactual_source_label_margin = logit_margin(
                    source_logits[batch_idx, source_last],
                    source_label,
                    label,
                )
                patched_source_label_margin = logit_margin(
                    patched_logits[batch_idx, base_last],
                    source_label,
                    label,
                )
                original_query_label_margin = logit_margin(
                    original_logits[batch_idx, base_last],
                    query_object_label,
                    label,
                )
                counterfactual_query_label_margin = logit_margin(
                    source_logits[batch_idx, source_last],
                    query_object_label,
                    label,
                )
                patched_query_label_margin = logit_margin(
                    patched_logits[batch_idx, base_last],
                    query_object_label,
                    label,
                )
                query_object_token_effects.append(
                    {
                        "sample_index": sample_index,
                        "query_box_object_token_id": query_object_label,
                        "query_box_object_token": tokenizer.decode(
                            [query_object_label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "answer_label_token_id": label,
                        "answer_label_token": tokenizer.decode(
                            [label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "original": original_object_stats,
                        "counterfactual": counterfactual_object_stats,
                        "patched": patched_object_stats,
                        "patched_minus_original": token_delta(
                            patched_object_stats,
                            original_object_stats,
                        ),
                        "query_label_minus_target_label_logit_margin": {
                            "original": original_query_label_margin,
                            "counterfactual": counterfactual_query_label_margin,
                            "patched": patched_query_label_margin,
                            "patched_minus_original": scalar_delta(
                                patched_query_label_margin,
                                original_query_label_margin,
                            ),
                            "patched_minus_counterfactual": scalar_delta(
                                patched_query_label_margin,
                                counterfactual_query_label_margin,
                            ),
                        },
                        "source_query_box_object_token_id": source_query_object_label,
                        "source_query_box_object_token": tokenizer.decode(
                            [source_query_object_label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "source_query_box_object_original": original_source_object_stats,
                        "source_query_box_object_counterfactual": (
                            counterfactual_source_object_stats
                        ),
                        "source_query_box_object_patched": patched_source_object_stats,
                        "source_query_box_object_patched_minus_original": token_delta(
                            patched_source_object_stats,
                            original_source_object_stats,
                        ),
                    }
                )
                source_label_token_effects.append(
                    {
                        "sample_index": sample_index,
                        "source_label_token_id": source_label,
                        "source_label_token": tokenizer.decode(
                            [source_label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "answer_label_token_id": label,
                        "answer_label_token": tokenizer.decode(
                            [label],
                            clean_up_tokenization_spaces=False,
                        ),
                        "original": original_source_label_stats,
                        "counterfactual": counterfactual_source_label_stats,
                        "patched": patched_source_label_stats,
                        "patched_minus_original": token_delta(
                            patched_source_label_stats,
                            original_source_label_stats,
                        ),
                        "patched_minus_counterfactual": token_delta(
                            patched_source_label_stats,
                            counterfactual_source_label_stats,
                        ),
                        "source_label_minus_original_label_logit_margin": {
                            "original": original_source_label_margin,
                            "counterfactual": counterfactual_source_label_margin,
                            "patched": patched_source_label_margin,
                            "patched_minus_original": scalar_delta(
                                patched_source_label_margin,
                                original_source_label_margin,
                            ),
                            "patched_minus_counterfactual": scalar_delta(
                                patched_source_label_margin,
                                counterfactual_source_label_margin,
                            ),
                        },
                    }
                )
                sample_index += 1

            del source_logits, source_cache, original_logits, patched_logits
            if original_cache is not None:
                del original_cache
            torch.cuda.empty_cache()

    finalized = {name: finalize_metrics(value) for name, value in metrics.items()}
    patched = finalized["patched"]
    original = finalized["original"]
    counterfactual = finalized["counterfactual"]
    finalized["deltas"] = {
        "patched_minus_original_accuracy": (
            None
            if patched["accuracy"] is None or original["accuracy"] is None
            else patched["accuracy"] - original["accuracy"]
        ),
        "patched_minus_original_candidate_accuracy": (
            None
            if patched["candidate_accuracy"] is None
            or original["candidate_accuracy"] is None
            else patched["candidate_accuracy"] - original["candidate_accuracy"]
        ),
        "patched_minus_original_mean_label_logit": (
            None
            if patched["mean_label_logit"] is None
            or original["mean_label_logit"] is None
            else patched["mean_label_logit"] - original["mean_label_logit"]
        ),
        "patched_minus_counterfactual_mean_label_logit": (
            None
            if patched["mean_label_logit"] is None
            or counterfactual["mean_label_logit"] is None
            else patched["mean_label_logit"] - counterfactual["mean_label_logit"]
        ),
        "patched_minus_original_mean_source_label_logit": (
            None
            if patched["mean_source_label_logit"] is None
            or original["mean_source_label_logit"] is None
            else patched["mean_source_label_logit"]
            - original["mean_source_label_logit"]
        ),
        "patched_minus_original_mean_source_label_prob": (
            None
            if patched["mean_source_label_prob"] is None
            or original["mean_source_label_prob"] is None
            else patched["mean_source_label_prob"]
            - original["mean_source_label_prob"]
        ),
        "patched_minus_counterfactual_mean_source_label_logit": (
            None
            if patched["mean_source_label_logit"] is None
            or counterfactual["mean_source_label_logit"] is None
            else patched["mean_source_label_logit"]
            - counterfactual["mean_source_label_logit"]
        ),
        "patched_minus_original_mean_source_label_minus_original_label_logit_margin": (
            optional_metric_delta(
                patched,
                original,
                "mean_source_label_minus_original_label_logit_margin",
            )
        ),
        "patched_minus_counterfactual_mean_source_label_minus_original_label_logit_margin": (
            optional_metric_delta(
                patched,
                counterfactual,
                "mean_source_label_minus_original_label_logit_margin",
            )
        ),
        "patched_minus_original_mean_query_object_logit": (
            None
            if patched["mean_query_object_logit"] is None
            or original["mean_query_object_logit"] is None
            else patched["mean_query_object_logit"]
            - original["mean_query_object_logit"]
        ),
        "patched_minus_original_mean_query_object_prob": (
            None
            if patched["mean_query_object_prob"] is None
            or original["mean_query_object_prob"] is None
            else patched["mean_query_object_prob"]
            - original["mean_query_object_prob"]
        ),
        "patched_minus_counterfactual_mean_query_object_logit": (
            None
            if patched["mean_query_object_logit"] is None
            or counterfactual["mean_query_object_logit"] is None
            else patched["mean_query_object_logit"]
            - counterfactual["mean_query_object_logit"]
        ),
        "patched_minus_original_mean_query_label_minus_target_label_logit_margin": (
            optional_metric_delta(
                patched,
                original,
                "mean_query_label_minus_target_label_logit_margin",
            )
        ),
        "patched_minus_counterfactual_mean_query_label_minus_target_label_logit_margin": (
            optional_metric_delta(
                patched,
                counterfactual,
                "mean_query_label_minus_target_label_logit_margin",
            )
        ),
        "patched_minus_original_mean_source_query_object_logit": (
            None
            if patched["mean_source_query_object_logit"] is None
            or original["mean_source_query_object_logit"] is None
            else patched["mean_source_query_object_logit"]
            - original["mean_source_query_object_logit"]
        ),
        "patched_minus_original_mean_source_query_object_prob": (
            None
            if patched["mean_source_query_object_prob"] is None
            or original["mean_source_query_object_prob"] is None
            else patched["mean_source_query_object_prob"]
            - original["mean_source_query_object_prob"]
        ),
        "patched_minus_counterfactual_mean_source_query_object_logit": (
            None
            if patched["mean_source_query_object_logit"] is None
            or counterfactual["mean_source_query_object_logit"] is None
            else patched["mean_source_query_object_logit"]
            - counterfactual["mean_source_query_object_logit"]
        ),
    }
    finalized["query_box_object_token_effects"] = query_object_token_effects
    finalized["source_label_token_effects"] = source_label_token_effects
    return finalized
