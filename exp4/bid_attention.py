from __future__ import annotations

from exp4.bid_data import (
    NUM_ENTITIES,
    pattern_hook_name,
)

import torch

def attention_masses_for_heads(
    *,
    cache: Mapping[str, torch.Tensor],
    heads: Sequence[Head],
    batch: Mapping[str, torch.Tensor],
    source_position_kinds: Sequence[str],
) -> Dict[Head, torch.Tensor]:
    masses_by_head: Dict[Head, torch.Tensor] = {}
    for layer, head in heads:
        pattern = cache[pattern_hook_name(layer)]
        batch_indices = torch.arange(pattern.shape[0], device=pattern.device)
        dest_positions = batch["base_answer_colon_positions"].to(pattern.device).to(torch.long)
        entity_masses = []
        for entity in range(NUM_ENTITIES):
            mass = torch.zeros(pattern.shape[0], device=pattern.device, dtype=torch.float32)
            for kind in source_position_kinds:
                src_positions = batch[f"{kind}_token_positions"][
                    :,
                    entity,
                ].to(pattern.device).to(torch.long)
                mass = mass + pattern[
                    batch_indices,
                    int(head),
                    dest_positions,
                    src_positions,
                ].detach().float()
            entity_masses.append(mass)
        masses_by_head[(int(layer), int(head))] = torch.stack(entity_masses, dim=1)
    return masses_by_head


def carrier_labels() -> List[str]:
    return ["question_query_box"] + [
        f"context_box_{entity}" for entity in range(NUM_ENTITIES)
    ]


def carrier_masses_for_heads(
    *,
    cache: Mapping[str, torch.Tensor],
    heads: Sequence[Head],
    batch: Mapping[str, torch.Tensor],
) -> Dict[Head, torch.Tensor]:
    masses_by_head: Dict[Head, torch.Tensor] = {}
    for layer, head in heads:
        pattern = cache[pattern_hook_name(layer)]
        batch_indices = torch.arange(pattern.shape[0], device=pattern.device)
        dest_positions = batch["base_answer_colon_positions"].to(pattern.device).to(torch.long)
        query_box_positions = batch["query_box_positions"].to(pattern.device).to(torch.long)
        query_box_mass = pattern[
            batch_indices,
            int(head),
            dest_positions,
            query_box_positions,
        ].detach().float()
        carrier_masses = [query_box_mass]
        for entity in range(NUM_ENTITIES):
            context_positions = batch["box_token_positions"][
                :,
                entity,
            ].to(pattern.device).to(torch.long)
            carrier_masses.append(
                pattern[
                    batch_indices,
                    int(head),
                    dest_positions,
                    context_positions,
                ].detach().float()
            )
        masses_by_head[(int(layer), int(head))] = torch.stack(carrier_masses, dim=1)
    return masses_by_head


def init_attention_bucket() -> Dict[str, object]:
    return {
        "before_mass_sum": [0.0 for _ in range(NUM_ENTITIES)],
        "after_mass_sum": [0.0 for _ in range(NUM_ENTITIES)],
        "count": 0,
        "log_ratio_before_sum": 0.0,
        "log_ratio_after_sum": 0.0,
        "switch_original_to_target": 0,
        "top_after_target": 0,
    }


def update_attention_bucket(
    bucket: Dict[str, object],
    *,
    before: torch.Tensor,
    after: torch.Tensor,
    original_entity: int,
    target_entity: int,
    eps: float,
) -> None:
    before_cpu = before.detach().float().cpu()
    after_cpu = after.detach().float().cpu()
    for entity in range(NUM_ENTITIES):
        bucket["before_mass_sum"][entity] += float(before_cpu[:, entity].sum().item())
        bucket["after_mass_sum"][entity] += float(after_cpu[:, entity].sum().item())
    bucket["count"] += int(before_cpu.shape[0])
    before_log_ratio = torch.log(before_cpu[:, target_entity] + eps) - torch.log(
        before_cpu[:, original_entity] + eps
    )
    after_log_ratio = torch.log(after_cpu[:, target_entity] + eps) - torch.log(
        after_cpu[:, original_entity] + eps
    )
    bucket["log_ratio_before_sum"] += float(before_log_ratio.sum().item())
    bucket["log_ratio_after_sum"] += float(after_log_ratio.sum().item())
    before_top = torch.argmax(before_cpu, dim=1)
    after_top = torch.argmax(after_cpu, dim=1)
    bucket["switch_original_to_target"] += int(
        ((before_top == original_entity) & (after_top == target_entity)).sum().item()
    )
    bucket["top_after_target"] += int((after_top == target_entity).sum().item())


def finalize_attention_bucket(
    bucket: Mapping[str, object],
    *,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    count = int(bucket["count"])
    if count == 0:
        return {
            "count": 0,
            "mean_before_mass_by_entity": None,
            "mean_after_mass_by_entity": None,
        }
    before = [value / count for value in bucket["before_mass_sum"]]
    after = [value / count for value in bucket["after_mass_sum"]]
    before_tmo = before[target_entity] - before[original_entity]
    after_tmo = after[target_entity] - after[original_entity]
    before_lr = float(bucket["log_ratio_before_sum"]) / count
    after_lr = float(bucket["log_ratio_after_sum"]) / count
    return {
        "count": count,
        "mean_before_mass_by_entity": {
            str(entity): before[entity] for entity in range(NUM_ENTITIES)
        },
        "mean_after_mass_by_entity": {
            str(entity): after[entity] for entity in range(NUM_ENTITIES)
        },
        "target_minus_original_attention_mass": {
            "before": before_tmo,
            "after": after_tmo,
            "delta_after_minus_before": after_tmo - before_tmo,
        },
        "log_ratio_target_over_original": {
            "before": before_lr,
            "after": after_lr,
            "delta_after_minus_before": after_lr - before_lr,
        },
        "switch_original_to_target_fraction": (
            int(bucket["switch_original_to_target"]) / count
        ),
        "top_after_target_fraction": int(bucket["top_after_target"]) / count,
    }


def init_attention_stats(heads: Sequence[Head]) -> Dict[str, object]:
    return {
        "overall": init_attention_bucket(),
        "by_head": {f"L{layer}H{head}": init_attention_bucket() for layer, head in heads},
    }


def update_attention_stats(
    stats: Dict[str, object],
    *,
    before_masses: Mapping[Head, torch.Tensor],
    after_masses: Mapping[Head, torch.Tensor],
    original_entity: int,
    target_entity: int,
    eps: float,
) -> None:
    for head, before in before_masses.items():
        after = after_masses[head]
        key = f"L{head[0]}H{head[1]}"
        update_attention_bucket(
            stats["by_head"][key],
            before=before,
            after=after,
            original_entity=original_entity,
            target_entity=target_entity,
            eps=eps,
        )
        update_attention_bucket(
            stats["overall"],
            before=before,
            after=after,
            original_entity=original_entity,
            target_entity=target_entity,
            eps=eps,
        )


def finalize_attention_stats(
    stats: Mapping[str, object],
    *,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    return {
        "overall": finalize_attention_bucket(
            stats["overall"],
            original_entity=original_entity,
            target_entity=target_entity,
        ),
        "by_head": {
            head_key: finalize_attention_bucket(
                bucket,
                original_entity=original_entity,
                target_entity=target_entity,
            )
            for head_key, bucket in stats["by_head"].items()
        },
    }


def init_carrier_bucket() -> Dict[str, object]:
    return {
        "before_mass_sum": [0.0 for _ in carrier_labels()],
        "after_mass_sum": [0.0 for _ in carrier_labels()],
        "count": 0,
        "log_ratio_before_sum": 0.0,
        "log_ratio_after_sum": 0.0,
        "top_before_question": 0,
        "top_after_question": 0,
        "switch_to_question": 0,
    }


def update_carrier_bucket(
    bucket: Dict[str, object],
    *,
    before: torch.Tensor,
    after: torch.Tensor,
    eps: float,
) -> None:
    before_cpu = before.detach().float().cpu()
    after_cpu = after.detach().float().cpu()
    labels = carrier_labels()
    for idx in range(len(labels)):
        bucket["before_mass_sum"][idx] += float(before_cpu[:, idx].sum().item())
        bucket["after_mass_sum"][idx] += float(after_cpu[:, idx].sum().item())
    bucket["count"] += int(before_cpu.shape[0])

    before_context = before_cpu[:, 1:].sum(dim=1)
    after_context = after_cpu[:, 1:].sum(dim=1)
    before_log_ratio = torch.log(before_cpu[:, 0] + eps) - torch.log(
        before_context + eps
    )
    after_log_ratio = torch.log(after_cpu[:, 0] + eps) - torch.log(
        after_context + eps
    )
    bucket["log_ratio_before_sum"] += float(before_log_ratio.sum().item())
    bucket["log_ratio_after_sum"] += float(after_log_ratio.sum().item())

    before_top = torch.argmax(before_cpu, dim=1)
    after_top = torch.argmax(after_cpu, dim=1)
    bucket["top_before_question"] += int((before_top == 0).sum().item())
    bucket["top_after_question"] += int((after_top == 0).sum().item())
    bucket["switch_to_question"] += int(
        ((before_top != 0) & (after_top == 0)).sum().item()
    )


def finalize_carrier_bucket(bucket: Mapping[str, object]) -> Dict[str, object]:
    count = int(bucket["count"])
    labels = carrier_labels()
    if count == 0:
        return {
            "count": 0,
            "mean_before_mass_by_carrier": None,
            "mean_after_mass_by_carrier": None,
        }
    before = [value / count for value in bucket["before_mass_sum"]]
    after = [value / count for value in bucket["after_mass_sum"]]
    before_context = sum(before[1:])
    after_context = sum(after[1:])
    before_qmc = before[0] - before_context
    after_qmc = after[0] - after_context
    before_lr = float(bucket["log_ratio_before_sum"]) / count
    after_lr = float(bucket["log_ratio_after_sum"]) / count
    return {
        "count": count,
        "carrier_order": labels,
        "mean_before_mass_by_carrier": {
            labels[idx]: before[idx] for idx in range(len(labels))
        },
        "mean_after_mass_by_carrier": {
            labels[idx]: after[idx] for idx in range(len(labels))
        },
        "question_minus_context_boxes_attention_mass": {
            "before": before_qmc,
            "after": after_qmc,
            "delta_after_minus_before": after_qmc - before_qmc,
        },
        "log_ratio_question_over_context_boxes": {
            "before": before_lr,
            "after": after_lr,
            "delta_after_minus_before": after_lr - before_lr,
        },
        "top_before_question_fraction": int(bucket["top_before_question"]) / count,
        "top_after_question_fraction": int(bucket["top_after_question"]) / count,
        "switch_to_question_fraction": int(bucket["switch_to_question"]) / count,
    }


def init_carrier_stats(heads: Sequence[Head]) -> Dict[str, object]:
    return {
        "overall": init_carrier_bucket(),
        "by_head": {f"L{layer}H{head}": init_carrier_bucket() for layer, head in heads},
    }


def update_carrier_stats(
    stats: Dict[str, object],
    *,
    before_masses: Mapping[Head, torch.Tensor],
    after_masses: Mapping[Head, torch.Tensor],
    eps: float,
) -> None:
    for head, before in before_masses.items():
        after = after_masses[head]
        key = f"L{head[0]}H{head[1]}"
        update_carrier_bucket(
            stats["by_head"][key],
            before=before,
            after=after,
            eps=eps,
        )
        update_carrier_bucket(
            stats["overall"],
            before=before,
            after=after,
            eps=eps,
        )


def finalize_carrier_stats(stats: Mapping[str, object]) -> Dict[str, object]:
    return {
        "overall": finalize_carrier_bucket(stats["overall"]),
        "by_head": {
            head_key: finalize_carrier_bucket(bucket)
            for head_key, bucket in stats["by_head"].items()
        },
    }


def init_logit_stats() -> Dict[str, object]:
    return {
        "before_logit_sum_by_entity": [0.0 for _ in range(NUM_ENTITIES)],
        "after_logit_sum_by_entity": [0.0 for _ in range(NUM_ENTITIES)],
        "delta_logit_sum_by_entity": [0.0 for _ in range(NUM_ENTITIES)],
        "pairwise_before_sum": 0.0,
        "pairwise_after_sum": 0.0,
        "pairwise_delta_sum": 0.0,
        "count": 0,
    }


def update_logit_stats(
    stats: Dict[str, object],
    *,
    before_logits_at_answer: torch.Tensor,
    after_logits_at_answer: torch.Tensor,
    object_token_ids: torch.Tensor,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    before_values = []
    after_values = []
    delta_values = []
    for entity in range(NUM_ENTITIES):
        token_id = int(object_token_ids[entity].item())
        before_logit = float(before_logits_at_answer[token_id].item())
        after_logit = float(after_logits_at_answer[token_id].item())
        delta = after_logit - before_logit
        stats["before_logit_sum_by_entity"][entity] += before_logit
        stats["after_logit_sum_by_entity"][entity] += after_logit
        stats["delta_logit_sum_by_entity"][entity] += delta
        before_values.append(before_logit)
        after_values.append(after_logit)
        delta_values.append(delta)

    before_pairwise = before_values[target_entity] - before_values[original_entity]
    after_pairwise = after_values[target_entity] - after_values[original_entity]
    pairwise_delta = after_pairwise - before_pairwise
    stats["pairwise_before_sum"] += before_pairwise
    stats["pairwise_after_sum"] += after_pairwise
    stats["pairwise_delta_sum"] += pairwise_delta
    stats["count"] += 1
    return {
        "before_logit_by_entity": {
            str(entity): before_values[entity] for entity in range(NUM_ENTITIES)
        },
        "after_logit_by_entity": {
            str(entity): after_values[entity] for entity in range(NUM_ENTITIES)
        },
        "delta_logit_by_entity": {
            str(entity): delta_values[entity] for entity in range(NUM_ENTITIES)
        },
        "pairwise_target_minus_original": {
            "before": before_pairwise,
            "after": after_pairwise,
            "delta_after_minus_before": pairwise_delta,
        },
    }


def finalize_logit_stats(
    stats: Mapping[str, object],
    *,
    original_entity: int,
    target_entity: int,
) -> Dict[str, object]:
    count = int(stats["count"])
    if count == 0:
        return {"count": 0}
    before = [value / count for value in stats["before_logit_sum_by_entity"]]
    after = [value / count for value in stats["after_logit_sum_by_entity"]]
    delta = [value / count for value in stats["delta_logit_sum_by_entity"]]
    other_entities = [
        entity
        for entity in range(NUM_ENTITIES)
        if entity not in (int(original_entity), int(target_entity))
    ]
    other_delta = (
        sum(delta[entity] for entity in other_entities) / len(other_entities)
        if other_entities
        else None
    )
    return {
        "count": count,
        "mean_before_logit_by_entity": {
            str(entity): before[entity] for entity in range(NUM_ENTITIES)
        },
        "mean_after_logit_by_entity": {
            str(entity): after[entity] for entity in range(NUM_ENTITIES)
        },
        "mean_delta_logit_by_entity": {
            str(entity): delta[entity] for entity in range(NUM_ENTITIES)
        },
        "raw_delta_roles": {
            "original_answer": delta[original_entity],
            "target_answer": delta[target_entity],
            "other_objects_mean": other_delta,
            "other_object_entities": other_entities,
        },
        "pairwise_delta_target_minus_original": {
            "before": float(stats["pairwise_before_sum"]) / count,
            "after": float(stats["pairwise_after_sum"]) / count,
            "delta_after_minus_before": float(stats["pairwise_delta_sum"]) / count,
        },
    }


def mean_mass_by_entity(masses_by_head: Mapping[Head, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(masses_by_head.values()), dim=0)
    return stacked.mean(dim=0)


def mean_carrier_mass(masses_by_head: Mapping[Head, torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(list(masses_by_head.values()), dim=0)
    return stacked.mean(dim=0)


def tensor_row_to_floats(row: torch.Tensor) -> Dict[str, float]:
    return {str(idx): float(value) for idx, value in enumerate(row.detach().cpu().tolist())}


def tensor_carrier_row_to_floats(row: torch.Tensor) -> Dict[str, float]:
    labels = carrier_labels()
    values = row.detach().cpu().tolist()
    return {labels[idx]: float(values[idx]) for idx in range(len(labels))}
