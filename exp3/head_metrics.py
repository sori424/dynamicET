from __future__ import annotations

import torch

def move_batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def new_metrics() -> Dict[str, float]:
    return {
        "correct": 0,
        "candidate_correct": 0,
        "total": 0,
        "candidate_total": 0,
        "label_logit_sum": 0.0,
        "label_prob_sum": 0.0,
        "source_label_logit_sum": 0.0,
        "source_label_prob_sum": 0.0,
        "source_label_rank_sum": 0.0,
        "source_label_top1": 0,
        "source_label_total": 0,
        "source_label_minus_original_label_logit_margin_sum": 0.0,
        "source_label_minus_original_label_logit_margin_total": 0,
        "query_object_logit_sum": 0.0,
        "query_object_prob_sum": 0.0,
        "query_object_rank_sum": 0.0,
        "query_object_top1": 0,
        "query_object_total": 0,
        "query_label_minus_target_label_logit_margin_sum": 0.0,
        "query_label_minus_target_label_logit_margin_total": 0,
        "source_query_object_logit_sum": 0.0,
        "source_query_object_prob_sum": 0.0,
        "source_query_object_rank_sum": 0.0,
        "source_query_object_top1": 0,
        "source_query_object_total": 0,
    }


def token_stats(logits_at_answer: torch.Tensor, token_id: int) -> Dict[str, float]:
    prob = float(torch.softmax(logits_at_answer, dim=-1)[token_id].item())
    logit = float(logits_at_answer[token_id].item())
    rank = int((logits_at_answer > logits_at_answer[token_id]).sum().item()) + 1
    return {"logit": logit, "prob": prob, "rank": rank}


def token_delta(
    patched_stats: Mapping[str, float],
    original_stats: Mapping[str, float],
) -> Dict[str, float]:
    return {
        "logit": patched_stats["logit"] - original_stats["logit"],
        "prob": patched_stats["prob"] - original_stats["prob"],
        "rank": patched_stats["rank"] - original_stats["rank"],
    }


def logit_margin(
    logits_at_answer: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> float:
    return float(
        (
            logits_at_answer[positive_token_id]
            - logits_at_answer[negative_token_id]
        ).item()
    )


def scalar_delta(patched_value: float, original_value: float) -> float:
    return patched_value - original_value


def format_optional_float(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.4f}"


def optional_metric_delta(
    patched_metrics: Mapping[str, Optional[float]],
    original_metrics: Mapping[str, Optional[float]],
    key: str,
) -> Optional[float]:
    patched_value = patched_metrics[key]
    original_value = original_metrics[key]
    if patched_value is None or original_value is None:
        return None
    return patched_value - original_value


def update_metrics(
    metrics: Dict[str, float],
    logits_at_answer: torch.Tensor,
    label: int,
    candidate_labels: Optional[torch.Tensor],
    source_label: Optional[int] = None,
    query_object_label: Optional[int] = None,
    source_query_object_label: Optional[int] = None,
) -> None:
    pred = int(torch.argmax(logits_at_answer).item())
    label_logit = float(logits_at_answer[label].item())
    label_prob = float(torch.softmax(logits_at_answer, dim=-1)[label].item())
    metrics["correct"] += int(pred == label)
    metrics["total"] += 1
    metrics["label_logit_sum"] += label_logit
    metrics["label_prob_sum"] += label_prob

    if source_label is not None:
        stats = token_stats(logits_at_answer, source_label)
        metrics["source_label_logit_sum"] += stats["logit"]
        metrics["source_label_prob_sum"] += stats["prob"]
        metrics["source_label_rank_sum"] += stats["rank"]
        metrics["source_label_top1"] += int(pred == source_label)
        metrics["source_label_total"] += 1
        metrics["source_label_minus_original_label_logit_margin_sum"] += logit_margin(
            logits_at_answer,
            source_label,
            label,
        )
        metrics["source_label_minus_original_label_logit_margin_total"] += 1

    if query_object_label is not None:
        stats = token_stats(logits_at_answer, query_object_label)
        metrics["query_object_logit_sum"] += stats["logit"]
        metrics["query_object_prob_sum"] += stats["prob"]
        metrics["query_object_rank_sum"] += stats["rank"]
        metrics["query_object_top1"] += int(pred == query_object_label)
        metrics["query_object_total"] += 1
        metrics["query_label_minus_target_label_logit_margin_sum"] += logit_margin(
            logits_at_answer,
            query_object_label,
            label,
        )
        metrics["query_label_minus_target_label_logit_margin_total"] += 1

    if source_query_object_label is not None:
        stats = token_stats(logits_at_answer, source_query_object_label)
        metrics["source_query_object_logit_sum"] += stats["logit"]
        metrics["source_query_object_prob_sum"] += stats["prob"]
        metrics["source_query_object_rank_sum"] += stats["rank"]
        metrics["source_query_object_top1"] += int(pred == source_query_object_label)
        metrics["source_query_object_total"] += 1

    if candidate_labels is None:
        return

    candidates = torch.unique(candidate_labels.detach().to(logits_at_answer.device).to(torch.long))
    if not bool((candidates == label).any().item()):
        candidates = torch.cat(
            [candidates, torch.tensor([label], device=logits_at_answer.device)]
        )
    candidate_logits = logits_at_answer.index_select(dim=0, index=candidates)
    candidate_pred = int(candidates[int(torch.argmax(candidate_logits).item())].item())
    metrics["candidate_correct"] += int(candidate_pred == label)
    metrics["candidate_total"] += 1


def finalize_metrics(metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
    total = int(metrics["total"])
    candidate_total = int(metrics["candidate_total"])
    source_label_total = int(metrics["source_label_total"])
    query_object_total = int(metrics["query_object_total"])
    source_query_object_total = int(metrics["source_query_object_total"])
    source_label_margin_total = int(
        metrics["source_label_minus_original_label_logit_margin_total"]
    )
    query_label_margin_total = int(
        metrics["query_label_minus_target_label_logit_margin_total"]
    )
    return {
        "accuracy": metrics["correct"] / total if total else None,
        "candidate_accuracy": (
            metrics["candidate_correct"] / candidate_total if candidate_total else None
        ),
        "correct": int(metrics["correct"]),
        "candidate_correct": int(metrics["candidate_correct"]),
        "total": total,
        "candidate_total": candidate_total,
        "mean_label_logit": metrics["label_logit_sum"] / total if total else None,
        "mean_label_prob": metrics["label_prob_sum"] / total if total else None,
        "mean_source_label_logit": (
            metrics["source_label_logit_sum"] / source_label_total
            if source_label_total
            else None
        ),
        "mean_source_label_prob": (
            metrics["source_label_prob_sum"] / source_label_total
            if source_label_total
            else None
        ),
        "mean_source_label_rank": (
            metrics["source_label_rank_sum"] / source_label_total
            if source_label_total
            else None
        ),
        "source_label_top1_rate": (
            metrics["source_label_top1"] / source_label_total
            if source_label_total
            else None
        ),
        "source_label_total": source_label_total,
        "mean_source_label_minus_original_label_logit_margin": (
            metrics["source_label_minus_original_label_logit_margin_sum"]
            / source_label_margin_total
            if source_label_margin_total
            else None
        ),
        "source_label_minus_original_label_logit_margin_total": (
            source_label_margin_total
        ),
        "mean_query_object_logit": (
            metrics["query_object_logit_sum"] / query_object_total
            if query_object_total
            else None
        ),
        "mean_query_object_prob": (
            metrics["query_object_prob_sum"] / query_object_total
            if query_object_total
            else None
        ),
        "mean_query_object_rank": (
            metrics["query_object_rank_sum"] / query_object_total
            if query_object_total
            else None
        ),
        "query_object_top1_rate": (
            metrics["query_object_top1"] / query_object_total
            if query_object_total
            else None
        ),
        "query_object_total": query_object_total,
        "mean_query_label_minus_target_label_logit_margin": (
            metrics["query_label_minus_target_label_logit_margin_sum"]
            / query_label_margin_total
            if query_label_margin_total
            else None
        ),
        "query_label_minus_target_label_logit_margin_total": (
            query_label_margin_total
        ),
        "mean_source_query_object_logit": (
            metrics["source_query_object_logit_sum"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "mean_source_query_object_prob": (
            metrics["source_query_object_prob_sum"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "mean_source_query_object_rank": (
            metrics["source_query_object_rank_sum"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "source_query_object_top1_rate": (
            metrics["source_query_object_top1"] / source_query_object_total
            if source_query_object_total
            else None
        ),
        "source_query_object_total": source_query_object_total,
    }
