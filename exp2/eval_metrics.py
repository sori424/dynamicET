from __future__ import annotations

from collections import defaultdict
import math
import statistics

def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def position_tensor(batch: Mapping[str, Any], position_mode: str):
    if position_mode == "last_token":
        return batch["base_last_token_indices"]
    field = f"base_{position_mode}_positions"
    if field not in batch:
        raise KeyError(f"Missing position field {field!r}.")
    return batch[field]


def new_metrics() -> Dict[str, float]:
    return {
        "correct": 0,
        "exact_correct": 0,
        "candidate_correct": 0,
        "total": 0,
        "candidate_total": 0,
        "label_logit_sum": 0.0,
        "label_prob_sum": 0.0,
        "candidate_margin_sum": 0.0,
    }


def _valid_unique_token_ids(token_ids, *, device, fallback: Optional[Sequence[int]] = None):
    import torch

    pieces = []
    if token_ids is not None:
        ids = token_ids.detach().flatten().to(device=device, dtype=torch.long)
        ids = ids[ids >= 0]
        if ids.numel():
            pieces.append(ids)
    if fallback is not None:
        pieces.append(torch.tensor(list(fallback), device=device, dtype=torch.long))
    if not pieces:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.cat(pieces).unique()


def update_metrics(
    metrics: Dict[str, float],
    logits_at_answer,
    label: int,
    candidates,
    *,
    label_aliases=None,
    candidate_aliases=None,
):
    import torch

    device = logits_at_answer.device
    pred = int(logits_at_answer.argmax().item())
    probs = logits_at_answer.softmax(dim=-1)
    label_ids = _valid_unique_token_ids(
        label_aliases,
        device=device,
        fallback=[label],
    )
    label_logits = logits_at_answer.index_select(dim=0, index=label_ids)
    label_score = float(label_logits.max().item())
    label_prob = float(probs.index_select(dim=0, index=label_ids).sum().item())

    metrics["exact_correct"] += int(pred == label)
    metrics["correct"] += int(bool((label_ids == pred).any().item()))
    metrics["total"] += 1
    metrics["label_logit_sum"] += label_score
    metrics["label_prob_sum"] += label_prob

    candidate_source = candidate_aliases if candidate_aliases is not None else candidates
    if candidate_source is None:
        return
    candidate_ids = _valid_unique_token_ids(candidate_source, device=device)
    if candidate_ids.numel() == 0:
        return
    if not bool(torch.isin(label_ids, candidate_ids).all().item()):
        candidate_ids = torch.cat(
            [candidate_ids, label_ids]
        ).unique()
    candidate_logits = logits_at_answer.index_select(dim=0, index=candidate_ids)
    candidate_pred = int(candidate_ids[int(candidate_logits.argmax().item())].item())
    metrics["candidate_correct"] += int(bool((label_ids == candidate_pred).any().item()))
    metrics["candidate_total"] += 1

    other_mask = ~torch.isin(candidate_ids, label_ids)
    if bool(other_mask.any().item()):
        other_logits = logits_at_answer.index_select(dim=0, index=candidate_ids[other_mask])
        margin = float(label_score - other_logits.max().item())
    else:
        margin = float("nan")
    if not math.isnan(margin):
        metrics["candidate_margin_sum"] += margin


def finalize_metrics(metrics: Mapping[str, float]) -> Dict[str, Optional[float]]:
    total = int(metrics["total"])
    candidate_total = int(metrics["candidate_total"])
    return {
        "accuracy": metrics["correct"] / total if total else None,
        "full_vocab_accuracy": metrics["correct"] / total if total else None,
        "exact_token_accuracy": metrics["exact_correct"] / total if total else None,
        "candidate_accuracy": (
            metrics["candidate_correct"] / candidate_total if candidate_total else None
        ),
        "correct": int(metrics["correct"]),
        "exact_correct": int(metrics["exact_correct"]),
        "candidate_correct": int(metrics["candidate_correct"]),
        "total": total,
        "candidate_total": candidate_total,
        "mean_label_logit": metrics["label_logit_sum"] / total if total else None,
        "mean_label_prob": metrics["label_prob_sum"] / total if total else None,
        "mean_candidate_margin": (
            metrics["candidate_margin_sum"] / candidate_total if candidate_total else None
        ),
    }


def compute_mean_activations(model, dataloader, hook_names: Sequence[str], device: Any):
    hook_set = set(hook_names)
    mean_activations: Dict[str, Any] = {}
    total = 0
    import torch
    from tqdm import tqdm

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Mean activations"):
            batch = move_batch_to_device(batch, device)
            _, cache = model.run_with_cache(
                batch["base_tokens"],
                names_filter=lambda name: name in hook_set,
                return_type="logits",
            )
            batch_size = int(batch["base_tokens"].shape[0])
            total += batch_size
            for name in hook_names:
                acts = cache[name].detach().to("cpu", dtype=torch.float32)
                summed = acts.sum(dim=0)
                if name in mean_activations:
                    mean_activations[name] += summed
                else:
                    mean_activations[name] = summed
            del cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if total == 0:
        raise ValueError("Cannot compute mean activations from an empty dataloader.")
    for name in hook_names:
        mean_activations[name] /= total
    return mean_activations


def mean_ablate_activation(
    activation,
    hook,
    *,
    batch: Mapping[str, Any],
    circuit_by_position: Mapping[str, Mapping[str, Sequence[int]]],
    mean_activations: Mapping[str, Any],
    ablate_non_circuit_positions: bool,
    preserve_all_heads_at_circuit_positions: bool,
):
    if activation.ndim != 4:
        raise ValueError(
            f"Expected head activation [batch, seq, head, dim], got "
            f"{tuple(activation.shape)} at {hook.name}."
        )
    mean_act = mean_activations[hook.name].to(
        device=activation.device,
        dtype=activation.dtype,
    )
    n_heads = int(activation.shape[2])
    all_heads = set(range(n_heads))

    for batch_idx in range(int(activation.shape[0])):
        pos_to_keep: Dict[int, set] = defaultdict(set)
        for mode, by_hook in circuit_by_position.items():
            heads = by_hook.get(hook.name, ())
            if not heads:
                continue
            pos = int(position_tensor(batch, mode)[batch_idx].item())
            pos_to_keep[pos].update(int(head) for head in heads if int(head) < n_heads)

        last_pos = int(batch["base_last_token_indices"][batch_idx].item())
        for token_pos in range(last_pos + 1):
            if token_pos >= mean_act.shape[0]:
                continue
            keep_heads = pos_to_keep.get(token_pos)
            if keep_heads is None:
                if ablate_non_circuit_positions:
                    activation[batch_idx, token_pos, :, :] = mean_act[token_pos, :, :]
                continue
            if preserve_all_heads_at_circuit_positions:
                continue
            ablate_heads = sorted(all_heads - keep_heads)
            if ablate_heads:
                activation[batch_idx, token_pos, ablate_heads, :] = mean_act[
                    token_pos, ablate_heads, :
                ]
    return activation


def evaluate_metrics(
    *,
    model,
    dataloader,
    device: Any,
    hook_names: Optional[Sequence[str]] = None,
    circuit_by_position: Optional[Mapping[str, Mapping[str, Sequence[int]]]] = None,
    mean_activations: Optional[Mapping[str, Any]] = None,
    ablate_non_circuit_positions: bool = True,
    preserve_all_heads_at_circuit_positions: bool = False,
):
    import torch
    from functools import partial
    from tqdm import tqdm

    metrics = new_metrics()
    use_hooks = bool(hook_names)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluate"):
            batch = move_batch_to_device(batch, device)
            if use_hooks:
                hooks = [
                    (
                        name,
                        partial(
                            mean_ablate_activation,
                            batch=batch,
                            circuit_by_position=circuit_by_position,
                            mean_activations=mean_activations,
                            ablate_non_circuit_positions=ablate_non_circuit_positions,
                            preserve_all_heads_at_circuit_positions=preserve_all_heads_at_circuit_positions,
                        ),
                    )
                    for name in hook_names
                ]
                logits = model.run_with_hooks(
                    batch["base_tokens"],
                    fwd_hooks=hooks,
                    return_type="logits",
                )
            else:
                logits = model(batch["base_tokens"], return_type="logits")

            for batch_idx in range(int(batch["labels"].shape[0])):
                last_idx = int(batch["base_last_token_indices"][batch_idx].item())
                label = int(batch["labels"][batch_idx].item())
                candidates = (
                    batch["candidate_labels"][batch_idx]
                    if "candidate_labels" in batch
                    else None
                )
                label_aliases = (
                    batch["label_aliases"][batch_idx]
                    if "label_aliases" in batch
                    else None
                )
                candidate_aliases = (
                    batch["candidate_alias_labels"][batch_idx]
                    if "candidate_alias_labels" in batch
                    else None
                )
                update_metrics(
                    metrics,
                    logits[batch_idx, last_idx],
                    label,
                    candidates,
                    label_aliases=label_aliases,
                    candidate_aliases=candidate_aliases,
                )
            del logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return finalize_metrics(metrics)


def metrics_mean(metrics_list: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [m.get(key) for m in metrics_list if m.get(key) is not None]
    return float(statistics.mean(values)) if values else None


def metrics_std(metrics_list: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    values = [m.get(key) for m in metrics_list if m.get(key) is not None]
    return float(statistics.pstdev(values)) if values else None
