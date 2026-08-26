from __future__ import annotations

from exp1.trace_contexts import (
    _get_act_name,
    _model_device,
    _move_batch_to_device,
)

from datasets.common import lookup_answer_logits
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import torch

def _candidate_prediction_summary(answer_logits: torch.Tensor, answer_tokens: torch.Tensor, target_idx: torch.Tensor):
    pred_idx = answer_logits.argmax(dim=-1)
    batch_indices = torch.arange(answer_logits.shape[0], device=answer_logits.device)
    pred_tokens = answer_tokens[batch_indices, pred_idx]
    target_tokens = answer_tokens[batch_indices, target_idx]
    target_log_probs = answer_logits[batch_indices, target_idx]
    return pred_idx, pred_tokens, target_tokens, target_log_probs


def find_correct_samples(
    model,
    dataset: TracePromptDataset,
    batch_size: int,
    num_samples: int,
    verbose: bool = False,
) -> List[int]:
    device = _model_device(model)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tokenizer = model.tokenizer

    correct_indices: List[int] = []
    seen = 0
    for batch in tqdm(dataloader, desc="Finding correct samples"):
        batch = _move_batch_to_device(batch, device)
        with torch.no_grad():
            clean_logits = model(batch["clean_tokens"], return_type="logits")
            source_logits = model(batch["source_tokens"], return_type="logits")

        source_answer_logits = lookup_answer_logits(
            source_logits,
            batch["source_answer_tokens"],
            query_position=batch["source_query_positions"],
        )
        clean_answer_logits = lookup_answer_logits(
            clean_logits,
            batch["source_answer_tokens"],
            query_position=batch["clean_query_positions"],
        )

        query_names = batch.get("source_query_names", batch["query_names"])
        clean_pred_idx, clean_pred_tokens, clean_target_tokens, _ = _candidate_prediction_summary(
            clean_answer_logits,
            batch["source_answer_tokens"],
            query_names,
        )
        source_pred_idx, source_pred_tokens, source_target_tokens, _ = _candidate_prediction_summary(
            source_answer_logits,
            batch["source_answer_tokens"],
            query_names,
        )

        source_ok = source_pred_idx.eq(query_names)

        if verbose:
            for i in range(int(query_names.shape[0])):
                clean_pred = tokenizer.decode([int(clean_pred_tokens[i].item())]).strip()
                clean_target = tokenizer.decode([int(clean_target_tokens[i].item())]).strip()
                source_pred = tokenizer.decode([int(source_pred_tokens[i].item())]).strip()
                source_target = tokenizer.decode([int(source_target_tokens[i].item())]).strip()
                print(
                    f"clean pred={clean_pred!r} target={clean_target!r} | "
                    f"source pred={source_pred!r} target={source_target!r}"
                )

        batch_size_actual = int(query_names.shape[0])
        for i in range(batch_size_actual):
            if bool(source_ok[i].item()):
                correct_indices.append(seen + i)
                if len(correct_indices) >= num_samples:
                    return correct_indices[:num_samples]
        seen += batch_size_actual

    return correct_indices[:num_samples]


def _patch_resid_pre_at_token(
    resid_pre: torch.Tensor,
    hook,
    *,
    source_cache,
    token_index: int,
    valid_mask: torch.Tensor,
):
    valid_mask = valid_mask.to(resid_pre.device)
    if valid_mask.any():
        resid_pre = resid_pre.clone()
        resid_pre[valid_mask, token_index, :] = source_cache[hook.name][valid_mask, token_index, :].to(
            resid_pre.device
        )
    return resid_pre


def _init_nested_scores(token_axis: Iterable[int], layer_axis: Iterable[int]):
    accuracy_sum = {int(token): {int(layer): 0.0 for layer in layer_axis} for token in token_axis}
    prob_sum = {int(token): {int(layer): 0.0 for layer in layer_axis} for token in token_axis}
    count_sum = {int(token): {int(layer): 0 for layer in layer_axis} for token in token_axis}
    return accuracy_sum, prob_sum, count_sum


def _build_result_payload(
    *,
    metadata: Dict[str, object],
    token_axis: List[int],
    layer_axis: List[int],
    accuracy_sum,
    prob_sum,
    count_sum,
    reference_batch: Optional[Dict[str, object]],
):
    scores = {}
    for token in token_axis:
        token_key = str(int(token))
        scores[token_key] = {}
        for layer in layer_axis:
            count = int(count_sum[int(token)][int(layer)])
            accuracy = (
                float(accuracy_sum[int(token)][int(layer)] / count) if count > 0 else None
            )
            mean_target_prob = (
                float(prob_sum[int(token)][int(layer)] / count) if count > 0 else None
            )
            scores[token_key][str(int(layer))] = {
                "accuracy": accuracy,
                "mean_target_prob": mean_target_prob,
                "count": count,
            }

    token_reference = {}
    if reference_batch is not None:
        clean_tokens = reference_batch["clean_tokens"]
        source_tokens = reference_batch["source_tokens"]
        clean_query_positions = reference_batch["clean_query_positions"]
        source_query_positions = reference_batch["source_query_positions"]
        tokenizer = reference_batch["tokenizer"]
        for token in token_axis:
            token_idx = int(token)
            valid = (
                (clean_query_positions >= token_idx)
                & (source_query_positions >= token_idx)
            ).nonzero(as_tuple=False)
            if valid.numel() == 0:
                continue
            sample_idx = int(valid[0].item())
            token_reference[str(token_idx)] = {
                "clean_token": tokenizer.decode(
                    [int(clean_tokens[sample_idx, token_idx].item())],
                    clean_up_tokenization_spaces=False,
                ),
                "source_token": tokenizer.decode(
                    [int(source_tokens[sample_idx, token_idx].item())],
                    clean_up_tokenization_spaces=False,
                ),
            }

    payload = {
        "metadata": metadata,
        "token_axis": token_axis,
        "layer_axis": layer_axis,
        "scores": scores,
        "token_reference": token_reference,
    }
    if reference_batch is not None:
        payload["examples"] = {
            "clean_prompt": reference_batch["clean_prompts"][0],
            "source_prompt": reference_batch["source_prompts"][0],
        }
    return payload


def save_tracing_results(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_tracing_experiment(
    *,
    model,
    dataset: TracePromptDataset,
    tracing_batch_size: int,
    results_path: Path,
    metadata: Dict[str, object],
    start_token: Optional[int] = None,
    end_token: int = 0,
    start_layer: int = 0,
    layer_step: int = 1,
    verbose: bool = False,
):
    device = _model_device(model)
    dataloader = DataLoader(dataset, batch_size=tracing_batch_size, shuffle=False)
    layer_axis = list(range(start_layer, model.cfg.n_layers, layer_step))

    valid_token_limits = torch.minimum(
        dataset.payload["clean_query_positions"],
        dataset.payload["source_query_positions"],
    )
    if start_token is None:
        start_token = int(valid_token_limits.max().item())
    token_axis = list(range(int(start_token), int(end_token) - 1, -1))

    accuracy_sum, prob_sum, count_sum = _init_nested_scores(token_axis, layer_axis)
    reference_batch = None

    for batch in tqdm(dataloader, desc="Tracing batches"):
        batch = _move_batch_to_device(batch, device)
        if reference_batch is None:
            reference_batch = {
                "clean_tokens": batch["clean_tokens"].detach().cpu(),
                "source_tokens": batch["source_tokens"].detach().cpu(),
                "clean_query_positions": batch["clean_query_positions"].detach().cpu(),
                "source_query_positions": batch["source_query_positions"].detach().cpu(),
                "clean_prompts": list(batch["clean_prompts"]),
                "source_prompts": list(batch["source_prompts"]),
                "tokenizer": model.tokenizer,
            }

        with torch.no_grad():
            _, source_cache = model.run_with_cache(
                batch["source_tokens"],
                names_filter=lambda name: "hook_resid_pre" in name,
                return_type="logits",
            )

            for token_idx in token_axis:
                valid_mask = (
                    (batch["clean_query_positions"] >= token_idx)
                    & (batch["source_query_positions"] >= token_idx)
                )
                valid_count = int(valid_mask.sum().item())
                if valid_count == 0:
                    continue

                for layer_idx in layer_axis:
                    hook_name = _get_act_name("resid_pre", layer_idx)
                    patched_logits = model.run_with_hooks(
                        batch["clean_tokens"],
                        fwd_hooks=[
                            (
                                hook_name,
                                lambda resid_pre, hook, source_cache=source_cache, token_idx=token_idx, valid_mask=valid_mask: _patch_resid_pre_at_token(
                                    resid_pre,
                                    hook,
                                    source_cache=source_cache,
                                    token_index=token_idx,
                                    valid_mask=valid_mask,
                                ),
                            )
                        ],
                        return_type="logits",
                    )

                    answer_logits = lookup_answer_logits(
                        patched_logits,
                        batch["source_answer_tokens"],
                        query_position=batch["clean_query_positions"],
                    )
                    query_names = batch.get("source_query_names", batch["query_names"])
                    pred_idx, _, _, target_log_probs = _candidate_prediction_summary(
                        answer_logits,
                        batch["source_answer_tokens"],
                        query_names,
                    )
                    correct_mask = pred_idx.eq(query_names) & valid_mask
                    accuracy_sum[token_idx][layer_idx] += float(correct_mask.sum().item())
                    prob_sum[token_idx][layer_idx] += float(
                        target_log_probs.exp()[valid_mask].sum().item()
                    )
                    count_sum[token_idx][layer_idx] += valid_count

                    if verbose:
                        acc = accuracy_sum[token_idx][layer_idx] / count_sum[token_idx][layer_idx]
                        print(
                            f"token={token_idx:>3} layer={layer_idx:>2} "
                            f"batch_valid={valid_count:>2} running_acc={acc:.4f}"
                        )

                payload = _build_result_payload(
                    metadata=metadata,
                    token_axis=token_axis,
                    layer_axis=layer_axis,
                    accuracy_sum=accuracy_sum,
                    prob_sum=prob_sum,
                    count_sum=count_sum,
                    reference_batch=reference_batch,
                )
                save_tracing_results(results_path, payload)

    final_payload = _build_result_payload(
        metadata=metadata,
        token_axis=token_axis,
        layer_axis=layer_axis,
        accuracy_sum=accuracy_sum,
        prob_sum=prob_sum,
        count_sum=count_sum,
        reference_batch=reference_batch,
    )
    save_tracing_results(results_path, final_payload)
    return final_payload
