from __future__ import annotations

from exp4.bid_data import (
    aggregate_deltas_by_kv_head,
    group_heads_by_layer,
    k_hook_name,
    key_value_head_for_attention_head,
    layer_from_hook_name,
    q_hook_name,
)

from collections import defaultdict
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

def hook_names_for_heads(heads: Sequence[Head], hook_fn) -> List[str]:
    return sorted({hook_fn(layer) for layer, _ in heads})


def cache_forward(
    model,
    tokens: torch.Tensor,
    *,
    cache_hook_names: Sequence[str],
    fwd_hooks: Optional[Sequence[Tuple[str, object]]] = None,
):
    hook_name_set = set(cache_hook_names)
    cache, cache_hooks, _ = model.get_caching_hooks(
        names_filter=lambda name: name in hook_name_set,
        incl_bwd=False,
        device=None,
        remove_batch_dim=False,
    )
    hooks = list(fwd_hooks or []) + list(cache_hooks)
    with model.hooks(fwd_hooks=hooks):
        logits = model(tokens, return_type="logits")
    return logits, cache


def batched_position_gather(
    activation: torch.Tensor,
    positions: torch.Tensor,
    head: int,
) -> torch.Tensor:
    batch_indices = torch.arange(activation.shape[0], device=activation.device)
    return activation[
        batch_indices,
        positions.to(activation.device).to(torch.long),
        int(head),
        :,
    ]


def estimate_binding_id_directions(
    *,
    model,
    dataloaders_by_source_slot: Mapping[int, DataLoader],
    heads: Sequence[Head],
    source_position_kinds: Sequence[str],
    device: torch.device,
    label: str,
    k_position_fields: Sequence[str] = ("query_box_positions",),
) -> Dict[str, object]:
    heads_by_layer = group_heads_by_layer(heads)
    logical_n_heads = int(getattr(model.cfg, "n_heads", 0)) or max(
        int(head) for _, head in heads
    ) + 1
    cache_hook_names = sorted(
        set(hook_names_for_heads(heads, q_hook_name))
        | set(hook_names_for_heads(heads, k_hook_name))
    )
    q_means: Dict[int, TensorByHead] = {}
    k_means: Dict[int, TensorByHead] = {}
    q_counts: Dict[int, Dict[str, int]] = {}
    k_counts: Dict[int, Dict[str, int]] = {}
    k_position_means: Dict[str, Dict[int, TensorByHead]] = {
        field: {} for field in k_position_fields
    }
    k_position_counts: Dict[str, Dict[int, Dict[str, int]]] = {
        field: {} for field in k_position_fields
    }

    model.eval()
    with torch.no_grad():
        for source_slot, dataloader in dataloaders_by_source_slot.items():
            q_sums: TensorByHead = {}
            k_sums: TensorByHead = {}
            q_count_by_head: Dict[Head, int] = defaultdict(int)
            k_count_by_head: Dict[Head, int] = defaultdict(int)
            k_position_sums: Dict[str, TensorByHead] = {
                field: {} for field in k_position_fields
            }
            k_position_count_by_head: Dict[str, Dict[Head, int]] = {
                field: defaultdict(int) for field in k_position_fields
            }
            for batch in tqdm(
                dataloader,
                desc=f"Estimate {label} Q/K ID slot={source_slot}",
            ):
                batch = move_batch_to_device(batch, device)
                _, cache = cache_forward(
                    model,
                    batch["base_tokens"],
                    cache_hook_names=cache_hook_names,
                )
                for layer, layer_heads in heads_by_layer.items():
                    q = cache[q_hook_name(layer)]
                    k = cache[k_hook_name(layer)]
                    answer_positions = batch["base_answer_colon_positions"]
                    for head in layer_heads:
                        key = (layer, head)
                        q_values = batched_position_gather(q, answer_positions, head)
                        q_sums[key] = q_sums.get(
                            key,
                            torch.zeros(q_values.shape[-1], dtype=torch.float32),
                        ) + q_values.detach().float().sum(dim=0).cpu()
                        q_count_by_head[key] += int(q_values.shape[0])

                        k_head = key_value_head_for_attention_head(
                            head,
                            logical_n_heads=logical_n_heads,
                            activation_n_heads=int(k.shape[2]),
                        )
                        for kind in source_position_kinds:
                            positions = batch[f"{kind}_token_positions"][
                                :,
                                int(source_slot),
                            ]
                            k_values = batched_position_gather(k, positions, k_head)
                            k_sums[key] = k_sums.get(
                                key,
                                torch.zeros(k_values.shape[-1], dtype=torch.float32),
                            ) + k_values.detach().float().sum(dim=0).cpu()
                            k_count_by_head[key] += int(k_values.shape[0])
                        for position_field in k_position_fields:
                            positions = batch[position_field]
                            k_values = batched_position_gather(k, positions, k_head)
                            k_position_sums[position_field][key] = (
                                k_position_sums[position_field].get(
                                    key,
                                    torch.zeros(
                                        k_values.shape[-1],
                                        dtype=torch.float32,
                                    ),
                                )
                                + k_values.detach().float().sum(dim=0).cpu()
                            )
                            k_position_count_by_head[position_field][key] += int(
                                k_values.shape[0]
                            )
                del cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            q_means[int(source_slot)] = {
                key: q_sums[key] / max(q_count_by_head[key], 1)
                for key in heads
            }
            k_means[int(source_slot)] = {
                key: k_sums[key] / max(k_count_by_head[key], 1)
                for key in heads
            }
            q_counts[int(source_slot)] = {
                f"L{layer}H{head}": int(q_count_by_head[(layer, head)])
                for layer, head in heads
            }
            k_counts[int(source_slot)] = {
                f"L{layer}H{head}": int(k_count_by_head[(layer, head)])
                for layer, head in heads
            }
            for position_field in k_position_fields:
                k_position_means[position_field][int(source_slot)] = {
                    key: (
                        k_position_sums[position_field][key]
                        / max(k_position_count_by_head[position_field][key], 1)
                    )
                    for key in heads
                }
                k_position_counts[position_field][int(source_slot)] = {
                    f"L{layer}H{head}": int(
                        k_position_count_by_head[position_field][(layer, head)]
                    )
                    for layer, head in heads
                }

    return {
        "q_means": q_means,
        "k_means": k_means,
        "q_counts": q_counts,
        "k_counts": k_counts,
        "k_position_means": k_position_means,
        "k_position_counts": k_position_counts,
    }


def delta_between(means: Mapping[int, TensorByHead], from_slot: int, to_slot: int) -> TensorByHead:
    return {
        head: means[int(to_slot)][head] - means[int(from_slot)][head]
        for head in means[int(from_slot)]
    }


def random_matched_deltas(
    reference: Mapping[Head, torch.Tensor],
    *,
    seed: int,
) -> TensorByHead:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    random_deltas: TensorByHead = {}
    for head, ref in sorted(reference.items()):
        ref_float = ref.detach().float().cpu()
        random_vec = torch.randn(
            ref_float.shape,
            generator=generator,
            dtype=torch.float32,
        )
        ref_norm = torch.linalg.vector_norm(ref_float)
        random_norm = torch.linalg.vector_norm(random_vec).clamp_min(1e-12)
        random_deltas[head] = random_vec * (ref_norm / random_norm)
    return random_deltas


def matched_norm_directions(
    directions: Mapping[Head, torch.Tensor],
    *,
    reference: Mapping[Head, torch.Tensor],
) -> TensorByHead:
    matched: TensorByHead = {}
    for head, direction in sorted(directions.items()):
        direction_float = direction.detach().float().cpu()
        ref_float = reference[head].detach().float().cpu()
        direction_norm = torch.linalg.vector_norm(direction_float).clamp_min(1e-12)
        ref_norm = torch.linalg.vector_norm(ref_float)
        matched[head] = direction_float * (ref_norm / direction_norm)
    return matched


def negate_deltas(deltas: Mapping[Head, torch.Tensor]) -> TensorByHead:
    return {head: -delta.detach().float().cpu() for head, delta in deltas.items()}


def make_q_patch_hook(
    *,
    heads_by_layer: Mapping[int, Sequence[int]],
    deltas: Mapping[Head, torch.Tensor],
    alpha: float,
    position_field: str,
    batch: Mapping[str, torch.Tensor],
):
    def patch_q(q: torch.Tensor, hook):
        layer = layer_from_hook_name(hook.name)
        layer_heads = heads_by_layer.get(layer, ())
        if not layer_heads:
            return q
        positions = batch[position_field].to(q.device).to(torch.long)
        batch_indices = torch.arange(q.shape[0], device=q.device)
        for head in layer_heads:
            delta = deltas[(layer, int(head))].to(q.device, dtype=q.dtype)
            q[batch_indices, positions, int(head), :] += float(alpha) * delta
        return q

    return patch_q


def make_k_patch_hook(
    *,
    heads_by_layer: Mapping[int, Sequence[int]],
    entity_deltas: Optional[Mapping[int, Mapping[Head, torch.Tensor]]],
    position_field_deltas: Optional[Mapping[str, Mapping[Head, torch.Tensor]]],
    alpha: float,
    source_position_kinds: Sequence[str],
    batch: Mapping[str, torch.Tensor],
    logical_n_heads: int,
):
    def patch_k(k: torch.Tensor, hook):
        layer = layer_from_hook_name(hook.name)
        layer_heads = heads_by_layer.get(layer, ())
        if not layer_heads:
            return k
        batch_indices = torch.arange(k.shape[0], device=k.device)

        def add_delta_at_positions(
            positions: torch.Tensor,
            *,
            kv_head: int,
            delta: torch.Tensor,
        ) -> None:
            positions = positions.to(k.device).to(torch.long)
            delta = delta.to(k.device, dtype=k.dtype)
            if positions.ndim == 1:
                k[batch_indices, positions, int(kv_head), :] += float(alpha) * delta
                return
            expanded_batch_indices = batch_indices.view(
                (k.shape[0],) + (1,) * (positions.ndim - 1)
            ).expand_as(positions)
            k[
                expanded_batch_indices.reshape(-1),
                positions.reshape(-1),
                int(kv_head),
                :,
            ] += float(alpha) * delta

        for entity, deltas in (entity_deltas or {}).items():
            for kind in source_position_kinds:
                positions = batch[f"{kind}_token_positions"][
                    :,
                    int(entity),
                ]
                kv_deltas = aggregate_deltas_by_kv_head(
                    layer_heads=layer_heads,
                    deltas=deltas,
                    layer=layer,
                    logical_n_heads=logical_n_heads,
                    activation_n_heads=int(k.shape[2]),
                )
                for kv_head, delta in kv_deltas.items():
                    add_delta_at_positions(positions, kv_head=kv_head, delta=delta)
        for position_field, deltas in (position_field_deltas or {}).items():
            positions = batch[position_field]
            kv_deltas = aggregate_deltas_by_kv_head(
                layer_heads=layer_heads,
                deltas=deltas,
                layer=layer,
                logical_n_heads=logical_n_heads,
                activation_n_heads=int(k.shape[2]),
            )
            for kv_head, delta in kv_deltas.items():
                add_delta_at_positions(positions, kv_head=kv_head, delta=delta)
        return k

    return patch_k


def make_intervention_hooks(
    *,
    heads: Sequence[Head],
    q_deltas: Optional[Mapping[Head, torch.Tensor]],
    q_position_field: Optional[str],
    k_entity_deltas: Optional[Mapping[int, Mapping[Head, torch.Tensor]]],
    k_position_field_deltas: Optional[Mapping[str, Mapping[Head, torch.Tensor]]],
    source_position_kinds: Sequence[str],
    alpha: float,
    batch: Mapping[str, torch.Tensor],
    logical_n_heads: int,
) -> List[Tuple[str, object]]:
    hooks: List[Tuple[str, object]] = []
    heads_by_layer = group_heads_by_layer(heads)
    if q_deltas is not None:
        if q_position_field is None:
            raise ValueError("q_position_field is required when q_deltas are provided.")
        q_hook = make_q_patch_hook(
            heads_by_layer=heads_by_layer,
            deltas=q_deltas,
            alpha=alpha,
            position_field=q_position_field,
            batch=batch,
        )
        hooks.extend((q_hook_name(layer), q_hook) for layer in sorted(heads_by_layer))
    if k_entity_deltas is not None or k_position_field_deltas is not None:
        k_hook = make_k_patch_hook(
            heads_by_layer=heads_by_layer,
            entity_deltas=k_entity_deltas,
            position_field_deltas=k_position_field_deltas,
            alpha=alpha,
            source_position_kinds=source_position_kinds,
            batch=batch,
            logical_n_heads=logical_n_heads,
        )
        hooks.extend((k_hook_name(layer), k_hook) for layer in sorted(heads_by_layer))
    return hooks
