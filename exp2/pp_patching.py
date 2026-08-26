from __future__ import annotations

from exp2.pp_prompts import (
    _attn_hook_name,
    _clean_answer,
    _clean_prompt,
    _encode_prompts,
    _encode_single_token_answer,
    _layer_from_hook_name,
    _model_num_heads,
    _model_num_layers,
    _move_batch_to_device,
    _read_jsonl_rows,
    _receiver_key,
    _z_hook_name,
)

from exp2.pp_corruptions import (
    _build_dataset_from_rows,
    _coerce_in_memory_dataset,
)

from exp2.pp_models import (
    compute_prev_query_box_pos,
    compute_topk_components,
)

from collections import defaultdict
from datasets import Dataset
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import numpy as np
import random
import torch

def load_dataloader(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    datafile: str = None,
    num_samples: int = None,
    num_boxes: int = 7,
    batch_size: int = 1,
    data_rows=None,
):
    """
    Loads clean/corrupt prompt pairs from a JSONL datafile.

    Args:
        model: model to use for tokenization compatibility.
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        num_boxes: unused legacy argument, retained for API compatibility.
        batch_size: batch size to use for the dataloader.
        data_rows: optional in-memory training data. This may be a DataLoader,
            Dataset, list of dict rows, or pretokenized mapping with
            base/source tokens and labels. If only clean/base prompts or clean
            tokens are provided, counterfactual prompts are synthesized from the
            anchor prompts using other generated samples as donors.
    """

    del model, num_boxes

    if isinstance(data_rows, DataLoader):
        return data_rows

    if data_rows is not None:
        dataset = _coerce_in_memory_dataset(tokenizer, data_rows, num_samples)
    else:
        if datafile is None:
            raise ValueError("Either `datafile` or `data_rows` must be provided.")
        rows = _read_jsonl_rows(datafile, num_samples)
        dataset = _build_dataset_from_rows(tokenizer, rows)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_caches(
    model: HookedTransformer,
    dataloader: torch.utils.data.DataLoader,
    hook_points: Optional[Sequence[str]] = None,
):
    """
    Computes the clean and corrupt caches for the model.

    Args:
        model: model to compute the caches for.
        dataloader: dataloader containing clean and corrupt inputs.
    """

    if hook_points is None:
        hook_points = [_z_hook_name(layer) for layer in range(_model_num_layers(model))]
    hook_name_set = set(hook_points)

    apply_softmax = torch.nn.Softmax(dim=-1)
    clean_cache, corrupt_cache = {}, {}
    clean_logit_outputs, corrupt_logit_outputs = defaultdict(dict), defaultdict(dict)

    with torch.no_grad():
        for bi, inp in tqdm(enumerate(dataloader), desc="Clean cache"):
            batch_size = inp["base_tokens"].size(0)
            inp = _move_batch_to_device(inp, model)

            logits, cache = model.run_with_cache(
                inp["base_tokens"],
                names_filter=lambda name: name in hook_name_set,
                return_type="logits",
            )

            clean_cache[bi] = {
                hook_name: cache[hook_name].detach().cpu() for hook_name in hook_points
            }

            for i in range(batch_size):
                logits_i = apply_softmax(
                    logits[i, int(inp["base_last_token_indices"][i].item())]
                )
                clean_logit_outputs[bi][i] = logits_i[int(inp["labels"][i].item())].item()

            del logits, cache
            torch.cuda.empty_cache()
        print("CLEAN CACHE COMPUTED")

        for bi, inp in tqdm(enumerate(dataloader), desc="Corrupt cache"):
            batch_size = inp["source_tokens"].size(0)
            inp = _move_batch_to_device(inp, model)

            logits, cache = model.run_with_cache(
                inp["source_tokens"],
                names_filter=lambda name: name in hook_name_set,
                return_type="logits",
            )

            corrupt_cache[bi] = {
                hook_name: cache[hook_name].detach().cpu() for hook_name in hook_points
            }

            for i in range(batch_size):
                logits_i = apply_softmax(
                    logits[i, int(inp["source_last_token_indices"][i].item())]
                )
                corrupt_logit_outputs[bi][i] = logits_i[int(inp["labels"][i].item())].item()

            del logits, cache
            torch.cuda.empty_cache()
        print("CORRUPT CACHE COMPUTED")

    return (
        clean_cache,
        corrupt_cache,
        clean_logit_outputs,
        corrupt_logit_outputs,
        hook_points,
    )


def patching_sender_heads(
    z: torch.Tensor,
    hook,
    model: HookedTransformer = None,
    clean_cache: Dict[str, torch.Tensor] = None,
    corrupt_cache: Dict[str, torch.Tensor] = None,
    base_tokens: torch.Tensor = None,
    sender_layer: int = None,
    sender_head: int = None,
    clean_last_token_indices: torch.Tensor = None,
    corrupt_last_token_indices: torch.Tensor = None,
    clean_positions: torch.Tensor = None,
    corrupt_positions: torch.Tensor = None,
    rel_pos: int = None,
    batch_size: int = None,
):
    """
    Patches head outputs at `hook_z` and stores the receiver activations via
    `run_with_cache` in the caller.
    """

    del model, clean_cache

    corrupt_head_outputs = corrupt_cache[hook.name].to(z.device)
    layer_idx = _layer_from_hook_name(hook.name)

    for bi in range(batch_size):
        clean_last_idx = int(clean_last_token_indices[bi].item())

        if clean_positions is not None and corrupt_positions is not None:
            sender_head_pos_in_clean = int(clean_positions[bi].item())
            sender_head_pos_in_corrupt = int(corrupt_positions[bi].item())
        elif rel_pos == -1:
            sender_head_pos_in_clean = compute_prev_query_box_pos(
                base_tokens[bi], clean_last_idx
            )
            sender_head_pos_in_corrupt = random.choice(range(6, 49, 7))
        else:
            sender_head_pos_in_clean = clean_last_idx - rel_pos
            sender_head_pos_in_corrupt = int(corrupt_last_token_indices[bi].item()) - rel_pos

        if layer_idx == sender_layer:
            z[bi, sender_head_pos_in_clean, sender_head] = corrupt_head_outputs[
                bi, sender_head_pos_in_corrupt, sender_head
            ]

    return z


def patching_receiver_heads(
    act: torch.Tensor,
    hook,
    model: HookedTransformer = None,
    base_tokens: torch.Tensor = None,
    patched_cache=None,
    receiver_heads: List[Tuple[int, int]] = None,
    clean_last_token_indices: torch.Tensor = None,
    clean_positions: torch.Tensor = None,
    rel_pos: int = None,
    batch_size: int = None,
):
    """
    Patches the input of the receiver head, i.e. key, query or value vectors.
    """

    current_layer = _layer_from_hook_name(hook.name)
    receiver_heads_in_curr_layer = [
        head_idx for layer_idx, head_idx in receiver_heads if layer_idx == current_layer
    ]
    if not receiver_heads_in_curr_layer:
        return act

    is_kv_receiver = hook.name.endswith("hook_k") or hook.name.endswith("hook_v")
    model_num_heads = int(getattr(model.cfg, "n_heads", act.shape[2]))
    model_num_kv_heads = int(
        getattr(
            model.cfg,
            "n_key_value_heads",
            getattr(model.cfg, "n_kv_heads", act.shape[2]),
        )
    )
    if is_kv_receiver and act.shape[2] != model_num_heads and model_num_kv_heads > 0:
        kv_group_size = max(1, model_num_heads // model_num_kv_heads)
        receiver_heads_in_curr_layer = sorted(
            {
                min(model_num_kv_heads - 1, head_idx // kv_group_size)
                for head_idx in receiver_heads_in_curr_layer
            }
        )

    patched_head_outputs = patched_cache[hook.name]
    if patched_head_outputs.device != act.device:
        patched_head_outputs = patched_head_outputs.to(act.device)

    for receiver_head in receiver_heads_in_curr_layer:
        for bi in range(batch_size):
            clean_last_idx = int(clean_last_token_indices[bi].item())
            if clean_positions is not None:
                receiver_head_pos_in_clean = int(clean_positions[bi].item())
            elif rel_pos == -1:
                receiver_head_pos_in_clean = compute_prev_query_box_pos(
                    base_tokens[bi], clean_last_idx
                )
            else:
                receiver_head_pos_in_clean = clean_last_idx - rel_pos

            act[bi, receiver_head_pos_in_clean, receiver_head] = patched_head_outputs[
                bi, receiver_head_pos_in_clean, receiver_head
            ]

    return act


def get_receiver_layers(
    model: HookedTransformer, receiver_heads: List[Tuple[int, int]], composition: str
):
    """
    Gets the receiver hook names from the receiver heads.

    Args:
        model: model under investigation.
        receiver_heads: receiver heads.
        composition: attention component to use for the receiver heads (k/q/v).
    """

    del model

    return sorted(
        {
            _attn_hook_name(composition, layer_idx)
            for layer_idx, _ in receiver_heads
        }
    )


def load_eval_data(
    tokenizer: PreTrainedTokenizerBase, datafile: str, num_samples: int, batch_size: int
):
    """
    Loads the dataset for evaluation.
    """

    print("Loading dataset...")
    rows = _read_jsonl_rows(datafile, num_samples)
    prompts = [_clean_prompt(row) for row in rows]
    labels = [_encode_single_token_answer(tokenizer, _clean_answer(row)) for row in rows]
    input_ids, last_token_indices = _encode_prompts(tokenizer, prompts)

    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "last_token_indices": last_token_indices,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    ).with_format("torch")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_ablation_data(
    tokenizer: PreTrainedTokenizerBase,
    datafile: str,
    num_samples: int,
    batch_size: int,
    num_boxes: int = 7,
):
    """
    Loads the dataset for ablation.
    """

    del num_boxes

    rows = _read_jsonl_rows(datafile, num_samples)
    prompts = [_clean_prompt(row) for row in rows]
    input_ids, last_token_indices = _encode_prompts(tokenizer, prompts)

    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "last_token_indices": last_token_indices,
        }
    ).with_format("torch")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_mean_activations(
    model: HookedTransformer,
    tokenizer: PreTrainedTokenizerBase,
    datafile: str,
    num_samples: int,
    batch_size: int,
):
    """
    Computes the mean activations of every attention head at all positions.
    """

    print("Computing mean activations...")
    ablation_dataloader = load_ablation_data(
        tokenizer=tokenizer,
        datafile=datafile,
        num_samples=num_samples,
        batch_size=batch_size,
    )

    modules = [_z_hook_name(layer) for layer in range(_model_num_layers(model))]
    module_set = set(modules)

    mean_activations: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for inp in tqdm(ablation_dataloader):
            inp = _move_batch_to_device(inp, model)

            _, cache = model.run_with_cache(
                inp["input_ids"],
                names_filter=lambda name: name in module_set,
                return_type="logits",
            )

            for layer in modules:
                layer_acts = cache[layer].detach().cpu()
                if layer in mean_activations:
                    mean_activations[layer] += torch.sum(layer_acts, dim=0)
                else:
                    mean_activations[layer] = torch.sum(layer_acts, dim=0)

            del cache
            torch.cuda.empty_cache()

        for layer in modules:
            mean_activations[layer] /= len(ablation_dataloader.dataset)

    return mean_activations, modules


def mean_ablate(
    z: torch.Tensor,
    hook,
    model: HookedTransformer = None,
    circuit_components: Dict[int, Dict[str, List[int]]] = None,
    mean_activations: Dict[str, torch.Tensor] = None,
    input_tokens: torch.Tensor = None,
    ablate_non_vital_pos: bool = None,
):
    """
    Ablates the model components that are not present in `circuit_components`
    by substituting their output with their corresponding mean activations.
    """

    mean_act = mean_activations[hook.name].to(z.device)
    pad_token_id = getattr(model.tokenizer, "pad_token_id", None)

    for bi in range(z.size(0)):
        if pad_token_id is None:
            last_pos = z.size(1) - 1
        else:
            non_pad = (input_tokens[bi] != pad_token_id).nonzero(as_tuple=False)
            last_pos = int(non_pad[-1].item()) if non_pad.numel() else z.size(1) - 1

        prev_query_box_pos = compute_prev_query_box_pos(input_tokens[bi], last_pos)

        for token_pos in range(last_pos + 1):
            if (
                token_pos != prev_query_box_pos
                and token_pos != last_pos - 2
                and token_pos != last_pos
                and ablate_non_vital_pos
            ):
                z[bi, token_pos, :] = mean_act[token_pos, :]
            elif token_pos == prev_query_box_pos:
                for head_idx in range(_model_num_heads(model)):
                    if head_idx not in circuit_components[-1][hook.name]:
                        z[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]
            elif token_pos == last_pos - 2:
                for head_idx in range(_model_num_heads(model)):
                    if head_idx not in circuit_components[2][hook.name]:
                        z[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]
            elif token_pos == last_pos:
                for head_idx in range(_model_num_heads(model)):
                    if head_idx not in circuit_components[0][hook.name]:
                        z[bi, token_pos, head_idx] = mean_act[token_pos, head_idx]

    return z


def eval_circuit_performance(
    model: HookedTransformer,
    dataloader: torch.utils.data.DataLoader,
    modules: List[str],
    circuit_components: Dict[int, Dict[str, List[int]]],
    mean_activations: Dict[str, torch.Tensor],
    ablate_non_vital_pos: bool = True,
):
    """
    Evaluates the performance of the model/circuit.
    """

    correct_count, total_count = 0, 0
    hooks = [
        (
            module_name,
            partial(
                mean_ablate,
                model=model,
                circuit_components=circuit_components,
                mean_activations=mean_activations,
                ablate_non_vital_pos=ablate_non_vital_pos,
            ),
        )
        for module_name in modules
    ]

    with torch.no_grad():
        for inp in tqdm(dataloader):
            inp = _move_batch_to_device(inp, model)

            outputs = model.run_with_hooks(
                inp["input_ids"],
                fwd_hooks=[
                    (
                        module_name,
                        partial(
                            hook_fn,
                            input_tokens=inp["input_ids"],
                        ),
                    )
                    for module_name, hook_fn in hooks
                ],
                return_type="logits",
            )

            for bi in range(inp["labels"].size(0)):
                label = int(inp["labels"][bi].item())
                last_idx = int(inp["last_token_indices"][bi].item())
                pred = int(torch.argmax(outputs[bi, last_idx]).item())

                if label == pred:
                    correct_count += 1
                total_count += 1

            del outputs
            torch.cuda.empty_cache()

    return round(correct_count / total_count, 2)


def get_circuit(
    model: HookedTransformer,
    circuit_root_path: str,
    n_value_fetcher: int,
    n_pos_trans: int,
    n_pos_detect: int,
    n_struct_read: int,
):
    """
    Computes the circuit components.
    """

    circuit_components = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}

    value_fetcher_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "value_fetcher.pt"),
        k=n_value_fetcher,
        largest=False,
    )
    pos_transmitter_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "pos_transmitter.pt"),
        k=n_pos_trans,
        largest=False,
    )
    pos_detector_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "pos_detector.pt"),
        k=n_pos_detect,
        largest=False,
    )
    struct_reader_heads = compute_topk_components(
        torch.load(Path(circuit_root_path) / "struct_reader.pt"),
        k=n_struct_read,
        largest=False,
    )

    intersection = [head for head in value_fetcher_heads if head in pos_transmitter_heads]
    for head in intersection:
        value_fetcher_heads.remove(head)

    for layer_idx, head in value_fetcher_heads:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in pos_transmitter_heads:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in pos_detector_heads:
        circuit_components[2][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in struct_reader_heads:
        circuit_components[-1][_receiver_key(layer_idx)].append(head)

    return (
        circuit_components,
        value_fetcher_heads,
        pos_transmitter_heads,
        pos_detector_heads,
        struct_reader_heads,
    )


def get_random_circuit(
    model: HookedTransformer,
    circuit: Dict[str, List[Tuple[int, int]]],
):
    """
    Computes a random circuit with same #heads in each group as in the circuit.
    """

    random_circuit = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}

    num_heads = _model_num_heads(model)
    num_layers = _model_num_layers(model)
    n_value_fetcher = len(circuit["value_fetcher"])
    n_pos_transmitter = len(circuit["pos_transmitter"])
    n_pos_detector = len(circuit["pos_detector"])
    n_struct_reader = len(circuit["struct_reader"])

    heads_at_last_pos = np.random.choice(
        list(range(num_heads * num_layers)),
        n_value_fetcher + n_pos_transmitter,
        replace=False,
    )
    heads_at_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)),
        n_pos_detector,
        replace=False,
    )
    heads_at_prev_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)),
        n_struct_reader,
        replace=False,
    )

    heads_at_last_pos = [[idx // num_heads, idx % num_heads] for idx in heads_at_last_pos]
    heads_at_query_box_pos = [[idx // num_heads, idx % num_heads] for idx in heads_at_query_box_pos]
    heads_at_prev_query_box_pos = [
        [idx // num_heads, idx % num_heads] for idx in heads_at_prev_query_box_pos
    ]

    for layer_idx, head in heads_at_last_pos:
        random_circuit[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in heads_at_query_box_pos:
        random_circuit[2][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in heads_at_prev_query_box_pos:
        random_circuit[-1][_receiver_key(layer_idx)].append(head)

    return random_circuit


def compute_pair_drop_values(
    model: HookedTransformer,
    heads: List[Tuple[int, int]],
    circuit_components: Dict[int, Dict[str, List[int]]],
    dataloader: torch.utils.data.DataLoader,
    modules: List[str],
    mean_activations: Dict[str, torch.Tensor],
    rel_pos: int = 0,
):
    """
    Computes the pair drop values for the given heads.
    """

    greedy_res = defaultdict(lambda: defaultdict(float))

    for layer_idx_1, head_1 in tqdm(heads, total=len(heads), desc="Pair drop values"):
        layer_1 = _receiver_key(layer_idx_1)
        circuit_components[rel_pos][layer_1].remove(head_1)

        for layer_idx_2, head_2 in heads:
            layer_2 = _receiver_key(layer_idx_2)

            if greedy_res[(layer_2, head_2)][(layer_1, head_1)] > 0.0:
                continue
            if layer_1 != layer_2 or head_1 != head_2:
                circuit_components[rel_pos][layer_2].remove(head_2)

            greedy_res[(layer_1, head_1)][(layer_2, head_2)] = eval_circuit_performance(
                model, dataloader, modules, circuit_components, mean_activations
            )
            if layer_1 != layer_2 or head_1 != head_2:
                circuit_components[rel_pos][layer_2].append(head_2)

        circuit_components[rel_pos][layer_1].append(head_1)

    res = defaultdict(lambda: defaultdict(float))
    for k in greedy_res:
        for k_2 in greedy_res[k]:
            if greedy_res[k][k_2] > 0.0:
                res[str(k)][str(k_2)] = greedy_res[k][k_2]
                res[str(k_2)][str(k)] = greedy_res[k][k_2]

    return res


def get_head_significance_score(
    model: HookedTransformer,
    heads: List[Tuple[int, int]],
    ranked: Dict,
    percentage: float,
    circuit_components: Dict[int, Dict[str, List[int]]],
    dataloader: torch.utils.data.DataLoader,
    modules: List[str],
    mean_activations: Dict[str, torch.Tensor],
    rel_pos: int,
):
    """
    Computes the head significance score for the given heads.
    """

    res = {}

    for layer_idx, head in tqdm(heads, total=len(heads), desc="Head significance score"):
        layer = _receiver_key(layer_idx)

        for r in ranked[str((layer, head))][
            : math.ceil(percentage * len(ranked.values()))
        ]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].remove(top_head)

        before = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        circuit_components[rel_pos][layer].remove(head)
        after = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        res[(layer, head)] = (before, after)

        for r in ranked[str((layer, head))][
            : math.ceil(percentage * len(ranked.values()))
        ]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].append(top_head)
        circuit_components[rel_pos][layer].append(head)

    return res


def get_final_circuit(model: HookedTransformer, circuit_heads: Dict[str, List[Tuple[int, int]]]):
    """
    Computes the final circuit.
    """

    del model

    circuit_components = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}

    for layer_idx, head in circuit_heads["value_fetcher"]:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in circuit_heads["pos_transmitter"]:
        circuit_components[0][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in circuit_heads["pos_detector"]:
        circuit_components[2][_receiver_key(layer_idx)].append(head)

    for layer_idx, head in circuit_heads["struct_reader"]:
        circuit_components[-1][_receiver_key(layer_idx)].append(head)

    return circuit_components
