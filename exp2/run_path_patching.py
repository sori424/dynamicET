#!/usr/bin/env python
"""Recover the five-stage dynamic-entity-tracking circuit with path patching.

The implementation intentionally keeps imports lightweight until after CLI
parsing so ``--help`` works before the research dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGES = (
    {
        "name": "value_fetcher",
        "filename": "value_fetcher.pt",
        "sender_position": "answer_colon",
        "receiver_stage": None,
        "receiver_composition": None,
        "receiver_position": None,
        "k_argument": "n_value_fetcher",
        "paper_role": "A: answer retriever",
    },
    {
        "name": "pos_transmitter",
        "filename": "pos_transmitter.pt",
        "sender_position": "answer_colon",
        "receiver_stage": "value_fetcher",
        "receiver_composition": "q",
        "receiver_position": "answer_colon",
        "k_argument": "n_pos_trans",
        "paper_role": "B: dereferencer",
    },
    {
        "name": "pos_detector",
        "filename": "pos_detector.pt",
        "sender_position": "question_query_box",
        "receiver_stage": "pos_transmitter",
        "receiver_composition": "v",
        "receiver_position": "question_query_box",
        "k_argument": "n_pos_detect",
        "paper_role": "C: position updater",
    },
    {
        "name": "struct_reader_swap_target",
        "filename": "struct_reader_swap_target.pt",
        "sender_position": "swap_target_box",
        "receiver_stage": "pos_detector",
        "receiver_composition": "q",
        "receiver_position": "question_query_box",
        "k_argument": "n_struct_read",
        "paper_role": "D: swap-position transmitter",
    },
    {
        "name": "struct_reader_context_target_box_to_target_box",
        "filename": "struct_reader_context_target_box_to_target_box.pt",
        "sender_position": "context_target_box",
        "receiver_stage": "struct_reader_swap_target",
        "receiver_composition": "v",
        "receiver_position": "context_target_box",
        "k_argument": "n_context_reader",
        "paper_role": "E: binding anchor",
    },
)


def parse_swap_pair(value: str) -> Tuple[int, int]:
    pieces = value.split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected a swap pair like 1,0.")
    try:
        first, second = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Swap pair entries must be integers.") from exc
    if first == second:
        raise argparse.ArgumentTypeError("Swap pair entries must differ.")
    return first, second


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the five greedy Q/V path-patching stages and save one "
            "[layer, head] score tensor per circuit role."
        )
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        default="llama-8b-hf",
        help="TransformerLens/Hugging Face model name or a repository alias.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        type=Path,
        default=None,
        help=(
            "Result directory. Defaults to "
            "results/path_patching/<sanitized-model-name>."
        ),
    )
    parser.add_argument(
        "--datafile",
        type=Path,
        default=None,
        help=(
            "Optional JSONL clean/corrupt prompt pairs. Without this option, "
            "deterministic BOXES prompt pairs are generated locally."
        ),
    )
    parser.add_argument("--num-samples", "--num_samples", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1)
    parser.add_argument("--num-boxes", "--num_boxes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (for example cuda, cuda:1, 1, or cpu).",
    )
    parser.add_argument(
        "--prompt-format",
        "--prompt_format",
        default="auto",
        choices=(
            "raw",
            "auto",
            "chat-template",
            "chat-prefill",
            "gemma-it",
            "gemma-it-prefill",
            "llama-instruct",
            "llama-instruct-prefill",
            "qwen-instruct",
            "qwen-instruct-prefill",
        ),
        help="Prompt wrapper for locally generated data.",
    )
    parser.add_argument("--vocab-tag", "--vocab_tag", default="BOXES")
    parser.add_argument("--vocab-split", "--vocab_split", default="train")
    parser.add_argument("--query-name", "--query_name", type=int, default=0)
    parser.add_argument(
        "--prompt-id-start",
        "--prompt_id_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--swap-pair",
        "--swap_pair",
        type=parse_swap_pair,
        default=(1, 0),
    )
    parser.add_argument("--template-type", "--template_type", default="normal")
    parser.add_argument(
        "--counterfactual-mode",
        "--counterfactual_mode",
        choices=("path_patching_shifted", "same_objects", "new_swap_objects"),
        default="path_patching_shifted",
    )
    parser.add_argument(
        "--n-value-fetcher",
        "--n_value_fetcher",
        type=int,
        default=50,
    )
    parser.add_argument("--n-pos-trans", "--n_pos_trans", type=int, default=50)
    parser.add_argument("--n-pos-detect", "--n_pos_detect", type=int, default=25)
    parser.add_argument("--n-struct-read", "--n_struct_read", type=int, default=10)
    parser.add_argument(
        "--n-context-reader",
        "--n_context_reader",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse valid stage tensors already present in the output directory.",
    )
    parser.add_argument(
        "--stop-after",
        choices=tuple(stage["name"] for stage in STAGES),
        default=None,
        help="Stop after this stage; useful for scheduled or incremental runs.",
    )
    parser.add_argument("--select-circuit", action="store_true",
                        help="Select circuit heads after completing all five stages.")
    parser.add_argument("--circuit-json", type=Path, default=None,
                        help="Selected circuit JSON (default: <output-path>/circuit.json).")
    return parser


def _safe_model_folder(model_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", model_name).strip("._-")
    return value or "model"


def _selected_stages(stop_after: Optional[str]) -> Sequence[Dict[str, Any]]:
    if stop_after is None:
        return STAGES
    stop_index = next(
        index for index, stage in enumerate(STAGES) if stage["name"] == stop_after
    )
    return STAGES[: stop_index + 1]


def _existing_stage_path(output_path: Path, stage: Dict[str, Any]) -> Optional[Path]:
    canonical = output_path / stage["filename"]
    if canonical.exists():
        return canonical
    if stage["name"] == "struct_reader_swap_target":
        legacy = output_path / "struct_reader.pt"
        if legacy.exists():
            return legacy
    return None


def _torch_load_cpu(torch_module, path: Path):
    try:
        tensor = torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch_module.load(path, map_location="cpu")
    if not isinstance(tensor, torch_module.Tensor):
        raise TypeError(f"Expected a tensor in {path}, found {type(tensor).__name__}.")
    if tensor.ndim != 2 or tensor.numel() == 0 or not torch_module.isfinite(tensor).all():
        raise ValueError(f"Expected finite, nonempty [layer, head] scores in {path}.")
    return tensor.detach().cpu()


def _file_digest(path: Path) -> str:
    from exp2.artifacts import file_digest

    return file_digest(path)


def _run_config(args) -> Dict[str, Any]:
    # Version the data/intervention semantics so older artifacts cannot silently
    # become results for a different experiment. Top-k affects downstream stages.
    excluded = {"output_path", "resume", "stop_after", "device", "datafile",
                "select_circuit", "circuit_json"}
    config = {key: value for key, value in vars(args).items() if key not in excluded}
    config["swap_pair"] = list(args.swap_pair)
    config["implementation_version"] = 2
    config["data_sha256"] = _file_digest(args.datafile) if args.datafile else None
    return config


def _resume_paths(args, output_path, stages, config):
    paths = [None] * len(stages)
    if not args.resume:
        return paths, None
    metadata_path = output_path / "path_patching_metadata.json"
    if not metadata_path.exists():
        if any(_existing_stage_path(output_path, stage) for stage in STAGES):
            raise ValueError("Cannot resume tensors without run metadata. Use a new "
                             "output directory or --no-resume to recompute.")
        return paths, None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_config") != config:
        raise ValueError("Saved path-patching configuration differs from this run. "
                         "Use a new output directory or --no-resume to recompute.")
    for index, stage in enumerate(stages):
        saved = metadata.get("stages", {}).get(stage["name"])
        path = output_path / stage["filename"]
        # Once a stage is missing, all dependent stages must be recomputed.
        if saved is None or not path.is_file():
            break
        if saved.get("sha256") != _file_digest(path):
            raise ValueError(f"Saved score tensor changed: {path}. Use --no-resume.")
        paths[index] = path
    return paths, metadata


def _select_circuit(args, output_path):
    if not args.select_circuit or len(_selected_stages(args.stop_after)) != len(STAGES):
        return
    output_json = (args.circuit_json or output_path / "circuit.json").resolve()
    command = [sys.executable, str(PROJECT_ROOT / "exp2" / "circuit_eval.py"),
               "--models", _safe_model_folder(args.model_name),
               "--model-name", args.model_name, "--results-dir", str(output_path),
               "--output-json", str(output_json), "--prompt-format", args.prompt_format,
               "--num-entities", str(args.num_boxes), "--no-model-eval"]
    for stage in STAGES:
        key = stage["k_argument"]
        command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _atomic_torch_save(torch_module, tensor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch_module.save(tensor.detach().cpu(), temporary)
    os.replace(temporary, path)


def _atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_boxes < 2:
        raise ValueError("--num-boxes must be at least 2.")
    if args.query_name < 0 or args.query_name >= args.num_boxes:
        raise ValueError("--query-name must index one of the configured boxes.")
    if args.query_name not in args.swap_pair:
        raise ValueError("--query-name must be one of the --swap-pair entries.")
    if any(index < 0 or index >= args.num_boxes for index in args.swap_pair):
        raise ValueError("--swap-pair entries must index configured boxes.")
    for stage in STAGES:
        if getattr(args, stage["k_argument"]) <= 0:
            raise ValueError(f"--{stage['k_argument'].replace('_', '-')} must be positive.")
    if args.datafile is not None and not args.datafile.is_file():
        raise FileNotFoundError(f"Data file not found: {args.datafile}")


def _set_seed(seed: int, torch_module, np_module) -> None:
    random.seed(seed)
    np_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _normalize_device(value: Optional[str], torch_module):
    if value is None:
        device = torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )
    elif str(value).isdigit():
        device = torch_module.device(f"cuda:{value}")
    else:
        device = torch_module.device(value)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    if device.type == "cuda" and device.index is not None:
        torch_module.cuda.set_device(device)
    return device


def _build_dataloader(args, *, model, tokenizer, torch_module, DataLoader, pp_utils):
    if args.datafile is not None:
        return pp_utils.load_dataloader(
            model=model,
            tokenizer=tokenizer,
            datafile=str(args.datafile),
            num_samples=args.num_samples,
            num_boxes=args.num_boxes,
            batch_size=args.batch_size,
        ), args.prompt_format

    from dynamicet_data import datascience
    from dynamicet_data.api import load_vocab
    from dynamicet_data.box import Statement
    from exp2.eval_core import resolve_prompt_format
    from exp2.eval_data import build_box_dataset

    resolved_prompt_format = resolve_prompt_format(args.model_name, args.prompt_format)
    dataset = build_box_dataset(
        torch_module=torch_module,
        pp_utils=pp_utils,
        datascience=datascience,
        load_vocab=load_vocab,
        Statement=Statement,
        tokenizer=tokenizer,
        model_name=args.model_name,
        vocab_tag=args.vocab_tag,
        vocab_split=args.vocab_split,
        num_entities=args.num_boxes,
        num_samples=args.num_samples,
        query_name=args.query_name,
        prompt_id_start=args.prompt_id_start,
        swap_pair=args.swap_pair,
        template_type=args.template_type,
        counterfactual_mode=args.counterfactual_mode,
        prompt_format=resolved_prompt_format,
    )
    return (
        DataLoader(dataset, batch_size=args.batch_size, shuffle=False),
        resolved_prompt_format,
    )


def _stage_metadata(stage, tensor, selected_heads, selected_scores, *, reused, seconds):
    return {
        "paper_role": stage["paper_role"],
        "file": stage["filename"],
        "shape": list(tensor.shape),
        "sender_position": stage["sender_position"],
        "receiver_stage": stage["receiver_stage"],
        "receiver_composition": stage["receiver_composition"],
        "receiver_position": stage["receiver_position"],
        "score_min": float(tensor.min().item()),
        "score_max": float(tensor.max().item()),
        "score_mean": float(tensor.float().mean().item()),
        "nonzero": int((tensor != 0).sum().item()),
        "selected_heads": selected_heads,
        "selected_scores": selected_scores,
        "reused": bool(reused),
        "duration_seconds": round(float(seconds), 3),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    _validate_args(args)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from transformers.utils import import_utils
    except ImportError:
        pass
    else:
        import_utils._torchvision_available = False
        import_utils._torchvision_version = "N/A"

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from exp2 import pp_utils

    stages = _selected_stages(args.stop_after)
    output_path = (
        args.output_path
        if args.output_path is not None
        else PROJECT_ROOT
        / "results"
        / "path_patching"
        / _safe_model_folder(args.model_name)
    ).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path / "path_patching_metadata.json"
    config = _run_config(args)
    existing_paths, previous_metadata = _resume_paths(args, output_path, stages, config)

    base_metadata: Dict[str, Any] = {
        "run_config": config,
        "model_name": args.model_name,
        "output_path": str(output_path),
        "datafile": str(args.datafile.resolve()) if args.datafile else None,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_boxes": args.num_boxes,
        "seed": args.seed,
        "requested_device": args.device,
        "requested_prompt_format": args.prompt_format,
        "query_name": args.query_name,
        "prompt_id_start": args.prompt_id_start,
        "swap_pair": list(args.swap_pair),
        "counterfactual_mode": args.counterfactual_mode,
        "resume": args.resume,
        "stop_after": args.stop_after,
        "score_definition": "mean((patched_probability-clean_probability)/clean_probability)",
        "selection": "most-negative top-k per stage",
        "stages": {},
    }

    # A completed resumable run should not reload a multi-billion-parameter model.
    if args.resume and all(path is not None for path in existing_paths):
        shapes = set()
        for stage, path in zip(stages, existing_paths):
            tensor = _torch_load_cpu(torch, path)
            if tensor.ndim != 2:
                raise ValueError(
                    f"Expected [layer, head] scores in {path}, got {tuple(tensor.shape)}."
                )
            canonical_path = output_path / stage["filename"]
            if path != canonical_path:
                _atomic_torch_save(torch, tensor, canonical_path)
            shapes.add(tuple(tensor.shape))
            k = int(getattr(args, stage["k_argument"]))
            if k > int(tensor.numel()):
                raise ValueError(
                    f"Requested top-k={k} for {stage['name']}, but its score "
                    f"tensor contains only {tensor.numel()} heads."
                )
            heads, values = pp_utils.compute_topk_components(
                tensor,
                k=k,
                largest=False,
                return_values=True,
            )
            base_metadata["stages"][stage["name"]] = _stage_metadata(
                stage,
                tensor,
                heads,
                values,
                reused=True,
                seconds=0.0,
            )
        if len(shapes) != 1:
            raise ValueError(f"Resumed stage tensors have inconsistent shapes: {shapes}")
        # Preserve the original run's provenance and any later completed stages.
        print(f"Reused {len(stages)} path-patching stages in {output_path}")
        _select_circuit(args, output_path)
        return

    device = _normalize_device(args.device, torch)
    pp_utils.set_device(device)
    _set_seed(args.seed, torch, np)

    print(f"Loading model {args.model_name} on {device}...", file=sys.stderr)
    model, tokenizer = pp_utils.get_model_and_tokenizer(args.model_name)
    model.eval()
    model_component_count = int(model.cfg.n_layers) * int(model.cfg.n_heads)
    for stage in stages:
        if getattr(args, stage["k_argument"]) > model_component_count:
            raise ValueError(f"Requested top-k for {stage['name']} exceeds the model's "
                             f"{model_component_count} attention heads.")
    dataloader, resolved_prompt_format = _build_dataloader(
        args,
        model=model,
        tokenizer=tokenizer,
        torch_module=torch,
        DataLoader=DataLoader,
        pp_utils=pp_utils,
    )
    if len(dataloader.dataset) == 0:
        raise ValueError("Path-patching dataset is empty.")
    base_metadata["device"] = str(device)
    base_metadata["resolved_prompt_format"] = resolved_prompt_format
    base_metadata["dataset_size"] = len(dataloader.dataset)
    # Invalidate old downstream files before starting a fresh/partial run.
    if previous_metadata:
        for stage, path in zip(stages, existing_paths):
            if path is not None:
                base_metadata["stages"][stage["name"]] = previous_metadata["stages"][stage["name"]]
    _atomic_json_dump(base_metadata, metadata_path)

    print("Caching clean and corrupt activations...", file=sys.stderr)
    (
        clean_cache,
        corrupt_cache,
        clean_logit_outputs,
        corrupt_logit_outputs,
        _,
    ) = pp_utils.get_caches(model, dataloader)
    clean_probs = [
        probability
        for batch in clean_logit_outputs.values()
        for probability in batch.values()
    ]
    corrupt_probs = [
        probability
        for batch in corrupt_logit_outputs.values()
        for probability in batch.values()
    ]
    base_metadata["clean_target_probability_mean"] = float(np.mean(clean_probs))
    base_metadata["corrupt_target_probability_mean"] = float(np.mean(corrupt_probs))

    selected_by_stage: Dict[str, Sequence[Sequence[int]]] = {}
    for stage, existing_path in zip(stages, existing_paths):
        started = time.monotonic()
        reused = bool(args.resume and existing_path is not None)
        if reused:
            print(f"Reusing {existing_path.name}...", file=sys.stderr)
            tensor = _torch_load_cpu(torch, existing_path)
            canonical_path = output_path / stage["filename"]
            if existing_path != canonical_path:
                _atomic_torch_save(torch, tensor, canonical_path)
        else:
            receiver_stage = stage["receiver_stage"]
            receiver_heads = None
            if receiver_stage is not None:
                if receiver_stage not in selected_by_stage:
                    raise RuntimeError(
                        f"Stage {stage['name']} requires unfinished stage {receiver_stage}."
                    )
                receiver_heads = [
                    (int(layer), int(head))
                    for layer, head in selected_by_stage[receiver_stage]
                ]
            print(
                f"Scoring {stage['paper_role']} ({stage['name']})...",
                file=sys.stderr,
            )
            tensor = pp_utils.score_path_patching_heads(
                model=model,
                dataloader=dataloader,
                clean_cache=clean_cache,
                corrupt_cache=corrupt_cache,
                clean_logit_outputs=clean_logit_outputs,
                sender_position_mode=stage["sender_position"],
                receiver_heads=receiver_heads,
                receiver_composition=stage["receiver_composition"],
                receiver_position_mode=stage["receiver_position"],
                description=stage["name"],
            )
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected [layer, head] scores for {stage['name']}, "
                f"got {tuple(tensor.shape)}."
            )
        expected_shape = (int(model.cfg.n_layers), int(model.cfg.n_heads))
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Score shape for {stage['name']} is {tuple(tensor.shape)}, "
                f"but model shape is {expected_shape}. Use --no-resume to recompute."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite scores for {stage['name']}; stage was not saved.")
        requested_k = int(getattr(args, stage["k_argument"]))
        if requested_k > model_component_count:
            raise ValueError(
                f"Requested top-k={requested_k} for {stage['name']}, but the model "
                f"has only {model_component_count} attention heads."
            )
        heads, values = pp_utils.compute_topk_components(
            tensor,
            k=requested_k,
            largest=False,
            return_values=True,
        )
        selected_by_stage[stage["name"]] = heads
        _atomic_torch_save(torch, tensor, output_path / stage["filename"])
        if stage["name"] == "struct_reader_swap_target":
            _atomic_torch_save(torch, tensor, output_path / "struct_reader.pt")
        base_metadata["stages"][stage["name"]] = _stage_metadata(
            stage,
            tensor,
            heads,
            values,
            reused=reused,
            seconds=time.monotonic() - started,
        )
        base_metadata["stages"][stage["name"]]["sha256"] = _file_digest(
            output_path / stage["filename"]
        )
        if stage["name"] == "struct_reader_swap_target":
            base_metadata["stages"][stage["name"]]["legacy_sha256"] = _file_digest(
                output_path / "struct_reader.pt"
            )
        _atomic_json_dump(base_metadata, metadata_path)

    print(f"Path-patching tensors written to {output_path}")
    print(f"Run metadata written to {metadata_path}")
    _select_circuit(args, output_path)


if __name__ == "__main__":
    main()
