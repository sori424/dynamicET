from __future__ import annotations

from exp2.eval_core import (
    MODEL_NAME_BY_FOLDER,
    STAGE_FILES,
    SimpleTLUtils,
    hook_component_name,
    hook_name,
    import_binding_modules,
    normalize_device,
    resolve_prompt_format,
    set_seed,
    tensor_topk_components,
)

from exp2.circuit_components import (
    append_attention_pattern_context_reader_components,
    circuit_by_position_from_components,
    component_count_by_position,
    compute_context_reader_attention_scores,
    load_circuit_components,
    metric_value,
    random_components_like,
    ranges_overlap,
    slice_dict_tensor_dataset,
    tensor_topk_from_candidates,
    update_output_circuit_fields,
)

from exp2.eval_data import (
    build_box_dataset,
)

from exp2.eval_metrics import (
    compute_mean_activations,
    evaluate_metrics,
    metrics_mean,
    metrics_std,
)

from exp2.completeness import (
    resolve_completeness_figure_path,
    run_completeness_evaluation,
    run_minimality_pruning,
)

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import sys

def resolve_model_results_dir(
    *,
    binding_root: Path,
    results_dir: Optional[Path],
    model_folder: str,
) -> Path:
    candidates: List[Path] = []
    if results_dir is not None:
        for candidate in (results_dir, results_dir / model_folder):
            if any((candidate / filename).is_file() for filename in STAGE_FILES.values()):
                return candidate.resolve()
        # An explicit directory must never silently fall back to another run.
        return results_dir.expanduser().resolve()
    candidates.extend(
        [
            Path.cwd() / "results" / "path_patching" / model_folder,
            binding_root / "results" / "path_patching" / model_folder,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and any((candidate / filename).exists() for filename in STAGE_FILES.values()):
            return candidate.resolve()
    return candidates[-1].resolve()


def model_name_for_folder(model_folder: str, explicit_model_name: Optional[str]) -> str:
    if explicit_model_name:
        return explicit_model_name
    return MODEL_NAME_BY_FOLDER.get(model_folder, model_folder)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=True) + "\n", encoding="utf-8")


def evaluate_one_model(args, binding_root: Path, model_folder: str) -> Dict[str, Any]:
    model_name = model_name_for_folder(model_folder, args.model_name)
    resolved_prompt_format = resolve_prompt_format(model_name, args.prompt_format)
    results_dir = resolve_model_results_dir(
        binding_root=binding_root,
        results_dir=args.results_dir,
        model_folder=model_folder,
    )
    use_attention_context_reader = (
        args.include_context_target_reader_heads
        and args.context_reader_source == "attention-pattern"
    )
    if args.no_model_eval and use_attention_context_reader:
        raise ValueError(
            "--context-reader-source attention-pattern requires loading the model; "
            "remove --no-model-eval or use --context-reader-source path-patching."
        )
    if args.no_model_eval and args.run_completeness:
        raise ValueError(
            "--run-completeness requires loading and evaluating the model; "
            "remove --no-model-eval."
        )
    if args.no_model_eval:
        utils = SimpleTLUtils()
    else:
        modules = import_binding_modules(binding_root)
        torch_module = modules["torch"]
        np_module = modules["np"]
        utils = modules["utils"]
        DataLoader = modules["DataLoader"]
        pp_utils = modules["pp_utils"]
        datascience = modules["datascience"]
        load_vocab = modules["load_vocab"]
        Statement = modules["Statement"]

        selected_device = normalize_device(args.device, torch_module, pp_utils)
        set_seed(args.seed, torch_module=torch_module, np_module=np_module)

    requested_hook_component = args.hook_component
    hook_component = hook_component_name(requested_hook_component)
    heldout_samples = (
        args.num_samples if args.heldout_samples is None else args.heldout_samples
    )
    heldout_prompt_id_start = (
        args.prompt_id_start + args.num_samples
        if args.heldout_prompt_id_start is None
        else args.heldout_prompt_id_start
    )
    if args.run_heldout:
        if heldout_samples <= 0:
            raise ValueError("--heldout-samples must be positive.")
        if ranges_overlap(
            args.prompt_id_start,
            args.num_samples,
            heldout_prompt_id_start,
            heldout_samples,
        ):
            raise ValueError(
                "Held-out prompt range overlaps the selection/eval range: "
                f"A=[{args.prompt_id_start}, {args.prompt_id_start + args.num_samples}), "
                f"B=[{heldout_prompt_id_start}, {heldout_prompt_id_start + heldout_samples})."
            )
    base_output: Dict[str, Any] = {
        "model_folder": model_folder,
        "model_name": model_name,
        "results_dir": str(results_dir),
        "model_only": bool(args.model_only),
        "circuit_source": args.circuit_source,
        "context_reader_source": args.context_reader_source,
        "hook_component": requested_hook_component,
        "resolved_hook_component": hook_component,
        "num_samples": args.num_samples,
        "mean_samples": args.mean_samples,
        "attention_pattern_samples": args.attention_pattern_samples,
        "completeness_samples": args.completeness_samples,
        "batch_size": args.batch_size,
        "prompt_id_start": args.prompt_id_start,
        "prompt_format": resolved_prompt_format,
        "requested_prompt_format": args.prompt_format,
        "system_prompt": args.system_prompt,
        "swap_pair": list(args.swap_pair),
        "counterfactual_mode": args.counterfactual_mode,
        "pos_detector_position_mode": args.pos_detector_position_mode,
        "struct_reader_position_mode": args.struct_reader_position_mode,
        "attention_pattern_position_mode": args.attention_pattern_position_mode,
        "attention_pattern_candidate_source": args.attention_pattern_candidate_source,
        "ablate_non_circuit_positions": args.ablate_non_circuit_positions,
        "preserve_all_heads_at_circuit_positions": args.preserve_all_heads_at_circuit_positions,
        "include_zero_contribution_heads": args.include_zero_contribution_heads,
        "zero_contribution_tolerance": args.zero_contribution_tolerance,
        "include_context_target_reader_heads": args.include_context_target_reader_heads,
        "primary_metric": args.primary_metric,
        "minimality_metric": args.minimality_metric,
        "run_heldout": bool(args.run_heldout),
        "heldout_samples": heldout_samples if args.run_heldout else None,
        "heldout_prompt_id_start": heldout_prompt_id_start if args.run_heldout else None,
    }
    if args.model_only:
        if args.no_model_eval:
            raise ValueError("--model-only requires loading the model.")
        print(f"Loading model {model_name} on {selected_device}...", file=sys.stderr)
        print(f"Using prompt format {resolved_prompt_format}...", file=sys.stderr)
        model, tokenizer = pp_utils.get_model_and_tokenizer(model_name)
        eval_dataset = build_box_dataset(
            torch_module=torch_module,
            pp_utils=pp_utils,
            datascience=datascience,
            load_vocab=load_vocab,
            Statement=Statement,
            tokenizer=tokenizer,
            model_name=model_name,
            vocab_tag=args.vocab_tag,
            vocab_split=args.vocab_split,
            num_entities=args.num_entities,
            num_samples=args.num_samples,
            query_name=args.query_name,
            prompt_id_start=args.prompt_id_start,
            swap_pair=args.swap_pair,
            template_type=args.template_type,
            counterfactual_mode=args.counterfactual_mode,
            prompt_format=resolved_prompt_format,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        print("Evaluating model...", file=sys.stderr)
        output = dict(base_output)
        output["model_metrics"] = evaluate_metrics(
            model=model,
            dataloader=eval_loader,
            device=selected_device,
        )
        if args.run_heldout:
            print("Evaluating held-out model...", file=sys.stderr)
            heldout_dataset = build_box_dataset(
                torch_module=torch_module,
                pp_utils=pp_utils,
                datascience=datascience,
                load_vocab=load_vocab,
                Statement=Statement,
                tokenizer=tokenizer,
                model_name=model_name,
                vocab_tag=args.vocab_tag,
                vocab_split=args.vocab_split,
                num_entities=args.num_entities,
                num_samples=heldout_samples,
                query_name=args.query_name,
                prompt_id_start=heldout_prompt_id_start,
                swap_pair=args.swap_pair,
                template_type=args.template_type,
                counterfactual_mode=args.counterfactual_mode,
                prompt_format=resolved_prompt_format,
            )
            heldout_loader = DataLoader(
                heldout_dataset, batch_size=args.batch_size, shuffle=False
            )
            output["heldout_estimate"] = {
                "prompt_id_start": heldout_prompt_id_start,
                "num_samples": heldout_samples,
                "model_metrics": evaluate_metrics(
                    model=model,
                    dataloader=heldout_loader,
                    device=selected_device,
                ),
            }
        return output

    components, selection_metadata = load_circuit_components(
        results_dir=results_dir,
        utils=utils,
        hook_component=hook_component,
        circuit_source=args.circuit_source,
        largest=args.largest,
        include_zero_contribution_heads=args.include_zero_contribution_heads,
        zero_tolerance=args.zero_contribution_tolerance,
        n_value_fetcher=args.n_value_fetcher,
        n_pos_trans=args.n_pos_trans,
        n_pos_detect=args.n_pos_detect,
        n_struct_read=args.n_struct_read,
        n_context_reader=args.n_context_reader,
        pos_detector_position_mode=args.pos_detector_position_mode,
        struct_reader_position_mode=args.struct_reader_position_mode,
        include_context_target_reader_heads=(
            args.include_context_target_reader_heads
            and args.context_reader_source == "path-patching"
        ),
    )
    circuit_by_position = circuit_by_position_from_components(components)
    hook_names = sorted(
        {
            hook_name(utils, hook_component, layer)
            for layer in range(
                selection_metadata["score_tensors"][
                    next(iter(selection_metadata["score_tensors"]))
                ]["shape"][0]
            )
        }
    )

    output: Dict[str, Any] = {
        **base_output,
        "component_count": len(components),
        "component_count_by_position": component_count_by_position(components),
        "components": [asdict(component) for component in components],
        "circuit_by_position": circuit_by_position,
        **selection_metadata,
    }

    if args.no_model_eval:
        return output

    print(f"Loading model {model_name} on {selected_device}...", file=sys.stderr)
    print(f"Using prompt format {resolved_prompt_format}...", file=sys.stderr)
    model, tokenizer = pp_utils.get_model_and_tokenizer(model_name)
    if hook_component == "result":
        if hasattr(model, "set_use_attn_result"):
            model.set_use_attn_result(True)
        else:
            model.cfg.use_attn_result = True

    eval_dataset = build_box_dataset(
        torch_module=torch_module,
        pp_utils=pp_utils,
        datascience=datascience,
        load_vocab=load_vocab,
        Statement=Statement,
        tokenizer=tokenizer,
        model_name=model_name,
        vocab_tag=args.vocab_tag,
        vocab_split=args.vocab_split,
        num_entities=args.num_entities,
        num_samples=args.num_samples,
        query_name=args.query_name,
        prompt_id_start=args.prompt_id_start,
        swap_pair=args.swap_pair,
        template_type=args.template_type,
        counterfactual_mode=args.counterfactual_mode,
        prompt_format=resolved_prompt_format,
    )
    mean_dataset = build_box_dataset(
        torch_module=torch_module,
        pp_utils=pp_utils,
        datascience=datascience,
        load_vocab=load_vocab,
        Statement=Statement,
        tokenizer=tokenizer,
        model_name=model_name,
        vocab_tag=args.vocab_tag,
        vocab_split=args.vocab_split,
        num_entities=args.num_entities,
        num_samples=args.mean_samples,
        query_name=args.query_name,
        prompt_id_start=args.prompt_id_start,
        swap_pair=args.swap_pair,
        template_type=args.template_type,
        counterfactual_mode=args.counterfactual_mode,
        prompt_format=resolved_prompt_format,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    mean_loader = DataLoader(mean_dataset, batch_size=args.batch_size, shuffle=False)

    if use_attention_context_reader:
        attention_samples = min(args.attention_pattern_samples, len(eval_dataset))
        attention_dataset = slice_dict_tensor_dataset(eval_dataset, attention_samples)
        attention_loader = DataLoader(
            attention_dataset, batch_size=args.batch_size, shuffle=False
        )
        print(
            "Selecting context-reader heads from attention patterns...",
            file=sys.stderr,
        )
        attention_scores = compute_context_reader_attention_scores(
            model=model,
            dataloader=attention_loader,
            device=selected_device,
            utils=utils,
            torch_module=torch_module,
        )
        if args.attention_pattern_candidate_source == "struct-reader-swap-target":
            candidate_heads = selection_metadata["selected_heads"].get(
                "struct_reader_swap_target", []
            )
            context_heads, context_scores = tensor_topk_from_candidates(
                attention_scores,
                candidate_heads,
                args.n_context_reader,
                largest=True,
            )
        else:
            context_heads, context_scores = tensor_topk_components(
                attention_scores,
                args.n_context_reader,
                largest=True,
            )
        if not args.include_zero_contribution_heads:
            filtered = [
                (head, score)
                for head, score in zip(context_heads, context_scores)
                if score > args.zero_contribution_tolerance
            ]
            context_heads = [head for head, _ in filtered]
            context_scores = [score for _, score in filtered]
        components = append_attention_pattern_context_reader_components(
            components=components,
            selected_heads=context_heads,
            selected_scores=context_scores,
            position_mode=args.attention_pattern_position_mode,
            hook_component=hook_component,
            utils=utils,
        )
        circuit_by_position = circuit_by_position_from_components(components)
        selection_metadata["selected_heads"][
            "struct_reader_context_target_box_to_target_box"
        ] = context_heads
        selection_metadata["selected_scores"][
            "struct_reader_context_target_box_to_target_box"
        ] = context_scores
        selection_metadata["attention_pattern_context_reader"] = {
            "samples": attention_samples,
            "query_position_field": "base_swap_target_box_positions",
            "key_position_field": "base_context_target_box_positions",
            "component_position_mode": args.attention_pattern_position_mode,
            "candidate_source": args.attention_pattern_candidate_source,
            "score_shape": list(attention_scores.shape),
            "score_min": float(attention_scores.min().item()),
            "score_max": float(attention_scores.max().item()),
            "selected_heads": context_heads,
            "selected_scores": context_scores,
        }
        output["selected_heads"] = selection_metadata["selected_heads"]
        output["selected_scores"] = selection_metadata["selected_scores"]
        output["attention_pattern_context_reader"] = selection_metadata[
            "attention_pattern_context_reader"
        ]
        update_output_circuit_fields(output, components)

    print("Computing mean activations...", file=sys.stderr)
    mean_activations = compute_mean_activations(
        model,
        mean_loader,
        hook_names,
        selected_device,
    )

    print("Evaluating model...", file=sys.stderr)
    model_metrics = evaluate_metrics(
        model=model,
        dataloader=eval_loader,
        device=selected_device,
    )
    print("Evaluating circuit...", file=sys.stderr)
    circuit_metrics = evaluate_metrics(
        model=model,
        dataloader=eval_loader,
        device=selected_device,
        hook_names=hook_names,
        circuit_by_position=circuit_by_position,
        mean_activations=mean_activations,
        ablate_non_circuit_positions=args.ablate_non_circuit_positions,
        preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
    )
    output["model_metrics"] = model_metrics
    output["circuit_metrics"] = circuit_metrics

    final_components = components
    if args.run_minimality:
        minimality_samples = min(args.minimality_samples, len(eval_dataset))
        minimality_dataset = slice_dict_tensor_dataset(eval_dataset, minimality_samples)
        minimality_loader = DataLoader(
            minimality_dataset, batch_size=args.batch_size, shuffle=False
        )
        print("Running minimality pruning...", file=sys.stderr)
        final_components, minimality_result = run_minimality_pruning(
            model=model,
            dataloader=minimality_loader,
            device=selected_device,
            components=components,
            mean_activations=mean_activations,
            hook_names=hook_names,
            primary_metric=args.minimality_metric,
            minimality_threshold=args.minimality_threshold,
            minimality_criterion=args.minimality_criterion,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
        )
        output["minimality_pruning"] = minimality_result
        output["minimal_circuit_by_position"] = circuit_by_position_from_components(
            final_components
        )
        print("Evaluating minimal circuit on full eval set...", file=sys.stderr)
        output["minimal_circuit_metrics"] = evaluate_metrics(
            model=model,
            dataloader=eval_loader,
            device=selected_device,
            hook_names=hook_names,
            circuit_by_position=circuit_by_position_from_components(final_components),
            mean_activations=mean_activations,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
        )

    if args.run_heldout:
        print("Building held-out eval set...", file=sys.stderr)
        heldout_dataset = build_box_dataset(
            torch_module=torch_module,
            pp_utils=pp_utils,
            datascience=datascience,
            load_vocab=load_vocab,
            Statement=Statement,
            tokenizer=tokenizer,
            model_name=model_name,
            vocab_tag=args.vocab_tag,
            vocab_split=args.vocab_split,
            num_entities=args.num_entities,
            num_samples=heldout_samples,
            query_name=args.query_name,
            prompt_id_start=heldout_prompt_id_start,
            swap_pair=args.swap_pair,
            template_type=args.template_type,
            counterfactual_mode=args.counterfactual_mode,
            prompt_format=resolved_prompt_format,
        )
        heldout_loader = DataLoader(
            heldout_dataset, batch_size=args.batch_size, shuffle=False
        )
        print("Evaluating held-out model...", file=sys.stderr)
        heldout_model_metrics = evaluate_metrics(
            model=model,
            dataloader=heldout_loader,
            device=selected_device,
        )
        print("Evaluating held-out circuit...", file=sys.stderr)
        heldout_circuit_metrics = evaluate_metrics(
            model=model,
            dataloader=heldout_loader,
            device=selected_device,
            hook_names=hook_names,
            circuit_by_position=circuit_by_position,
            mean_activations=mean_activations,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
        )
        heldout_result: Dict[str, Any] = {
            "prompt_id_start": heldout_prompt_id_start,
            "num_samples": heldout_samples,
            "selection_prompt_id_start": args.prompt_id_start,
            "selection_num_samples": args.num_samples,
            "mean_prompt_id_start": args.prompt_id_start,
            "mean_samples": args.mean_samples,
            "component_set": (
                "minimality_pruning.kept_components"
                if args.run_minimality
                else "components"
            ),
            "component_count": len(final_components),
            "metric_name": args.primary_metric,
            "model_metrics": heldout_model_metrics,
            "circuit_metrics": heldout_circuit_metrics,
        }
        heldout_final_metrics = heldout_circuit_metrics
        if args.run_minimality:
            print("Evaluating held-out minimal circuit...", file=sys.stderr)
            heldout_final_metrics = evaluate_metrics(
                model=model,
                dataloader=heldout_loader,
                device=selected_device,
                hook_names=hook_names,
                circuit_by_position=circuit_by_position_from_components(final_components),
                mean_activations=mean_activations,
                ablate_non_circuit_positions=args.ablate_non_circuit_positions,
                preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
            )
            heldout_result["minimal_circuit_metrics"] = heldout_final_metrics
        heldout_result["final_circuit_metrics"] = heldout_final_metrics
        heldout_model_primary = metric_value(heldout_model_metrics, args.primary_metric)
        heldout_circuit_primary = metric_value(heldout_final_metrics, args.primary_metric)
        heldout_result["faithfulness_ratio_circuit_over_model"] = (
            heldout_circuit_primary / heldout_model_primary
            if heldout_model_primary
            else None
        )
        output["heldout_estimate"] = heldout_result

    if args.run_completeness:
        completeness_samples = min(args.completeness_samples, len(eval_dataset))
        completeness_dataset = slice_dict_tensor_dataset(
            eval_dataset,
            completeness_samples,
        )
        completeness_loader = DataLoader(
            completeness_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        completeness_figure_path = None
        if args.completeness_plot:
            completeness_figure_path = resolve_completeness_figure_path(
                requested_path=args.completeness_figure,
                results_dir=results_dir,
                model_folder=model_folder,
                metric_name=args.primary_metric,
            )
        print("Running completeness evaluation...", file=sys.stderr)
        output["completeness"] = run_completeness_evaluation(
            model=model,
            dataloader=completeness_loader,
            device=selected_device,
            components=final_components,
            mean_activations=mean_activations,
            hook_names=hook_names,
            primary_metric=args.primary_metric,
            random_iters=args.completeness_random_iters,
            random_removal_probability=args.completeness_random_removal_probability,
            seed=args.seed + 2,
            ablate_non_circuit_positions=args.ablate_non_circuit_positions,
            utils=utils,
            hook_component=hook_component,
            figure_path=completeness_figure_path,
            plot_title=(
                f"{model_folder} circuit completeness "
                f"({args.primary_metric})"
            ),
        )

    if args.random_circuit_iters > 0:
        rng = random.Random(args.seed + 1)
        random_metrics = []
        for idx in range(args.random_circuit_iters):
            print(f"Evaluating random circuit {idx + 1}/{args.random_circuit_iters}...", file=sys.stderr)
            random_components = random_components_like(
                components=final_components,
                model=model,
                utils=utils,
                hook_component=hook_component,
                rng=rng,
            )
            random_metrics.append(
                evaluate_metrics(
                    model=model,
                    dataloader=eval_loader,
                    device=selected_device,
                    hook_names=hook_names,
                    circuit_by_position=circuit_by_position_from_components(
                        random_components
                    ),
                    mean_activations=mean_activations,
                    ablate_non_circuit_positions=args.ablate_non_circuit_positions,
                    preserve_all_heads_at_circuit_positions=args.preserve_all_heads_at_circuit_positions,
                )
            )
        output["random_circuit_metrics"] = random_metrics
        output["random_circuit_metric_mean"] = metrics_mean(
            random_metrics, args.primary_metric
        )
        output["random_circuit_metric_std"] = metrics_std(
            random_metrics, args.primary_metric
        )

    model_primary = metric_value(model_metrics, args.primary_metric)
    circuit_primary = metric_value(
        output.get("minimal_circuit_metrics", circuit_metrics), args.primary_metric
    )
    output["faithfulness_ratio_circuit_over_model"] = (
        circuit_primary / model_primary if model_primary else None
    )
    random_mean = output.get("random_circuit_metric_mean")
    if random_mean is not None:
        denom = model_primary - random_mean
        output["faithfulness_over_random"] = (
            (circuit_primary - random_mean) / denom if denom else None
        )

    return output
