# Retrieval-conditioned mechanism of entity rebinding in large language models

https://arxiv.org/pdf/2606.08644

This repository contains the code used for the experiments in the paper *A retrieval conditioned rebinding circuit for dynamic entity tracking in
large language models*. We study how instruction-tuned Gemma and Llama models implement rebinding during dynamic state tracking.

We conduct our experiments in four steps:

- **Exp 1.** Interchange intervention in the residual stream at target tokens (Section 3)
- **Exp 2.** Path patching for circuit analysis (Section 4.1)
- **Exp 3.** Interchange intervention in attention patterns (Section 4.2)
- **Exp 4.** Binding ID intervention (Section 5)

### Experiment 1: residual-stream tracing

- `tracing.py`: command-line entry point and experiment orchestration
- `prompt_formatting.py`: model aliases, chat templates, tokenization, and answer-token formatting
- `trace_contexts.py`: trace-mode validation and clean/source context construction
- `trace_data.py`: tracing datasets, token selectors, and correct-sample filtering
- `trace_runner.py`: activation patching, metric aggregation, and result serialization
- `tracing_utils.py`: compatibility facade for existing imports

### Experiment 2: circuit discovery and evaluation

- `circuit_eval.py`: command-line entry point
- `eval_core.py`: shared data types, model/runtime setup, and prompt utilities
- `eval_data.py`: paired-context and BOXES dataset construction
- `eval_metrics.py`: circuit evaluation and metric aggregation
- `circuit_components.py`: component loading, selection, and random-circuit construction
- `completeness.py`: completeness and minimality analyses
- `eval_runner.py`: per-model evaluation workflow and output writing
- `eval_cli.py`: command-line arguments and top-level dispatch
- `pp_prompts.py`, `pp_corruptions.py`, `pp_models.py`, `pp_patching.py`: path-patching data, model, and intervention utilities
- `pp_utils.py`: compatibility facade for existing imports

### Experiment 3: attention-head roles

- `head_role.py`: command-line entry point
- `head_config.py`: constants, component selection, and model/runtime configuration
- `head_data.py`: prompt formatting and paired-dataset construction
- `head_metrics.py`: token- and condition-level metrics
- `head_interventions.py`: attention-pattern and head-output interventions
- `role_scores.py`: role-specific scoring and dataset previews
- `head_cli.py`: command-line arguments
- `head_runner.py`: end-to-end experiment orchestration

### Experiment 4: binding-ID intervention

- `bid_intervention.py`: command-line entry point
- `bid_data.py`: condition parsing, dataset construction, and Group-B head selection
- `bid_directions.py`: binding-direction estimation and Q/K intervention hooks
- `bid_attention.py`: attention, carrier, and logit statistics
- `bid_evaluation.py`: condition evaluation and intervention specifications
- `bid_cli.py`: serialization and command-line arguments
- `bid_runner.py`: end-to-end experiment orchestration
