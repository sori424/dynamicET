# Retrieval-conditioned mechanism of entity rebinding in large language models

https://arxiv.org/pdf/2606.08644

This repository contains the code used for the experiments in the paper *A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models* EMNLP 2026 Main. We study how instruction-tuned Gemma and Llama models implement rebinding during dynamic state tracking.

We conduct our experiments in four steps:

- **Exp 1.** Interchange intervention in the residual stream at target tokens (Section 3)
- **Exp 2.** Path patching for circuit analysis (Section 4.1)
- **Exp 3.** Interchange intervention in attention patterns (Section 4.2)
- **Exp 4.** Binding ID intervention (Section 5)

## Running path patching

Path patching produces the attention-head score tensors consumed by the later
experiments. The wrapper generates BOXES prompt pairs by default, runs the five
greedy Q/V-patching stages, and selects the resulting circuit:

```bash
bash exp2/run_path_patching.sh
```

Set `MODEL_NAME`, `MODEL_FOLDER`, `RESULTS_DIR`, or the other variables at the
top of the wrapper to change a run. Completed stage tensors are reused by
default; pass `--no-resume` to recompute them. The Python entry point also
supports `--stop-after STAGE` for long scheduled runs:

```bash
python exp2/run_path_patching.py --help
```
