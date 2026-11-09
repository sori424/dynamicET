# Retrieval-conditioned mechanism of entity rebinding in large language models

https://arxiv.org/pdf/2606.08644

This repository contains the code used for the experiments in the paper *A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models* EMNLP 2026 Main. We study how instruction-tuned Gemma and Llama models implement rebinding during dynamic state tracking.

We conduct our experiments in four steps:

- **Exp 1.** Interchange intervention in the residual stream at target tokens (Section 3)
- **Exp 2.** Path patching for circuit analysis (Section 4.1)
- **Exp 3.** Interchange intervention in attention patterns (Section 4.2)
- **Exp 4.** Binding ID intervention (Section 5)

## Installation

Use Python 3.11 in a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The core versions are PyTorch 2.7.1, TransformerLens 2.16.1, and Transformers
4.52.4. Install the appropriate PyTorch wheel for your GPU platform first if
needed. PyArrow is pinned to remain compatible with NumPy 1.26. Local BOXES
helpers live in `dynamicet_data`; `datasets` refers to the Hugging Face package.

The full experiments require the pretrained model weights and suitable hardware.

## Exp 1: combined tracing SVG

`exp1/plot_tracing_overlay_svg.py` overlays multiple JSON results from
`exp1/tracing.py` into one layer-by-token heatmap, using the rendering from the
original `binding-iclr` combined SVG plot. It uses only the Python standard
library; plotting saved results does not require model weights or a GPU.

Run from the `dynamicET` root. For example, with the three result files used for
the object / swap target / reference box overlay:

```bash
RESULTS_DIR="results/tracing_entity"
python -m exp1.plot_tracing_overlay_svg \
  --results_paths \
    "$RESULTS_DIR/boxes_countries_n3_q0.json" \
    "$RESULTS_DIR/boxes_names_swap_target_box_swap1-0_n3_q0.json" \
    "$RESULTS_DIR/boxes_names_swap_target_box_to_remaining_context_box_swap1-0_n3_q0.json" \
  --series_labels "Object token" "Swap target box" "Reference box" \
  --metric iia \
  --output_path "$RESULTS_DIR/combined_overlay_iia.svg"
```

Set `RESULTS_DIR` and the filenames to your saved results. Filenames depend on
the tracing settings (for example, an explicit swap pair adds `_swap1-0`).
Result JSON files are not bundled with this repository. All inputs must have
matching token and layer axes and should come from the same model and prompt
setup. Relative input and output paths are resolved from your working directory.
You can also run `python exp1/plot_tracing_overlay_svg.py` with the same options.

The default palette is gold, pink, and blue and expects three inputs. For a
different number of inputs, provide one `--series_colors` hex value per file.
The default `--blend_mode max` colors each cell by its strongest visible series;
`--blend_mode alpha` blends overlapping series. Scores at or below 0.2 are white.
Use `--series_visibility_thresholds` to specify an additional cutoff per series.
Token labels start at `Context:` by default; `--include_prefix` shows the full
prompt. Use `--axis_label_prefix LABEL=PREFIX` and `--exclude_axis_label LABEL`
(repeatable) to customize labels, and `--metric mean_target_prob` to plot target
probabilities. Run with `--help` for all options.
