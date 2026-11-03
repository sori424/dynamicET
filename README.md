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
