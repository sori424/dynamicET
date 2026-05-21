# Mechanisms of dymamic entity rebinding in large language models

This repository contains the code used for the experiments in the paper Mechanisms of dymamic entity rebinding in large language models. We study how instruction-tuned LLMs {Gemma-9B, Gemma-12B, Llama-3B, Llama-8B}-it implement rebinding in a dynamic state tracking. 

We conduct our experiments in four steps:

- **Exp 1.** Interchange intervention in the residual stream at target tokens (Section 3)
- **Exp 2.** Path patching for circuit analysis (Section 4.1)
- **Exp 3.** Interchange intervention in attention patterns (Section 4.2)
- **Exp 4.** Binding ID intervention (Section 5)
