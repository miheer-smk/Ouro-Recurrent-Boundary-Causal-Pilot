# Model-facing experimental architecture

The diagram illustrates **observable experiment-level interfaces** supported by the pinned source and archived implementation. It does **not** claim an identified mechanism or a novel internal recurrent architecture.

- The initial input is a frozen tokenized SciQ support-conditioned prompt.
- The Ouro recurrent model calculates a full-sequence state. The original execution hook acts **after model normalization** at boundary 2 or 3 and modifies the **last prompt-token only**.
- All other token states are preserved at the selected boundary; one additional recurrent iteration follows, holding remaining computation horizon equal across boundaries.
- The final readout is the gold-minus-fixed-incorrect first-token logit margin; two intervention signs and 16 matched-control vectors feed the finite-difference statistic.
- The DGX FP32 validation upcasts the FP16-loaded model and starts from DGX FP16-generated states; it does not reconstruct a pure FP32 trajectory.

Editable Mermaid sources are available in [`figures/source/`](../figures/source/). Raster/vector renders are in [`figures/`](../figures/). Inspect the original evaluator and pinned model source in the local source archive before claiming internal Ouro implementation fidelity beyond the depicted API-level flow.

## Pinned recurrent model source inspection

The pinned `modeling_ouro.py` source (SHA-256 `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413`) specifies an `embed_tokens` layer, a decoder `ModuleList` reused for each `current_ut` iteration, `OuroRMSNorm` after each decoder sweep, and an `early_exit_gate` computed for each step. `OuroForCausalLM` projects the fixed selected hidden state through `lm_head` when `exit_at_step` is requested. The public repository intentionally excludes the third-party source file pending license clearance; this schematic is a derived description, not a source-code republication. See [source-verified core schematic](../figures/rendered/00_model_core_schematic.svg) and [Mermaid source](../figures/source/ouro_core_schematic.mmd).
