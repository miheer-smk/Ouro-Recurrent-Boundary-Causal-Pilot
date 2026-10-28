# Third-party assets and licensing boundaries

The repository's **original code, diagrams, text, and generated plots** are offered under its root MIT License, insofar as the author owns the requisite rights. The MIT license must **not** be interpreted as relicensing the underlying Ouro model, Hugging Face/Transformers, SciQ, the PyTorch runtime, or third-party checkpoints and datasets.

* **Ouro-1.4B model weights and source:** not distributed here. Full inference reproduction requires obtaining the correct version independently and complying with its original terms. The historical DGX runner is retained as an execution record but depends on external model code and local assets; its presence does not confer rights in those dependencies.
* **SciQ:** original dataset and prompt text are not redistributed. Public NPZ files contain numerical intervention outputs and opaque row IDs. Anyone repeating the inference must obtain SciQ from its authorized distributor and verify applicable terms.
* **SAE checkpoints, controls and model state checkpoints:** the SAE checkpoint files, original control bank, model weights, and baseline tensors are not distributed. Their SHA-256 commitments and provenance are documented in the historical closeout record.
* **Python libraries:** NumPy, Matplotlib, PyTorch, Transformers, and lm-eval retain their own licensing terms. The CPU-only reanalysis uses NumPy and Matplotlib; deep-learning dependencies are not needed.
* **Independent AI-assisted review:** the Astra assessment is labeled AI-assisted independent analysis, not formal journal peer review.

Before switching repository visibility from private to public, inspect every tracked file and confirm the licensing and authorship of any externally contributed components.

## Verified upstream licenses

- **Ouro-1.4B:** The upstream [ByteDance model card](https://huggingface.co/ByteDance/Ouro-1.4B) identifies the model license as Apache-2.0.
- **SciQ:** The upstream [AllenAI dataset card](https://huggingface.co/datasets/allenai/sciq) identifies the dataset license as Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0).

These upstream terms are separate from this repository's MIT license. This repository does not redistribute the original model weights or SciQ question text. The licensing status of research-derived numerical outputs should not be inferred solely from the absence of original text.
