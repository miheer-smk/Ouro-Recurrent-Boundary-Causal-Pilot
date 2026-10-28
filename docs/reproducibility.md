# Reproducibility guide

## Scope

The **CPU numerical audit** is reproducible from this repository's two immutable NPZ records. **Full model intervention inference** is not self-contained: Ouro weights and pinned third-party implementation, SciQ source, SAE source/checkpoints, saved control bank, and GPU environment must be obtained separately. The historical runner is an evidence artifact, not a plug-and-play independent model reproduction.

## Python setup

```bash
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/verify_artifacts.py
pytest -q
python scripts/reproduce_statistics.py --output results/derived/reproduced.json
python scripts/generate_figures.py
```

Verification checks SHA-256 of both raw arrays, their shapes, finite values, all 30 question IDs/order, directions/order, and byte-wise equality of original margins embedded in the DGX file. Statistics use 20,000 paired bootstrap question resamples with seed 20261030; alpha 0.08 for DGX precision and original primary, alpha 0.16 original secondary only. Plots and summaries are derived outputs.

## Interpreting reproducibility

Two-direction Bonferroni intervals use 97.5% individual coverage **within each mode**. Four-comparison FP32 intervals use 98.75% individual coverage **across two directions and two FP32 modes**. The four-comparison family does not account for all historic exploratory model/feature selection. The failed numerical-equivalence proposal (±0.005 T units) is a **post-pilot diagnostic**, not an original preregistration.

## Integrity evidence and missing inputs

See [provenance](source_provenance_closeout.md) and [scientific closeout](scientific_closeout.md). Historical runner SHA-256 is committed in the DGX JSON report; exact model source and original evaluator SHA-256 can be compared with local copies, but are **not** embedded into this compact public analysis repository. Do not infer full independent execution verification from CPU-only numerical reproduction.
