# Evidence location, SHA-256 identities and reproducibility scope

The final bundle is deliberately **not** a second copy of the entire multi-GB Ouro model and Hugging Face cache. It is a numerical and documentary closeout package. The original source/data files should remain at the Windows project paths below, unchanged.

## Original Windows project (user verified 2026-10-28)

| Item | Expected project-relative path | Recorded SHA-256 | User-reported status |
|---|---|---|---|
| Original evaluator | `run_boundary_causal_evaluation.py` | `a7872dabcfbdaac54dd3082e82da25b1ed98f7f4fac34b838efb0d8e67e77693` | PASS |
| Pinned modeling source | `hf_cache/modules/transformers_modules/ouro_model/modeling_ouro.py` | `c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413` | PASS |
| Matched control tensors | `results/boundary_pilot_matched_controls_v1.pt` | `0ecaf8237652307ab727dcca97b79e04902c2025584c4f778f2cbefad040f3cc` | PASS |
| Baseline-state tensor | `results/boundary_pilot_baseline_states_v1.pt` | `824d09c1addcdae7ac0d7dd567132e7ea77e7bf9e9b47ace2091456c81d1fb54` | PASS |
| Original raw NPZ | `results/boundary_pilot_causal_evaluation_v1.npz` | `2218aac6b7222f426b34e8427c25160c99e5ada151393fcdaec23abbc642b2ed` | PASS |
| DGX evaluator v2 | `results/dgx_validation/dgx_source_snapshot/ouro_dgx_fp32_primary_panel_v2.py` | `8e1f4fa592c1a514632597032d6423b710159314ee1e7f345743a5090ffab69f` | PASS |
| DGX primary NPZ | `results/dgx_validation/dgx_source_snapshot/fp32_robustness/fp32_primary_full_v2.npz` | `3e2812f46e492d02619e72fd21ff77c5ff92dfd4daf7091f1a9fb1733d9c68a3` | PASS |

User saved an integrity report at `results/dgx_validation/final_closeout/critical_file_integrity_v1.json`. That local report is not included in this package because it was not uploaded; status values above are based on the user's provided PowerShell output. It should remain preserved and is checked again by `scripts/verify_windows_sources.ps1`.

## Additional expected original inputs

The data Arrow, contrasts JSON, model weights, original source dependencies, tokenizer cache and matching implementation are within the original Windows project and/or pinned HF cache. The DGX FP32 report records dataset hash `656cdf16f40617360683b039b8c2a2b48b479be06d1df5c8b023afbb648da9bd`, contrast JSON hash `b0a2412b736c25e69826e110d067c043f572517eaeb0e11cb37ee96a699fd1cd`, and model-weights hash `58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af`. No claim is made that this closeout ZIP independently contains or rechecks the full model checkpoint.

## DGX provenance

Device NVIDIA GB10, aarch64, CUDA 13.0; Docker image local tag `ouro-research:25.11` based on NVIDIA PyTorch 25.11; PyTorch `2.10.0a0+b558c986e8.nv25.11` and Python 3.12.3, as recorded by the DGX report. Batching selected at 8 with final partial chunk 4. Full run 30/30 finished exit status 0, reported elapsed ~269.3 s. Cross-hardware Windows RTX 4060 original and DGX GB10 follow-up must not be merged as if identical numerical execution.

## External DGX evidence snapshot

Location: `results/dgx_validation/dgx_source_snapshot/` (21 user-reported files), including FP16 split replay adaptations, the original failed v1 preflight, FP16 batch diagnostic, v2 passed preflight, full v2 run log, analysis JSON, results NPZ and checkpoint. These are **not duplicated** in this closeout ZIP. Keep the complete snapshot unchanged.

## Scope of independent review

Astra received a 13-file numerical audit package that omitted the original evaluator, pinned modeling source, actual control and baseline tensors, full checkpoint and execution logs. They **do exist in the user's Windows project** and the 7 named critical files passed user-run checks. Their absence from the audit package prevents claiming that Astra independently verified their implementation or re-executed them. This closeout likewise avoids overclaiming a complete source-level audit.

## Tamper-evident package manifest

`BUNDLE_SHA256_MANIFEST.txt` lists exact relative SHA-256 hashes for the files in this bundle. It verifies these packaged copies, not all files in the user's original project or the DGX evidence snapshot.
