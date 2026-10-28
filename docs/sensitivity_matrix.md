# Completed sensitivity analyses (not all independent ablations)

| Comparison | Evidence | Status | Important limit |
|---|---|---|---|
| Alpha 0.08 vs 0.16 | Original Windows FP16 | Completed | Same selected questions; no DGX 0.16 replay |
| Candidate vs 16 matched controls | Original Windows and DGX | Completed | Matching on limited projections |
| Native FP16 vs FP32 realized endpoint | DGX α=0.08 | Completed | Equivalence not established |
| FP32 realized vs intended endpoints | DGX α=0.08 | Completed | FP16 preboundary states and weights remain |
| DGX FP16 vs Windows FP16 | Paired archive | Completed | Different hardware and arithmetic implementation |
| Batch-8 vs batch-4 intervention chunks | DGX v2 diagnostic | Batch dependence demonstrated, T bias unquantified | Not a corrected pilot or an independent ablation |
| Independent held-out questions | None | **Not performed** | No replication claim |
| End-to-end FP32 forward computation | None | **Not performed** | No full FP32 trajectory claim |
| Feature semantic ablation | None | **Not performed** | No identified reasoning algorithm |

No experiments were added or modified to construct this repository.
