# Ouro-1.4B SAE-derived recurrent loop-boundary pilot — final scientific closeout

**Decision:** **GO — close as a qualified exploratory causal pilot** (Astra Recommendation A).  
**Not established:** numerical equivalence; independent held-out replication; SAE feature semantics; an identified recurrent reasoning mechanism.  
**Report date:** 2026-10-28. **Scope:** existing 30-question SciQ primary pilot and post-pilot numerical audit. No new experimental selection or interventions.

## Executive summary

On 30 frozen support-conditioned SciQ questions, two previously selected SAE-derived directions displayed negative mean candidate-minus-16-control contrasts in finite-amplitude answer-margin sensitivity between recurrent boundaries 3 and 2. This descriptive sign persisted in archived Windows FP16 and all three DGX numerical modes. The original Windows FP16 analysis supported **seed123_feature2437** under its original two-direction familywise procedure; follow-up DGX FP32 within-mode analyses excluded zero for both directions. **When the four FP32 direction × precision comparisons are jointly adjusted, only seed42_feature2913 excludes zero in both FP32 modes.** This does not convert it into a preregistered discovery, independent replication, or a mechanism.

The post-pilot proposed FP16–FP32 equivalence criterion (±0.005 T units) was **not met** in any of the four direction/mode comparisons. Confidence intervals for the mean *differences* include zero, so failure to establish equivalence is not proof of non-equivalence. Individual-question effects can differ materially. A source-level review independently identified that in DGX version 2 two seed123 controls run in a four-sample trailing chunk while candidate and other controls use eight-sample chunks; FP16 baseline states are batch-shape-sensitive. This is a potential, **unquantified** seed123 control-comparability confound.

The final verdict is to **preserve and report the pilot as a qualified exploratory result** and **not authorize additional feature-level experiments or stronger mechanistic claims on the basis of this evidence alone**.

## Research question and frozen estimand

**Question:** Do frozen SAE-derived directions change the response of Ouro-1.4B to recurrent loop-boundary interventions differently from matched random directions?

- Model: Ouro-1.4B; task: support-conditioned first-continuation-token SciQ answer contrast.
- Panel: 30 fixed evaluation questions; candidates: `seed42_feature2913`, `seed123_feature2437`; each compared with 16 fixed matched controls per question.
- Recurrent intervention boundaries: 2 and 3, each followed by **exactly one further loop**. Patch only the final prompt-token post-norm state. No cache or parameter updates.
- Original pilot perturbations: primary α=0.08; secondary α=0.16. The DGX FP32 follow-up evaluates **primary α=0.08 only**.
- Readout: correct minus frozen incorrect first-continuation-token logit margin. For vector v, boundary b and amplitude α:

  `S(v,b,α) = [m(+αv,b) − m(−αv,b)] / (2α)`

  `D(v) = S(v,3,α) − S(v,2,α)`

  `P(q,j) = D(candidate(q,j)) − mean_16 D(control(q,j,k))`

  `T(j) = mean_30 P(q,j)`.

**Interpretation:** Negative T means the candidate's boundary-3-minus-boundary-2 finite-response slope is more negative than the matched-controls' average. This is not an infinitesimal derivative, evidence of accuracy improvement, nor evidence of a particular semantic feature.

## Numerical-validation design

The DGX follow-up collected **12,240 saved margins**: 30 questions × 2 boundaries × 2 directions × 17 variants × 2 signs × 3 arithmetic modes. The modes were DGX native FP16, FP32 continuation from FP16-realized rounded endpoints, and FP32 continuation from intended unrounded endpoints. FP32 continuation starts from **DGX FP16-generated complete-sequence boundary states** and a model loaded in FP16 then upcast to FP32: **not** an end-to-end FP32 checkpoint or trajectory validation.

The v2 runner verifies exact original FP16 hook versus split-replay output for every condition using matching batch shapes. The initial v1 preflight failed this requirement and was preserved; a diagnostic showed that capturing states at batch size 8 restores exact parity for tested conditions. The v2 full run subsequently passed its hook/split parity checks and completed all 30 questions. **Within-DGX parity does not imply cross-hardware FP16 equality.** Full underlying implementation was not independently executed by Astra.

## Main results — effect T and question-bootstrap intervals

20,000 paired question resamples, RNG seed `20261030`; individual 97.5% percentile intervals provide nominal Bonferroni 95% coverage **within each two-direction arithmetic mode**. Individual 98.75% percentile intervals are provided for the **joint four-comparison FP32 follow-up family**. These confidence intervals are approximate for n=30 and cannot correct outcome-dependent feature selection.

| Direction | Mode | Mean T | Two-direction interval | Four-comparison FP32 interval | Negative P(q) |
|---|---|---:|---|---|---:|
| seed42_feature2913 | Archived Windows FP16 | -0.035219 | [-0.073312, +0.007414] | — | 21/30 |
| seed42_feature2913 | DGX FP16 | -0.033716 | [-0.077190, +0.010109] | — | 21/30 |
| seed42_feature2913 | DGX FP32 realized | -0.039519 | [-0.070694, -0.006235] | [-0.074207, -0.001588] | 24/30 |
| seed42_feature2913 | DGX FP32 intended | -0.039061 | [-0.069446, -0.006311] | [-0.072746, -0.001839] | 24/30 |
| seed123_feature2437 | Archived Windows FP16 | -0.039126 | [-0.071564, -0.004412] | — | 24/30 |
| seed123_feature2437 | DGX FP16 | -0.031783 | [-0.074584, +0.010491] | — | 20/30 |
| seed123_feature2437 | DGX FP32 realized | -0.033467 | [-0.065711, -0.002057] | [-0.069134, +0.001204] | 25/30 |
| seed123_feature2437 | DGX FP32 intended | -0.033604 | [-0.065598, -0.002427] | [-0.069001, +0.000971] | 24/30 |

**Inference:** Original Windows FP16: seed123 excludes zero, seed42 does not. DGX FP16: neither excludes zero at this correction. Separately within each FP32 mode: both exclude zero. **Across all four FP32 direction×mode comparisons:** only the two seed42 FP32 intervals exclude zero. Both FP32 follow-ups reuse the original questions and must not be represented as replication.

Astra independently reproduced the effects and reported follow-up bootstrap calculations from the arrays. Small differences from original Windows report confidence endpoints arise with the independently reconstructed common resampling stream; the original raw means and effects match exactly. The stronger four-comparison family is appropriate for reporting the aggregate post-hoc set; alternatives must not be chosen retrospectively to obtain significance.

## Numerical precision and equivalence

The proposed equivalence criterion requires a paired bootstrap 95% CI for FP32-minus-DGX-FP16 mean T to lie **wholly inside ±0.005**. This post-pilot criterion was not a preregistered original discovery threshold.

| Direction | Comparison | Mean ΔT | Paired 95% CI | Question RMS ΔT | Sign changes |
|---|---|---:|---|---:|---:|
| seed42_feature2913 | FP32 realized | -0.005804 | [-0.027076, +0.014249] | 0.057542 | 9/30 |
| seed42_feature2913 | FP32 intended | -0.005345 | [-0.026580, +0.014792] | 0.057690 | 9/30 |
| seed123_feature2437 | FP32 realized | -0.001684 | [-0.023426, +0.019664] | 0.060725 | 9/30 |
| seed123_feature2437 | FP32 intended | -0.001821 | [-0.023310, +0.019345] | 0.060240 | 8/30 |

**Equivalence: NOT ESTABLISHED (all four comparisons).** All four difference CIs include zero. Question-level numerical discrepancies are material relative to mean effects. In addition, DGX FP16 does not reproduce archived Windows FP16 question by question: RMS T differences are approximately 0.07428 for seed42 and 0.06630 for seed123.

Intended-minus-realized FP32 mean T is approximately +0.000459 (seed42) and −0.000137 (seed123), with 95% intervals crossing zero. Thus endpoint rounding is a relatively small observed source of average variation in this panel, **not** proof that all quantization effects are harmless.

## Adversarial source issue — condition/chunk comparability

The DGX v2 evaluator has 68 conditions per boundary. With batch size 8, the last chunk contains four conditions: two controls for `seed123_feature2437`, each with ±sign. Because GPU FP16 baseline states were observed to depend on batch size, these two control conditions use potentially different numerical baseline states from the candidate and 14 other controls. Two of 16 controls represent **one-eighth** of the control mean. Original-hook versus split-replay equality **within** each shape does not eliminate **between-shape** differences. The size or direction of any bias to T is **not established**. The control-bank matching vectors were unavailable to Astra within the audit bundle. This issue does not explain seed42's analogous contrast, where the v2 conditions all used batch size 8.

Do not silently recompute, repair or replace the archived results. Document the discovered limitation as an integrity-relevant negative finding. This issue is confirmed for DGX v2 from the independent source review; the original Windows implementation remains a separately archived source record.

## Claims and limitations

**Defensible:** This exploratory, specified-prompt and specified-readout study shows negative *mean* boundary-dependent candidate-minus-control finite perturbation responses under recorded arithmetic modes, with certain mode- and family-specific bootstrap exclusions of zero. It demonstrates a rigorous numerical-sensitivity audit and exposes a batch-shape comparability concern.

**Not defensible:** semantic interpretation of either SAE feature, common semantic identity across seeds, a particular recurrent algorithm, causal necessity for successful answering, effect on answer accuracy, strong generalization beyond these prompts, end-to-end FP32 reproduction, numerical equivalence, or independent replication. Control matching on norm, gate projection and one margin projection does not match the entire recurrent Jacobian or vocabulary distribution. Boundaries differ in preintervention states and local processing context even with equal remaining loop count.

## Evidence grading

- **Array-verified:** hashes of the supplied original and DGX NPZ, complete finite array shapes, paired-effect reconstruction, interval recreation and archived primary α identity.
- **User-verified on Windows:** seven SHA-256 critical-file checks all PASS; preserved files include pinned modeling source, original evaluator, controls and baseline-state capture. Details in `PROVENANCE_AND_FILES.md`. These checks establish identity against recorded hashes, not full independent execution verification.
- **Execution-log-supported, not independently replayed by Astra:** full DGX run, parameter upcast checks, exact full-logit split parity, prompt-context validation and no unintended modifications.
- **Source-audited:** DGX script structure, FP32 continuation design, trailing-chunk issue; independent implementation-level certification remains incomplete.

## Final decision and future scope

**GO — close the pilot as an exploratory report, retaining the complete negative/qualified evidence.** No new magnitude selection, feature screening, or retrospective criterion shifts. This decision does not permanently prohibit future independent research; a new hypothesis and predeclared design would require a separate scientific decision and clearer validation foundations.

## Files and reproducing statistical audit

- `evidence/`: unchanged original and DGX numerical records, archived original report and source of DGX v2 evaluator.
- `independent_audit/`: independently written audit script and recreated JSON output, accompanied by immutable audit input copies.
- `review/ASTRA_FINAL_REVIEW_VERBATIM.md`: complete external read-only assessment, unedited.
- `figures/`: numerical effect CI and per-question FP16/FP32 comparison charts.
- `RESULTS_TABLE.csv`: machine-readable effect and interval summary.
- `PROVENANCE_AND_FILES.md`: paths, provenance checks, missing-package-versus-absent-in-project distinction.

Run `python independent_audit/independent_precision_audit.py` in a Python environment with NumPy available. This does **not** run a model or require a GPU; its program writes `independent_precision_audit.json` within its own directory. Follow `README.md` for Windows installation.
