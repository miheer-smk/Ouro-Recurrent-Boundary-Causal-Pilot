## Executive verdict

**GO — close the present study as an appropriately qualified pilot report. Recommendation A.**

**I do not authorize a held-out replication on the strength of this package alone.** The pilot contains reproducible statistical evidence of negative boundary contrasts, but numerical equivalence was not established, the inferential conclusions depend on the comparison family, and source inspection reveals an additional candidate/control batch-shape issue.

This is not evidence that the effects are entirely numerical artifacts. It is also not sufficient evidence for a precision-robust feature-level mechanism.

I inspected the archive contents and independently recomputed effects and bootstrap statistics from the arrays, without modifying files or executing model interventions.

## 1. Computational verification

The archive contains 13 files. Despite the request’s description, it does **not** include the original evaluator, pinned `modeling_ouro.py`, control-bank tensors, baseline-state tensors, checkpoint, or execution/preflight logs. The README acknowledges these omissions.

Consequently, this is a substantial numerical and partial source-level audit—not a complete independent execution audit.

| Item                               | Independently checked evidence                                                                                                                 | Assessment                                                                                                    |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Artifact identity                  | DGX NPZ, Windows NPZ and DGX evaluator hashes match their stated SHA-256 values                                                                | Verified                                                                                                      |
| Report/source linkage              | DGX JSON references the supplied NPZ and evaluator hashes correctly                                                                            | Verified                                                                                                      |
| Panel completeness                 | DGX margins have shape `[30,2,2,17,3,2]`; all 12,240 entries are finite; 30 unique question IDs                                                | Verified at the saved-array level                                                                             |
| Windows archive preservation       | Embedded Windows primary margins and effects equal the original NPZ exactly; question/direction ordering agrees                                | Verified                                                                                                      |
| Effect reconstruction              | Windows sensitivities, boundary differences and paired effects; DGX paired effects                                                             | All reproduce exactly                                                                                         |
| Bootstrap reconstruction           | Common question resamples, 20,000 draws, seed `20261030`                                                                                       | Reported results reproduce; audit-JSON endpoint differences are at most approximately \\(1.4\times10^{-17}\\) |
| Full-state continuation            | Wrapper supplies complete-sequence boundary states through `inputs_embeds`, disables cache, requests one loop and reads the last-token LM head | Consistent with the intended design; underlying model implementation is absent                                |
| FP32 parameter identity            | Source checks parameters and floating buffers against an exact upcast of the FP16 model                                                        | Implemented check; successful execution cannot be independently replayed here                                 |
| FP16 hook/split parity             | Source requires exact full-logit equality for every condition before completion                                                                | Strong implemented safeguard; parity tensors and execution logs are absent                                    |
| Prompt/token identity              | Wrapper checks context hashes and token counts                                                                                                 | Exact token-sequence identity is not independently verifiable; tokenizer artifacts are absent                 |
| Matching and perturbation fidelity | Control-bank hash agrees between reports                                                                                                       | Actual vectors, matching constraints and physical perturbation fidelity cannot be rechecked                   |

**A hash match authenticates the supplied bytes against a recorded commitment. It does not independently establish every upstream provenance claim.**

### What the continuation actually tests

The FP32 arms use:

- DGX FP16-generated boundary states;
- parameters and floating buffers first loaded in FP16, then upcast;
- FP32 continuation and readout;
- either FP16-rounded or FP32-constructed intervention endpoints.

This is a coherent way to investigate **downstream arithmetic sensitivity conditional on an FP16 trajectory and FP16-rounded parameter values**.

It is not an end-to-end FP32 model evaluation. It cannot recover information lost when weights or preboundary states were rounded to FP16.

The wrapper preserves the whole sequence, changes only the final prompt-token state, and disables cache. Those choices support the intended counterfactual. Nevertheless, resetting the recurrent model to one iteration also relies on the omitted implementation’s handling of positions, recurrent indexing and state initialization. Reported exact FP16 hook/split parity is strong evidence for that path; it does not independently certify every FP32 internal operation.

### Additional source-level issue: asymmetric batch shapes

This issue is not discussed in the supplied scientific audit.

`build_conditions` creates 68 conditions: two directions × 17 vectors × two signs. With the recorded batch size eight, the last chunk contains four conditions.

Those final conditions are:

- seed123 control 15, positive and negative;
- seed123 control 16, positive and negative.

Both candidates and the other controls use batch size eight. The script captures separate baseline states for batch sizes eight and four.

Therefore, **seed123’s candidate and two of its controls are evaluated from potentially different numerically realized baseline states**, even though their conceptual prompt and boundary are identical. These two controls contribute one eighth of its control mean.

Exact hook/split parity establishes agreement **within each batch shape**. It does not establish equality across batch shapes. PyTorch explicitly does not guarantee identical results between mathematically equivalent batched and individual computations. [PyTorch 2.10 documentation](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html?utm_source=chatgpt.com)

The distinction matters:

- **Demonstrated:** candidate/control assignment to different batch shapes.
- **Not demonstrated:** the resulting bias in \\(T\\), or whether it changes a conclusion.
- **Not explained by this issue:** seed42’s result; its candidate and controls all use batch size eight.

This is a material unresolved control-comparability limitation, not proof that the pilot effect is spurious.

## 2. Independently reconstructed statistical results

I reconstructed

\\[ P\_{qj} = \left[S\_{qj,\mathrm{candidate},3}       -S\_{qj,\mathrm{candidate},2}\right] - \frac1{16}\sum_k \left[S\_{qjk,3}-S\_{qjk,2}\right], \\]

then averaged over questions. All values below concern primary \\(\alpha=0.08\\).

### Two-direction family within each arithmetic mode

Each interval has **97.5% individual coverage**, corresponding to nominal 95% Bonferroni family coverage across two directions.

| Direction      | Execution mode         | Mean \\(T\\) | Two-direction-adjusted interval |
| -------------- | ---------------------- | ------------ | ------------------------------- |
| seed42 / 2913  | Windows FP16           | −0.035219    | [−0.073312, +0.007414]          |
| seed42 / 2913  | DGX FP16               | −0.033716    | [−0.077190, +0.010109]          |
| seed42 / 2913  | FP32 realized endpoint | −0.039519    | [−0.070694, −0.006235]          |
| seed42 / 2913  | FP32 intended endpoint | −0.039061    | [−0.069446, −0.006311]          |
| seed123 / 2437 | Windows FP16           | −0.039126    | [−0.071564, −0.004412]          |
| seed123 / 2437 | DGX FP16               | −0.031783    | [−0.074584, +0.010491]          |
| seed123 / 2437 | FP32 realized endpoint | −0.033467    | [−0.065711, −0.002057]          |
| seed123 / 2437 | FP32 intended endpoint | −0.033604    | [−0.065598, −0.002427]          |

The original Windows report has slightly different bootstrap endpoints. I reproduce its point estimates and underlying effects exactly, and recover the same classifications with the independently reconstructed common resampling stream. Without the original evaluator or saved bootstrap indices, I cannot authenticate the precise reason for its endpoint differences.

### Four-comparison FP32 follow-up family

If reporting individual findings across **two directions × two FP32 modes**, each interval needs 98.75% coverage for nominal 95% Bonferroni family coverage:

| Direction      | FP32 realized endpoint | FP32 intended endpoint |
| -------------- | ---------------------- | ---------------------- |
| seed42 / 2913  | [−0.074207, −0.001588] | [−0.072746, −0.001839] |
| seed123 / 2437 | [−0.069134, +0.001204] | [−0.069001, +0.000971] |

**Only seed42 excludes zero in both FP32 modes under this four-comparison procedure.**

That is a legitimate numerical sensitivity result. It does not retroactively make seed42 a successful original primary pilot result, or constitute replication of seed123.

### Which multiplicity family is appropriate?

The two-direction procedure is correct **within a specified mode**. It does not jointly cover claims selected across both FP32 modes.

For this retrospective report, I recommend showing both tables and using the four-comparison table when discussing individual FP32 findings collectively.

Four comparisons are not universally mandatory merely because two numerical modes exist. A prospectively specified requirement that a direction be negative in **both** modes could instead be an intersection–union claim. Likewise, one prospectively primary mode with a diagnostic secondary mode would define a different testing structure.

Neither alternative should be retroactively invoked to obtain the preferred conclusion here.

Also, four-comparison adjustment addresses that defined follow-up family. It does not account for every historical analytical choice or turn reused pilot questions into untouched confirmation.

### Is the bootstrap reasonable?

Yes, as an **approximate pilot-level procedure**, conditional on the selected directions and controls:

- Questions are resampled as paired units.
- Candidate and control results stay together.
- Arithmetic modes share resampled question identities.
- Controls and signs are not treated as independent observations.

Its limitations remain substantial:

- Thirty questions provide limited information about the population distribution.
- Percentile-bootstrap tail coverage is approximate; Bonferroni does not repair imperfect individual coverage.
- The four-comparison endpoints use approximately the outer 125 bootstrap draws in each tail.
- A bootstrap does not repair outcome-dependent selection or establish representative sampling.
- The supplied package does not independently establish the full selection history.

The leave-one-out means are all negative, which rules out reversal of the mean from removing any single question. It does **not** establish leave-one-out significance, numerical equivalence or population generalization.

## 3. Numerical equivalence: failed certification, not established inequality

The reconstructed FP32-minus-DGX-FP16 discrepancies are:

| Direction | FP32 mode | Mean discrepancy | Paired 95% interval    | Question-level RMS | Sign changes |
| --------- | --------- | ---------------- | ---------------------- | ------------------ | ------------ |
| seed42    | Realized  | −0.005804        | [−0.027076, +0.014249] | 0.057542           | 9/30         |
| seed42    | Intended  | −0.005345        | [−0.026580, +0.014792] | 0.057690           | 9/30         |
| seed123   | Realized  | −0.001684        | [−0.023426, +0.019664] | 0.060725           | 9/30         |
| seed123   | Intended  | −0.001821        | [−0.023310, +0.019345] | 0.060240           | 8/30         |

**All four fail the proposed ±0.005 equivalence gate.**

Three distinctions are essential.

**First, the finite-panel discrepancies are observed quantities.** Their bootstrap intervals describe uncertainty under question resampling, not uncertainty about the arithmetic values already stored.

**Second, every discrepancy interval includes zero.** The study does not resolve a nonzero population mean precision shift. Nor does it establish that the population discrepancy lies outside the equivalence band.

**Third, failure to demonstrate equivalence leaves consequential differences possible.** Intervals spanning roughly −0.02 to +0.02 are broad relative to effects around −0.03 to −0.04.

The ±0.005 band was a post-pilot review criterion, not an original discovery threshold or an independently justified scientific equivalence margin. It was defensible as a demanding validation target. Its failure must remain reported; it should not be relaxed after seeing these results.

### Endpoint rounding versus downstream arithmetic

The intended-minus-realized FP32 comparisons are much smaller:

| Direction | Mean difference | Paired 95% interval    |
| --------- | --------------- | ---------------------- |
| seed42    | +0.000459       | [−0.000939, +0.001781] |
| seed123   | −0.000137       | [−0.001436, +0.001150] |

Their question-level RMS differences are approximately 0.00383 and 0.00355.

This supports the narrower observation that **endpoint rounding has a smaller measured effect than changing downstream arithmetic in this panel**. It does not show that all state quantization is harmless: both modes retain FP16-generated baseline states and FP16-rounded parameters.

DGX-versus-Windows FP16 question-level RMS discrepancies are approximately **0.07428** and **0.06630**. Thus the original Windows computation is not reproduced question by question on DGX.

I also verified that approximately **8.63%** of DGX FP16 positive/negative margin pairs coincide, whereas none coincide in the FP32 arms. This is consistent with limited FP16 resolution, but is not a causal decomposition of the observed effect.

Overall, the precision results undermine a claim of **numerically stable certification of the original inference**. They do not demonstrate that its negative mean was created by rounding.

## 4. What causal conclusion survives?

The strongest defensible conclusion is:

> In the recorded 30-question support-conditioned SciQ pilot, both frozen SAE-derived directions exhibited negative mean candidate-minus-control contrasts in centered answer-margin sensitivity between boundaries 3 and 2 across all four examined execution modes. FP32 follow-up analyses provide exploratory statistical support, strongest for seed42 under the four-comparison family, but numerical equivalence and complete implementation-level validation remain unresolved.

The intervention supplies causal evidence about the response to an imposed state perturbation, conditional on execution validity. That is stronger than a feature–outcome correlation.

However, the estimand does **not** isolate an abstract “effect of being at loop 3”:

- Boundary 2 and boundary 3 contain different full-sequence states.
- Their subsequent attention and nonlinear transformations operate in different local contexts.
- Answer-margin response and calibration can differ.
- Exactly one remaining loop removes the unequal-horizon confound, but not these state differences.

Moreover, \\(S\\) is a finite-amplitude centered response. It is not automatically an infinitesimal derivative.

A negative \\(T\\) means the candidate’s boundary-3-minus-boundary-2 sensitivity is more negative than the control average. It does not by itself mean the candidate helps at boundary 2, harms at boundary 3, improves accuracy, or carries useful reasoning.

### What the controls establish—and do not

Matching norm, gate projection and one answer-margin projection defines a useful conditional reference. It does not match the entire vocabulary response, state geometry or recurrent Jacobian.

Consequently, even a reliable candidate/control difference could reflect ordinary geometric differences between directions rather than a privileged semantic feature.

The actual matching cannot be independently rechecked from this bundle because the vectors and matching implementation are absent. The batch-shape issue adds a specific comparability concern for seed123.

### Unsupported claims

The evidence does not establish:

- Semantic identity of either SAE feature.
- A feature shared across SAE seeds.
- Mediation of a particular reasoning algorithm.
- Necessity of either direction for answering.
- Utility beyond the native gate and entropy.
- Generalization beyond the selected prompts and readout.
- Independent replication.
- End-to-end FP32 or hardware-independent reproducibility.

The two seed-specific directions are two physical interventions, not two replications of one identified feature.

## 5. Why recommendation A is preferable

**A preserves the actual contribution:** an auditable causal pilot with exact array-level reconstruction, negative mean effects, and an informative numerical-validation failure.

**B is premature on the available evidence.** The proposed equivalence gate did not pass; the complete execution dependency chain is unavailable for review; and the source exposes a control/batch-shape limitation. Treating an FP32 confidence interval as sufficient authorization would bypass the validation question that motivated the follow-up.

**C would overstate the negative evidence.** No analysis here demonstrates that the effects vanish, reverse consistently, or are entirely numerical. Semantic mechanism identification was never established, so these results cannot decisively falsify such a mechanism either.

The appropriate stopping point is therefore **closure of this pilot, without authorizing another experiment in this review**.

The remaining work is documentary and bounded:

1. Preserve all original and follow-up results without replacing the original inferential record.
2. Report both multiplicity families, failed equivalence, and the batch-shape limitation.
3. Identify missing implementation artifacts and logs explicitly; attach existing ones if available, without claiming validations that cannot be independently inspected.

No additional feature screening, intervention magnitude selection or retrospective acceptance threshold is justified.

## Publication-suitable wording

> We evaluated two frozen SAE-derived directions using paired positive and negative perturbations at recurrent boundaries 2 and 3 in Ouro-1.4B, followed by one additional recurrent iteration, on 30 support-conditioned SciQ questions. Mean candidate-minus-matched-control boundary contrasts were negative in the original Windows FP16 evaluation and all three DGX numerical-validation modes. Independent reconstruction reproduced the stored effects and reported follow-up bootstrap statistics. Both directions excluded zero under within-mode two-direction adjustment in FP32; only the seed42 direction excluded zero when jointly adjusting across both directions and both FP32 modes. The proposed FP16–FP32 numerical-equivalence criterion was not met, and individual-question effects were numerically sensitive. Source review also identified unequal batch shapes for two seed123 controls. These results constitute exploratory evidence of boundary-dependent perturbation responses under the specified implementation, rather than independent replication, semantic feature identification, or identification of a recurrent reasoning mechanism.