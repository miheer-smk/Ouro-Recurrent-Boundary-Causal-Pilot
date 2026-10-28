# Scientific limitations and negative findings

1. **Numerical equivalence was not established.** Every FP32-minus-FP16 paired-mean discrepancy 95% interval extends outside the Astra-proposed ±0.005 band. Lack of equivalence evidence is not proof of unequal population means.
2. **Question-level numerical instability.** Eight or nine of 30 question-level paired contrasts change sign depending on FP32 mode; DGX FP16 and Windows FP16 also differ question by question.
3. **Batch-shape confound.** In DGX v2, candidate and fourteen `seed123` controls execute at batch size eight; two controls execute in a trailing batch of four. Preflight showed batch shape can change FP16 baseline states. The bias on T remains unquantified. This was not retrospectively repaired.
4. **FP32 interpretation.** DGX FP32 is **one downstream recurrent iteration**, starting from DGX FP16 boundary trajectories with upcast FP16 checkpoint weights; not end-to-end FP32.
5. **Post-hoc multiplicity.** Four-comparison FP32 family changes the status of seed123 relative to within-mode two-direction families. These follow-ups reuse the same 30 questions and do not constitute replication.
6. **Not a mechanistic identification study.** Two learned SAE directions are not verified to share a semantic identity, embody a reasoning procedure, mediate correct answers, or generalize to held-out prompts.
7. **Comparator matching is limited.** Matched controls constrain selected gate and answer-margin projections, not the entire local state geometry or logit/Jacobian response.
8. **Finite perturbations and sample size.** Centered finite-amplitude responses are not infinitesimal derivatives. The 30 frozen questions are a small selected exploratory panel; bootstrap coverage is approximate.
9. **External dependencies.** Complete model implementation and inference inputs are not in this public-safe release; numerical replay only is supplied.

The final independent AI-assisted Astra review recommended **closing the pilot with these caveats**, not further experiments or publication of a specific reasoning mechanism.
