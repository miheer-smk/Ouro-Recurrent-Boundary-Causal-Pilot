"""Reanalysis of the saved intervention margins; no GPU needed.

Axis definitions:
  original: [question,boundary,direction,variant,alpha,sign]
  DGX:      [question,boundary,direction,variant,arm,sign]
  variant 0=candidate, 1..16=controls; sign 0=positive, 1=negative.
"""
from __future__ import annotations
import numpy as np

SEED = 20261030
REPS = 20000
DIRECTIONS = ("seed42_feature2913", "seed123_feature2437")
ARMS = ("Windows FP16", "DGX FP16", "DGX FP32 realized", "DGX FP32 intended")


def paired_question_effects(margins: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """Return [question,direction,comparison] paired candidate-control effects."""
    m = np.asarray(margins, dtype=np.float64)
    if m.shape[:4] != (30, 2, 2, 17) or m.shape[-1] != 2:
        raise ValueError("Unexpected margins shape")
    if not np.isfinite(m).all():
        raise ValueError("Non-finite margins")
    if np.any(np.asarray(alpha) <= 0):
        raise ValueError("alpha must be positive")
    a = np.asarray(alpha, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape((1,) * (m.ndim - 2) + (len(a),))
    slope = (m[..., 0] - m[..., 1]) / (2.0 * a)
    boundary_diff = slope[:, 1] - slope[:, 0]
    return boundary_diff[:, :, 0] - boundary_diff[:, :, 1:].mean(axis=2)


def bootstrap_indices(n: int = 30, reps: int = REPS, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(reps, n))


def ci(samples: np.ndarray, coverage: float) -> list[float]:
    tail = (1.0 - coverage) / 2
    return [float(v) for v in np.quantile(samples, [tail, 1 - tail])]


def calculate(original: dict, dgx: dict, reps: int = REPS, seed: int = SEED) -> dict:
    window = paired_question_effects(original["margins"], original["alphas"])
    panel = paired_question_effects(dgx["margins"], float(dgx["alpha"]))
    if not np.array_equal(window, original["paired_effects"]):
        raise ValueError("Original paired effect reconstruction failed")
    if not np.array_equal(panel, dgx["paired_effects"]):
        raise ValueError("DGX paired effect reconstruction failed")
    if not np.array_equal(window[..., 0], dgx["archived_windows_fp16_paired_effects"]):
        raise ValueError("DGX copy of original primary effects mismatch")
    draws = bootstrap_indices(30, reps, seed)
    winboot = window[draws].mean(axis=1)
    dgxboot = panel[draws].mean(axis=1)
    means = np.concatenate((window[..., 0, None], panel), axis=-1)
    boots = np.concatenate((winboot[..., 0, None], dgxboot), axis=-1)
    items = []
    for j, d in enumerate(DIRECTIONS):
        for k, name in enumerate(ARMS):
            p = means[:, j, k]
            items.append({
                "direction": d,
                "mode": name,
                "mean_T": float(p.mean()),
                "CI_97p5pct_two_direction": ci(boots[:, j, k], 0.975),
                "CI_98p75pct_four_FP32": ci(boots[:, j, k], 0.9875) if k in (2,3) else None,
                "negative_questions": int((p < 0).sum()),
            })
    differences = []
    for j,d in enumerate(DIRECTIONS):
        for arm in (1,2):
            diff = panel[:,j,arm] - panel[:,j,0]
            bs = dgxboot[:,j,arm] - dgxboot[:,j,0]
            interval = ci(bs,0.95)
            differences.append({
                "direction": d,
                "mode": ARMS[arm+1],
                "mean_discrepancy": float(diff.mean()),
                "CI_95pct": interval,
                "question_RMS": float(np.sqrt(np.mean(diff**2))),
                "question_sign_changes": int(np.count_nonzero(np.sign(panel[:,j,arm])!=np.sign(panel[:,j,0]))),
                "equivalence_within_plus_minus_0p005_established": bool(interval[0]>-0.005 and interval[1]<0.005),
            })
    return {"status":"completed exploratory pilot; numeric equivalence NOT established",
            "bootstrap_seed":seed,"bootstrap_draws":reps,"rows":items,"precision_discrepancies":differences,
            "original_secondary_alpha_effects": [float(v) for v in window[...,1].mean(axis=0)]}
