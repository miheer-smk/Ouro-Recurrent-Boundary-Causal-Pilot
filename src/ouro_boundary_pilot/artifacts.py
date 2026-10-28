"""Identity checks and array loading for immutable, published pilot records."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

SHA256 = {
    "windows_fp16_original_v1.npz": "2218aac6b7222f426b34e8427c25160c99e5ada151393fcdaec23abbc642b2ed",
    "dgx_precision_primary_v2.npz": "3e2812f46e492d02619e72fd21ff77c5ff92dfd4daf7091f1a9fb1733d9c68a3",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(root: Path | None = None) -> tuple[dict, dict]:
    root = root or project_root()
    out = []
    for filename in SHA256:
        path = root / "results" / "raw" / filename
        actual = sha256(path)
        if actual != SHA256[filename]:
            raise ValueError(f"SHA256 mismatch: {path}: {actual}")
        with np.load(path, allow_pickle=False) as archive:
            out.append({key: np.array(archive[key], copy=True) for key in archive.files})
    original, dgx = out
    validate(original, dgx)
    return original, dgx


def validate(original: dict, dgx: dict) -> None:
    if original["margins"].shape != (30, 2, 2, 17, 2, 2):
        raise ValueError("Unexpected original pilot margins shape")
    if dgx["margins"].shape != (30, 2, 2, 17, 3, 2):
        raise ValueError("Unexpected DGX margins shape")
    if not np.isfinite(original["margins"]).all() or not np.isfinite(dgx["margins"]).all():
        raise ValueError("Non-finite intervention margins")
    if not np.array_equal(original["row_ids"], dgx["row_ids"]):
        raise ValueError("Row identities differ")
    if not np.array_equal(dgx["archived_windows_fp16_margins"], original["margins"][..., 0, :]):
        raise ValueError("Original FP16 primary array does not match embedded record")
    if list(dgx["directions"]) != ["seed42_feature2913", "seed123_feature2437"]:
        raise ValueError("Direction order altered")
    if list(dgx["arms"]) != ["dgx_fp16", "dgx_fp32_realized", "dgx_fp32_intended"]:
        raise ValueError("Arithmetic arm order altered")
    if float(dgx["alpha"]) != 0.08:
        raise ValueError("Unexpected primary alpha")
