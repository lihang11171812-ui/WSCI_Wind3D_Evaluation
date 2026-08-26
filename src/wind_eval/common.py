from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

EPS = 1e-12


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def as_tzyx(a: np.ndarray) -> np.ndarray:
    """Convert (y,x), (z,y,x), or (t,z,y,x) to (t,z,y,x)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 2:
        return a[None, None, ...]
    if a.ndim == 3:
        return a[None, ...]
    if a.ndim == 4:
        return a
    raise ValueError(f"Expected 2-D, 3-D, or 4-D grid; got shape {a.shape}")


def weighted_available(values: Mapping[str, float | None],
                       weights: Mapping[str, float]):
    usable = {k: float(weights[k]) for k, v in values.items()
              if v is not None and np.isfinite(v)}
    total = sum(usable.values())
    if total <= 0:
        return None, {}
    effective = {k: w / total for k, w in usable.items()}
    score = sum(effective[k] * float(values[k]) for k in effective)
    return clip01(score), effective


def _gaussian_kernel1d(sigma: float):
    if sigma <= 0:
        raise ValueError("gaussian sigma must be positive")
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _separable_blur2d(a: np.ndarray, sigma: float):
    kernel = _gaussian_kernel1d(sigma)
    radius = len(kernel) // 2
    padded = np.pad(a, ((radius, radius), (0, 0)), mode="reflect")
    temp = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"),
                               0, padded)
    padded = np.pad(temp, ((0, 0), (radius, radius)), mode="reflect")
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="valid"),
                              1, padded)


def masked_blur2d(a: np.ndarray, mask: np.ndarray, sigma: float):
    num = _separable_blur2d(np.where(mask, a, 0.0), sigma)
    den = _separable_blur2d(mask.astype(float), sigma)
    return num / np.maximum(den, EPS)


def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(result), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")
