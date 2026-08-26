from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import as_tzyx


def _first(data, names):
    for name in names:
        if name in data:
            return data[name], name
    return None, None


def load_wind_npz(path: str | Path):
    """Load a documented NPZ wind-field schema and validate dimensions."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    raw_u, u_name = _first(data, ["u", "pred", "speed"])
    if raw_u is None:
        raise KeyError("NPZ must contain 'u', 'pred', or 'speed'")
    u = as_tzyx(raw_u)
    raw_v, _ = _first(data, ["v"])
    raw_w, _ = _first(data, ["w"])
    v = as_tzyx(raw_v) if raw_v is not None else None
    w = as_tzyx(raw_w) if raw_w is not None else None
    for name, a in [("v", v), ("w", w)]:
        if a is not None and a.shape != u.shape:
            raise ValueError(f"{name} shape {a.shape} does not match u {u.shape}")
    speed = np.hypot(u, v) if v is not None else np.abs(u)

    raw_true, true_name = _first(data, ["true_speed", "true", "reference"])
    true_speed = as_tzyx(raw_true) if raw_true is not None else None
    if true_speed is not None and true_speed.shape != speed.shape:
        raise ValueError("Reference field shape must match the evaluated field")

    if "air_mask" in data:
        mask = np.asarray(data["air_mask"], dtype=bool)
        mask = as_tzyx(mask).astype(bool)
        mask = np.broadcast_to(mask, speed.shape).copy()
    else:
        mask = np.ones(speed.shape, dtype=bool)
    mask &= np.isfinite(speed)

    z = np.asarray(data["z_abs_m"], dtype=float) if "z_abs_m" in data else None
    if z is None:
        z = np.arange(1, speed.shape[1] + 1, dtype=float)
        z_source = "index"
    else:
        z_source = "z_abs_m"
    if z.ndim != 1 or z.size != speed.shape[1]:
        raise ValueError("z_abs_m must be one-dimensional with length z")

    return {
        "path": path, "data": data, "u": u, "v": v, "w": w,
        "speed": speed, "true_speed": true_speed, "mask": mask,
        "z": z, "z_source": z_source, "u_source": u_name,
        "true_source": true_name,
        "dx": float(data["dx_m"]) if "dx_m" in data else 1.0,
        "dy": float(data["dy_m"]) if "dy_m" in data else
              (float(data["dx_m"]) if "dx_m" in data else 1.0),
        "dem": np.asarray(data["dem_m"], dtype=float) if "dem_m" in data else None,
    }
