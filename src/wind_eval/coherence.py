from __future__ import annotations

import numpy as np

from .common import EPS, clip01, weighted_available
from .io import load_wind_npz
from .variogram import spatial_semivariogram, temporal_semivariogram, variogram_scores

TOP_WEIGHTS = {"space": 0.45, "time": 0.35, "terrain": 0.20}
SPACE_WEIGHTS = {"P_mono": 0.60, "P_small": 0.40}
TIME_WEIGHTS = {"P_persist": 0.40, "P_mono": 0.35, "P_small": 0.25}


def _space(speed, mask, dx, max_lag):
    gamma, pairs = spatial_semivariogram(speed, mask, max_lag)
    sub = variogram_scores(gamma)
    score, effective = weighted_available(
        {k: sub[k] for k in SPACE_WEIGHTS}, SPACE_WEIGHTS)
    return {"score": score, "effective_weights": effective, **sub,
            "lag_m": (np.arange(1, max_lag + 1) * dx).tolist(),
            "gamma_m2_s2": gamma.tolist(), "pair_count": pairs.tolist()}


def _time(speed, mask, dt, max_lag):
    if speed.shape[0] < 3:
        return {"score": None, "reason": "fewer than 3 time frames"}
    gamma, pairs, corr = temporal_semivariogram(speed, mask, max_lag)
    sub = variogram_scores(gamma)
    persistence = clip01((corr[0] + 1.0) / 2.0) if np.isfinite(corr[0]) else None
    values = {"P_persist": persistence, "P_mono": sub["P_mono"],
              "P_small": sub["P_small"]}
    score, effective = weighted_available(values, TIME_WEIGHTS)
    return {"score": score, "effective_weights": effective, **sub,
            "P_persist": persistence,
            "lag_seconds": (np.arange(1, len(gamma) + 1) * dt).tolist(),
            "gamma_m2_s2": gamma.tolist(), "pair_count": pairs.tolist(),
            "pattern_correlation": corr.tolist()}


def _terrain(u, v, w, mask, dem, dx, dy):
    if v is None or w is None or dem is None:
        return {"score": None, "reason": "terrain term requires u, v, w and dem_m"}
    if dem.shape != u.shape[-2:]:
        return {"score": None, "reason": "dem_m shape does not match (y,x)"}
    dHdy, dHdx = np.gradient(dem, dy, dx)
    residuals, speeds = [], []
    for t in range(u.shape[0]):
        available = mask[t].any(axis=0)
        first = np.argmax(mask[t], axis=0)
        yy, xx = np.where(available)
        zz = first[yy, xx]
        uu, vv, ww = u[t, zz, yy, xx], v[t, zz, yy, xx], w[t, zz, yy, xx]
        residuals.append(ww - uu * dHdx[yy, xx] - vv * dHdy[yy, xx])
        speeds.append(np.sqrt(uu * uu + vv * vv + ww * ww))
    residual, speed = np.concatenate(residuals), np.concatenate(speeds)
    ok = np.isfinite(residual) & np.isfinite(speed)
    if ok.sum() < 3:
        return {"score": None, "reason": "insufficient near-surface samples"}
    rmse = float(np.sqrt(np.mean(residual[ok] ** 2)))
    rms_speed = float(np.sqrt(np.mean(speed[ok] ** 2)))
    return {"score": clip01(np.exp(-rmse / (rms_speed + EPS))),
            "definition": "exp[-RMSE(w-u*dH/dx-v*dH/dy)/RMS(|V|)]",
            "normal_velocity_rmse_m_s": rmse,
            "near_surface_rms_speed_m_s": rms_speed,
            "sample_count": int(ok.sum())}


def evaluate_coherence(path, *, max_space_lag=8, max_time_lag=8, dt=1.0):
    wind = load_wind_npz(path)
    space = _space(wind["speed"], wind["mask"], wind["dx"], max_space_lag)
    time = _time(wind["speed"], wind["mask"], dt, max_time_lag)
    terrain = _terrain(wind["u"], wind["v"], wind["w"], wind["mask"],
                       wind["dem"], wind["dx"], wind["dy"])
    total, effective = weighted_available(
        {"space": space["score"], "time": time["score"],
         "terrain": terrain["score"]}, TOP_WEIGHTS)
    return {"metric": "spatiotemporal-terrain coherence",
            "standard_status": "project-defined diagnostic; not an industry standard",
            "source": str(wind["path"]), "shape_tzyx": list(wind["speed"].shape),
            "nominal_weights": TOP_WEIGHTS, "effective_weights": effective,
            "components": {"space": space, "time": time, "terrain": terrain},
            "coherence_0_1": total,
            "coherence_0_100": None if total is None else 100.0 * total}

