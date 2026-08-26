from __future__ import annotations

import numpy as np

from .common import EPS, clip01, masked_blur2d, weighted_available
from .io import load_wind_npz
from .variogram import spatial_semivariogram, variogram_scores

WSCI_WEIGHTS = {"Iv": 0.35, "Is": 0.35, "Id": 0.30}
IV_WEIGHTS = {"R2_profile": 0.60, "P_alpha": 0.25, "P_smooth": 0.15}
IS_WEIGHTS = {"P_mono": 0.55, "P_nugget": 0.35, "P_var": 0.10}


def _vertical(speed, mask, z, min_coverage, alpha_min, alpha_max, alpha_k):
    if speed.shape[1] < 3:
        return {"score": None, "reason": "fewer than 3 vertical levels"}
    counts = mask.sum(axis=(0, 2, 3))
    possible = mask.shape[0] * mask.shape[2] * mask.shape[3]
    coverage = counts / possible
    profile = np.asarray([
        speed[:, iz][mask[:, iz]].mean() if counts[iz] else np.nan
        for iz in range(speed.shape[1])])
    good = np.isfinite(profile) & (profile > EPS) & (z > 0) & (coverage >= min_coverage)
    if good.sum() < 3:
        return {"score": None, "reason": "fewer than 3 valid profile levels",
                "profile": profile.tolist(), "coverage": coverage.tolist()}
    alpha, loga = np.polyfit(np.log(z[good]), np.log(profile[good]), 1)
    fit = np.exp(loga) * z[good] ** alpha
    denom = np.sum((profile[good] - profile[good].mean()) ** 2)
    r2_raw = 1.0 - np.sum((profile[good] - fit) ** 2) / (denom + EPS)
    r2 = clip01(r2_raw)
    if alpha_min <= alpha <= alpha_max:
        p_alpha = 1.0
    else:
        distance = alpha_min - alpha if alpha < alpha_min else alpha - alpha_max
        p_alpha = clip01(np.exp(-alpha_k * distance))
    p_smooth = clip01(np.exp(-np.mean(np.abs(np.diff(profile[good], n=2))) /
                             (np.std(profile[good]) + EPS)))
    score = clip01(sum(IV_WEIGHTS[k] * v for k, v in
                       {"R2_profile": r2, "P_alpha": p_alpha,
                        "P_smooth": p_smooth}.items()))
    return {"score": score, "R2_profile": r2, "R2_raw": float(r2_raw),
            "alpha": float(alpha), "P_alpha": p_alpha, "P_smooth": p_smooth,
            "z": z.tolist(), "profile": profile.tolist(),
            "coverage": coverage.tolist(), "used_levels": good.tolist()}


def _spatial(speed, mask, truth, max_lag, dx, missing_policy):
    gamma, pairs = spatial_semivariogram(speed, mask, max_lag)
    sub = variogram_scores(gamma)
    p_mono, p_nugget = sub["P_mono"], sub["P_small"]
    if truth is not None:
        ok = mask & np.isfinite(truth)
        vp, vt = float(np.var(speed[ok])), float(np.var(truth[ok]))
        p_var = clip01(min(vp / (vt + EPS), vt / (vp + EPS)))
        p_var_status = "computed from reference field"
    elif missing_policy == "legacy-perfect":
        p_var = 1.0
        p_var_status = "forced to 1 by legacy-perfect policy"
    else:
        p_var = None
        p_var_status = "unavailable and excluded"
    score, effective = weighted_available(
        {"P_mono": p_mono, "P_nugget": p_nugget, "P_var": p_var}, IS_WEIGHTS)
    return {"score": score, "effective_weights": effective,
            "P_mono": p_mono, "P_nugget": p_nugget, "P_var": p_var,
            "P_var_status": p_var_status, "corr_lag_gamma": sub["corr_lag_gamma"],
            "lag_m": (np.arange(1, max_lag + 1) * dx).tolist(),
            "gamma_m2_s2": gamma.tolist(), "pair_count": pairs.tolist()}


def _disturbance(speed, mask, sigma):
    residual, coherent, raw = [], [], []
    for t in range(speed.shape[0]):
        for z in range(speed.shape[1]):
            m, a = mask[t, z], speed[t, z]
            if not m.any():
                continue
            bg = masked_blur2d(a, m, sigma)
            up = a - bg
            coh = masked_blur2d(up, m, sigma)
            residual.append(up[m]); coherent.append(coh[m]); raw.append(a[m])
    if not residual:
        return {"score": None, "reason": "no valid cells"}
    up, coh, values = np.concatenate(residual), np.concatenate(coherent), np.concatenate(raw)
    p_rd = clip01(float(np.std(up)) / (abs(float(np.mean(values))) + EPS))
    c_d = clip01(float(np.var(coh)) / (float(np.var(up)) + EPS))
    return {"score": clip01(p_rd * c_d), "P_Rd": p_rd, "C_d": c_d,
            "mean_speed_m_s": float(np.mean(values)),
            "perturbation_std_m_s": float(np.std(up))}


def evaluate_wsci(path, *, max_lag=8, gaussian_sigma=2.0,
                  alpha_min=0.05, alpha_max=0.50, alpha_k=5.0,
                  min_vertical_coverage=0.50, missing_policy="reweight"):
    if missing_policy not in {"reweight", "legacy-perfect"}:
        raise ValueError("missing_policy must be reweight or legacy-perfect")
    wind = load_wind_npz(path)
    iv = _vertical(wind["speed"], wind["mask"], wind["z"],
                   min_vertical_coverage, alpha_min, alpha_max, alpha_k)
    iss = _spatial(wind["speed"], wind["mask"], wind["true_speed"],
                   max_lag, wind["dx"], missing_policy)
    ids = _disturbance(wind["speed"], wind["mask"], gaussian_sigma)
    total, effective = weighted_available(
        {"Iv": iv["score"], "Is": iss["score"], "Id": ids["score"]},
        WSCI_WEIGHTS)
    return {"metric": "WSCI", "standard_status":
            "project-defined composite score; not an industry standard",
            "source": str(wind["path"]), "shape_tzyx": list(wind["speed"].shape),
            "field": "horizontal speed sqrt(u^2+v^2), or abs(u) when v is absent",
            "missing_policy": missing_policy, "nominal_weights": WSCI_WEIGHTS,
            "effective_weights": effective,
            "components": {"Iv": iv, "Is": iss, "Id": ids},
            "total_0_1": total, "WSCI_0_100": None if total is None else 100.0 * total}

