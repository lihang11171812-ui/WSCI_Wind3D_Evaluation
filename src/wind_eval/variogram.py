from __future__ import annotations

import numpy as np

from .common import EPS, clip01


def spatial_semivariogram(field, mask, max_lag=8):
    """Pooled omnidirectional horizontal semivariogram over t and z."""
    gamma, pairs = [], []
    for h in range(1, max_lag + 1):
        ss = 0.0
        n = 0
        for axis in (-2, -1):
            a1 = np.take(field, range(h, field.shape[axis]), axis=axis)
            a0 = np.take(field, range(field.shape[axis] - h), axis=axis)
            m1 = np.take(mask, range(h, mask.shape[axis]), axis=axis)
            m0 = np.take(mask, range(mask.shape[axis] - h), axis=axis)
            ok = m1 & m0 & np.isfinite(a1) & np.isfinite(a0)
            d = a1 - a0
            ss += float(np.sum(d[ok] ** 2))
            n += int(ok.sum())
        gamma.append(0.5 * ss / n if n else np.nan)
        pairs.append(n)
    return np.asarray(gamma), np.asarray(pairs)


def temporal_semivariogram(field, mask, max_lag=8):
    gamma, pairs, pattern_corr = [], [], []
    for lag in range(1, min(max_lag, field.shape[0] - 1) + 1):
        a, b = field[lag:], field[:-lag]
        ok = mask[lag:] & mask[:-lag] & np.isfinite(a) & np.isfinite(b)
        d = a - b
        gamma.append(0.5 * float(np.mean(d[ok] ** 2)) if ok.any() else np.nan)
        pairs.append(int(ok.sum()))
        if ok.sum() > 2 and np.std(a[ok]) > EPS and np.std(b[ok]) > EPS:
            pattern_corr.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
        else:
            pattern_corr.append(np.nan)
    return np.asarray(gamma), np.asarray(pairs), np.asarray(pattern_corr)


def variogram_scores(gamma):
    good = np.isfinite(gamma)
    if good.sum() < 3 or np.nanmax(gamma) <= EPS:
        return {"corr_lag_gamma": None, "P_mono": None, "P_small": None}
    h = np.arange(1, len(gamma) + 1, dtype=float)[good]
    g = gamma[good]
    corr = float(np.corrcoef(h, g)[0, 1]) if np.std(g) > EPS else 0.0
    return {
        "corr_lag_gamma": corr,
        "P_mono": clip01((corr + 1.0) / 2.0),
        "P_small": clip01(1.0 - g[0] / (np.max(g) + EPS)),
    }

