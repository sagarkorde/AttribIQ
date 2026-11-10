"""Ranking, calibration and resampling utilities."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ap(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    return float(roc_auc_score(y, s))


def r_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y)
    r = int(y.sum())
    if r == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s), kind="mergesort")
    return float(y[order[:r]].mean())


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(s), kind="mergesort")
    return float(np.asarray(y)[order[:k]].mean())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append((edges[b], edges[b + 1], int(m.sum()),
                     float(p[m].mean()) if m.any() else np.nan,
                     float(y[m].mean()) if m.any() else np.nan))
    return rows


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05,
                 groups: np.ndarray | None = None, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile CI; resamples groups (clusters) when given."""
    v = np.asarray(values, dtype=float)
    keep = ~np.isnan(v)
    v = v[keep]
    rng = np.random.default_rng(seed)
    if len(v) == 0:
        return (float("nan"),) * 3
    if groups is None:
        bs = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    else:
        g = np.asarray(groups)[keep]
        ug = np.unique(g)
        members = [v[g == u] for u in ug]
        bs = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, len(ug), len(ug))
            bs[i] = np.concatenate([members[k] for k in pick]).mean()
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(v.mean()), float(lo), float(hi)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                     seed: int = 0) -> tuple[float, float, float, float]:
    """Mean difference a-b with 95% CI and two-sided bootstrap p-value."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(p, 1.0))
