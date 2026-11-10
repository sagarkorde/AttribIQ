"""Label-free evidence fusion with calibration and exact decomposition.

Each evidence line is mapped through the empirical CDF of that line over a
label-free reference set of host pairs (all pairs of the incident, or all
query-library pairs across incidents). A value of 0.99 means "this pair shares
more of this evidence than 99% of host pairs". The fused score is the mean of
the available percentiles:

    S(a, b) = (1 / k) * sum_{i available} F_i(E_i(a, b)).

A two-parameter logistic calibrator, fitted on training families only, maps S
to a probability p = sigmoid(alpha * S + beta). Because S is an equal-weight
sum, the log-odds decompose exactly:

    logit p = (alpha / 2 + beta) + sum_i (alpha / k) * (F_i - 1/2),

so phi_i = (alpha / k)(F_i - 1/2) is the contribution of evidence line i
relative to a pair at the median of every line.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .evidence import EVIDENCE


class ReferenceCDF:
    def __init__(self, features: list[str] | None = None):
        self.features = list(EVIDENCE if features is None else features)
        self.sorted: dict[str, np.ndarray] = {}

    def fit(self, ref: pd.DataFrame) -> "ReferenceCDF":
        for f in self.features:
            v = ref[f].to_numpy(dtype=float)
            self.sorted[f] = np.sort(v[~np.isnan(v)])
        return self

    def transform(self, pairs: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=pairs.index)
        for f in self.features:
            ref = self.sorted[f]
            x = pairs[f].to_numpy(dtype=float)
            if len(ref) == 0:
                out["F_" + f] = np.nan
                continue
            lo = np.searchsorted(ref, x, side="left")
            hi = np.searchsorted(ref, x, side="right")
            pct = (lo + hi) / (2.0 * len(ref))          # mid-rank percentile
            pct[np.isnan(x)] = np.nan
            out["F_" + f] = pct
        return out


def fused_score(pct: pd.DataFrame) -> np.ndarray:
    return np.nanmean(pct.to_numpy(dtype=float), axis=1)


class Calibrator:
    """Platt scaling of the fused score; fitted on training families only."""

    def __init__(self, C: float = 1e4):
        self.model = LogisticRegression(C=C, max_iter=2000)

    def fit(self, s: np.ndarray, y: np.ndarray) -> "Calibrator":
        self.model.fit(np.asarray(s).reshape(-1, 1), np.asarray(y).astype(int))
        self.alpha = float(self.model.coef_[0, 0])
        self.beta = float(self.model.intercept_[0])
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        z = self.alpha * np.asarray(s) + self.beta
        return 1.0 / (1.0 + np.exp(-z))

    def decompose(self, pct: pd.DataFrame) -> pd.DataFrame:
        P = pct.to_numpy(dtype=float)
        k = (~np.isnan(P)).sum(axis=1).astype(float)
        contrib = (self.alpha / k)[:, None] * (P - 0.5)
        out = pd.DataFrame(contrib, index=pct.index,
                           columns=[c.replace("F_", "phi_") for c in pct.columns])
        out.insert(0, "base", self.alpha / 2 + self.beta)
        return out
