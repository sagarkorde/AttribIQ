"""Combining evidence lines into a calibrated, decomposable attribution score.

``EvidenceCombiner`` fits one logistic model per *availability pattern* of the
evidence (reduced models): when a line is unavailable for a pair (for example
temporal co-activity between hosts observed in different incidents), the pair
is scored by the model trained without that line instead of imputing zero.

For a linear-logistic model the log-odds decompose exactly into additive
per-evidence contributions

    logit p = base + sum_i w_i (x_i - mu_i),     base = b + sum_i w_i mu_i,

where mu_i is the training mean of evidence i. The contribution
phi_i = w_i (x_i - mu_i) states how much evidence i moved this pair away from an
average training pair, in log-odds units; the contributions sum to
logit p - base.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .evidence import EVIDENCE

V1_WEIGHTS = {"E_beh": 0.35, "E_inf": 0.25, "E_tmp": 0.20, "E_ti": 0.20}


@dataclass
class _Reduced:
    features: list[str]
    model: LogisticRegression
    mu: np.ndarray


class EvidenceCombiner:
    def __init__(self, features: list[str] | None = None, C: float = 1.0):
        self.features = list(EVIDENCE if features is None else features)
        self.C = C
        self.models: dict[tuple[str, ...], _Reduced] = {}

    def _patterns(self) -> list[tuple[str, ...]]:
        pats = [tuple(self.features)]
        if "E_tmp" in self.features:
            pats.append(tuple(f for f in self.features if f != "E_tmp"))
        return pats

    def fit(self, pairs: pd.DataFrame, y: np.ndarray) -> "EvidenceCombiner":
        y = np.asarray(y).astype(int)
        for pat in self._patterns():
            X = pairs[list(pat)].to_numpy(dtype=float)
            ok = ~np.isnan(X).any(axis=1)
            m = LogisticRegression(C=self.C, max_iter=2000)
            m.fit(X[ok], y[ok])
            self.models[pat] = _Reduced(list(pat), m, X[ok].mean(axis=0))
        return self

    def _pattern_of(self, pairs: pd.DataFrame) -> np.ndarray:
        if "E_tmp" in self.features:
            return np.where(pairs["E_tmp"].isna().to_numpy(), 1, 0)
        return np.zeros(len(pairs), dtype=int)

    def logit(self, pairs: pd.DataFrame) -> np.ndarray:
        pats = self._patterns()
        which = self._pattern_of(pairs)
        out = np.full(len(pairs), np.nan)
        for k, pat in enumerate(pats):
            sel = which == k
            if not sel.any():
                continue
            r = self.models[pat]
            X = pairs.loc[sel, r.features].to_numpy(dtype=float)
            out[sel] = r.model.decision_function(X)
        return out

    def predict_proba(self, pairs: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(pairs)))

    def decompose(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Per-pair base log-odds and contribution of every evidence line."""
        pats = self._patterns()
        which = self._pattern_of(pairs)
        out = pd.DataFrame(0.0, index=pairs.index, columns=["base"] + self.features)
        avail = pd.DataFrame(False, index=pairs.index, columns=self.features)
        for k, pat in enumerate(pats):
            sel = which == k
            if not sel.any():
                continue
            r = self.models[pat]
            X = pairs.loc[sel, r.features].to_numpy(dtype=float)
            w = r.model.coef_[0]
            out.loc[sel, "base"] = r.model.intercept_[0] + float(w @ r.mu)
            out.loc[sel, r.features] = (X - r.mu) * w
            avail.loc[sel, r.features] = True
        out = out.join(avail.add_prefix("avail_"))
        return out

    def coefficients(self) -> pd.DataFrame:
        rows = []
        for pat, r in self.models.items():
            row = {"pattern": "+".join(pat), "intercept": r.model.intercept_[0]}
            row.update({f: w for f, w in zip(r.features, r.model.coef_[0])})
            rows.append(row)
        return pd.DataFrame(rows)


def v1_score(pairs: pd.DataFrame) -> np.ndarray:
    """The original AttribIQ weighting applied to the corrected evidence lines.

    Behaviour is the mean of the two behavioural lines, infrastructure is the
    bounded IDF-weighted cosine overlap (``inf_cos``); unavailable evidence
    (temporal across incidents, threat intelligence everywhere) is scored as
    zero, as in the original paper.
    """
    beh = pairs[["E_flow", "E_stat"]].mean(axis=1).to_numpy()
    tmp = np.nan_to_num(pairs["E_tmp"].to_numpy(), nan=0.0).clip(0, 1)
    # the original infrastructure term was a bounded overlap fraction; use the
    # bounded IDF-weighted cosine overlap when it is available
    inf = pairs["inf_cos"].to_numpy() if "inf_cos" in pairs else pairs["E_inf"].clip(0, 1).to_numpy()
    return (V1_WEIGHTS["E_beh"] * beh + V1_WEIGHTS["E_inf"] * inf
            + V1_WEIGHTS["E_tmp"] * tmp + V1_WEIGHTS["E_ti"] * 0.0)


def mean_available(pairs: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    """Unweighted mean of the available evidence lines (missing lines skipped)."""
    feats = EVIDENCE if features is None else features
    X = pairs[feats].to_numpy(dtype=float, copy=True)
    if "E_tmp" in feats:
        j = feats.index("E_tmp")
        X[:, j] = np.clip(X[:, j], 0, 1)
    return np.nanmean(X, axis=1)
