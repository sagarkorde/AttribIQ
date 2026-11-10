"""Split-conformal false-attribution threshold (added in the second revision).

Given best-family scores of ``n`` confirmed non-infected hosts from the incident
itself, the threshold is the ceil((n + 1)(1 - eps))-th smallest score; a new
non-infected host exchangeable with them exceeds it with probability at most eps
(Angelopoulos and Bates, 2023). With fewer than 1/eps - 1 labelled hosts no finite
threshold exists and nothing is attributed.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_threshold(benign_scores, eps: float) -> float:
    s = np.sort(np.asarray(benign_scores, dtype=float))
    n = len(s)
    r = math.ceil((n + 1) * (1 - eps))
    return float("inf") if r > n else float(s[r - 1])


def min_labelled(eps: float) -> int:
    """Smallest calibration size for which a finite threshold exists."""
    return math.ceil(1 / eps) - 1
