"""Stress tests that remove the laboratory artefacts of CTU-13.

In CTU-13 every infected host is a virtual machine that runs only the malware,
so its traffic is pure botnet traffic. Real compromised machines also carry
their user's traffic. ``blend`` emulates this: every infected host is merged
with a distinct benign host of the same capture (time-aligned, the benign host
leaves the candidate pool), and flows are subsampled so that the malicious
share of the merged host's traffic equals ``malicious_fraction``.

``truncate`` keeps only the first ``minutes`` of an incident, measured from the
first flow of the earliest infected host, to study how quickly attribution
becomes possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def blend(flows: pd.DataFrame, roles: pd.Series, malicious_fraction: float,
          rng: np.random.Generator, min_benign_flows: int = 50
          ) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """Return (blended flows, blended roles, {bot: absorbed benign host})."""
    if malicious_fraction >= 1.0:
        return flows, roles, {}
    m = float(malicious_fraction)
    f = flows.copy()
    f["src"] = f["src"].astype(str)
    counts = f.groupby("src").size()
    bots = sorted(roles[roles == "bot"].index.intersection(counts.index))
    benign = [h for h in counts.index if roles.get(h, "background") != "bot"
              and counts[h] >= min_benign_flows]
    if len(benign) < len(bots):
        raise ValueError("not enough benign hosts to blend with")
    partners = rng.choice(benign, size=len(bots), replace=False)
    keep = np.ones(len(f), dtype=bool)
    src = f["src"].to_numpy(dtype=object)
    mapping = {}
    for b, u in zip(bots, partners):
        ib = np.flatnonzero(src == b)
        iu = np.flatnonzero(src == u)
        nb, nu = len(ib), len(iu)
        # target: kb / (kb + ku) = m with kb <= nb, ku <= nu
        kb = int(round(m / (1 - m) * nu))
        ku = nu
        if kb > nb:
            kb = nb
            ku = int(round(nb * (1 - m) / m))
        kb = max(kb, 1)
        drop_b = rng.choice(ib, size=nb - kb, replace=False) if kb < nb else np.array([], dtype=int)
        drop_u = rng.choice(iu, size=nu - ku, replace=False) if ku < nu else np.array([], dtype=int)
        keep[drop_b] = False
        keep[drop_u] = False
        src[iu] = b                      # the benign host's traffic now belongs to the bot
        mapping[b] = u
    f["src"] = src
    f = f[keep].reset_index(drop=True)
    new_roles = roles.drop(index=[u for u in mapping.values() if u in roles.index])
    return f, new_roles, mapping


def truncate(flows: pd.DataFrame, roles: pd.Series, minutes: float) -> pd.DataFrame:
    bots = roles[roles == "bot"].index
    src = flows["src"].astype(str)
    start = flows.loc[src.isin(bots), "ts"].min()
    return flows[(flows["ts"] >= start) & (flows["ts"] < start + 60.0 * minutes)].reset_index(drop=True)
