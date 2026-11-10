"""Pairwise evidence that two hosts belong to the same campaign.

Four evidence lines are computed for a pair of hosts (a, b):

* ``E_flow`` - Bhattacharyya coefficient between the hosts' flow-type
  distributions (1 = identical traffic mix);
* ``E_stat`` - similarity of robust-standardised statistical profiles,
  exp(-mean squared difference / 2);
* ``E_inf``  - shared rare infrastructure: log(1 + sum of inverse host
  frequency over the destinations both hosts contacted). Destinations that
  many hosts use (resolvers, update servers) contribute almost nothing; the
  score is not normalised by the hosts' total number of destinations, so
  unrelated benign traffic on an infected host does not dilute it;
* ``E_tmp``  - temporal co-activity: Pearson correlation of log(1 + per-minute
  flow counts). Only defined when both hosts were observed over the same time
  span (the same incident); otherwise it is *unavailable* (NaN), not zero.

All reference statistics (robust scale of the profile statistics, inverse host
frequencies) are fitted without labels on a reference host population.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .profiles import STAT_NAMES, HostProfiles

EVIDENCE = ["E_flow", "E_stat", "E_inf", "E_tmp"]
INFRA_MODES = ("dot", "dot2", "cos")


class EvidenceSpace:
    """Label-free reference statistics shared by all pairwise computations.

    Parameters
    ----------
    profiles: reference population (one or several incidents).
    infra_mode: ``dot`` (default, sum of IDF over shared destinations),
        ``dot2`` (sum of squared IDF) or ``cos`` (IDF-weighted cosine).
    external_only: ignore destinations inside the monitored prefix.
    use_idf: if False all destinations get weight 1.
    """

    def __init__(self, profiles: list[HostProfiles], infra_mode: str = "dot",
                 external_only: bool = False, use_idf: bool = True):
        if infra_mode not in INFRA_MODES:
            raise ValueError(infra_mode)
        self.infra_mode = infra_mode
        self.external_only = external_only
        self.use_idf = use_idf
        stats = pd.concat([p.stats for p in profiles], axis=0)
        self.median = stats.median(skipna=True)
        q75, q25 = stats.quantile(0.75), stats.quantile(0.25)
        iqr = q75 - q25
        scale = iqr.where(iqr > 1e-9, stats.std(skipna=True))
        self.scale = scale.where(scale > 1e-9, 1.0).fillna(1.0)
        self.tokens = sorted(set().union(*[set(p.tokens.columns) for p in profiles]))
        df: Counter = Counter()
        n_hosts = 0
        for p in profiles:
            for h in p.hosts:
                df.update(self._filter(p.dsts[h]))
                n_hosts += 1
        self.n_hosts = n_hosts
        self.idf = {d: math.log((1 + n_hosts) / (1 + c)) + 1.0 for d, c in df.items()}

    def _filter(self, dsts: frozenset) -> frozenset:
        if not self.external_only:
            return dsts
        return frozenset(d for d in dsts if not d.startswith("147.32."))

    # -- representations ---------------------------------------------------
    def token_matrix(self, p: HostProfiles, idx: list[int]) -> np.ndarray:
        return p.tokens.reindex(columns=self.tokens, fill_value=0.0).to_numpy()[idx]

    def z_stats(self, p: HostProfiles, idx: list[int]) -> np.ndarray:
        s = p.stats[STAT_NAMES].fillna(self.median[STAT_NAMES])
        z = ((s - self.median[STAT_NAMES]) / self.scale[STAT_NAMES]).clip(-6, 6)
        return z.to_numpy()[idx]

    def _incidence(self, sets: list[frozenset], vocab: dict[str, int]) -> sp.csr_matrix:
        rows, cols = [], []
        for i, s in enumerate(sets):
            for d in s:
                j = vocab.get(d)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sets), len(vocab)))

    def infra_matrix(self, pa: HostProfiles, rows: list[str], pb: HostProfiles,
                     cols: list[str]) -> np.ndarray:
        Sa = [self._filter(pa.dsts[h]) for h in rows]
        Sb = [self._filter(pb.dsts[h]) for h in cols]
        vocab_list = sorted(set().union(*Sa) & set().union(*Sb)) if Sa and Sb else []
        vocab = {d: i for i, d in enumerate(vocab_list)}
        if not vocab:
            return np.zeros((len(rows), len(cols)))
        w = np.array([self.idf.get(d, math.log(1 + self.n_hosts) + 1.0) if self.use_idf else 1.0
                      for d in vocab_list])
        A, B = self._incidence(Sa, vocab), self._incidence(Sb, vocab)
        if self.infra_mode == "dot":
            return np.log1p((A @ sp.diags(w) @ B.T).toarray())
        if self.infra_mode == "dot2":
            return np.log1p((A @ sp.diags(w * w) @ B.T).toarray())
        # cosine over the full (not intersected) weighted destination vectors
        def norms(sets):
            return np.array([math.sqrt(sum((self.idf.get(d, 1.0) if self.use_idf else 1.0) ** 2
                                           for d in s)) for s in sets])
        na, nb = norms(Sa), norms(Sb)
        dot = (A @ sp.diags(w * w) @ B.T).toarray()
        den = na[:, None] * nb[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(den > 0, dot / den, 0.0)
        return out

    # -- evidence ----------------------------------------------------------
    @staticmethod
    def flow_affinity(Pa: np.ndarray, Pb: np.ndarray) -> np.ndarray:
        return np.sqrt(Pa) @ np.sqrt(Pb).T

    @staticmethod
    def stat_similarity(Za: np.ndarray, Zb: np.ndarray) -> np.ndarray:
        sq_a = (Za ** 2).sum(1)[:, None]
        sq_b = (Zb ** 2).sum(1)[None, :]
        d2 = np.maximum(sq_a + sq_b - 2 * Za @ Zb.T, 0.0) / Za.shape[1]
        return np.exp(-0.5 * d2)

    @staticmethod
    def temporal(Sa: np.ndarray, Sb: np.ndarray, min_active: int = 3) -> np.ndarray:
        """Pearson correlation of log1p per-minute counts; NaN when undefined."""
        A, B = np.log1p(Sa), np.log1p(Sb)
        act_a, act_b = (Sa > 0).sum(1), (Sb > 0).sum(1)
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
        sa, sb = np.sqrt((A ** 2).sum(1)), np.sqrt((B ** 2).sum(1))
        num = A @ B.T
        den = sa[:, None] * sb[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / den
        bad = (act_a[:, None] < min_active) | (act_b[None, :] < min_active) | (den == 0)
        r[bad] = np.nan
        return r

    def matrices(self, pa: HostProfiles, rows: list[str], pb: HostProfiles,
                 cols: list[str], use_temporal: bool = True) -> dict[str, np.ndarray]:
        ia = [pa.hosts.index(h) for h in rows]
        ib = [pb.hosts.index(h) for h in cols]
        out = {
            "E_flow": self.flow_affinity(self.token_matrix(pa, ia), self.token_matrix(pb, ib)),
            "E_stat": self.stat_similarity(self.z_stats(pa, ia), self.z_stats(pb, ib)),
            "E_inf": self.infra_matrix(pa, rows, pb, cols),
        }
        same = pa.sid == pb.sid and pa.t0 == pb.t0 and pa.series.shape[1] == pb.series.shape[1]
        if same and use_temporal:
            out["E_tmp"] = self.temporal(pa.series[ia], pb.series[ib])
        else:
            out["E_tmp"] = np.full(out["E_flow"].shape, np.nan)
        return out

    def pair_table(self, pa: HostProfiles, rows: list[str], pb: HostProfiles,
                   cols: list[str] | None = None, use_temporal: bool = True) -> pd.DataFrame:
        """Long table of evidence for every (row host, column host) pair."""
        cols = pb.hosts if cols is None else cols
        M = self.matrices(pa, rows, pb, cols, use_temporal)
        rr, cc = np.meshgrid(np.arange(len(rows)), np.arange(len(cols)), indexing="ij")
        df = pd.DataFrame({
            "a": np.asarray(rows, dtype=object)[rr.ravel()],
            "b": np.asarray(cols, dtype=object)[cc.ravel()],
            "sid_a": pa.sid, "sid_b": pb.sid,
            **{k: v.ravel() for k, v in M.items()},
        })
        if pa.sid == pb.sid:
            df = df[df["a"] != df["b"]]
        return df.reset_index(drop=True)
