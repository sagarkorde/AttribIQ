"""Graph-based seeded-expansion baseline (added for the revision).

Random walk with restart (personalised PageRank) on the bipartite graph that links
each eligible host to the destination addresses it contacts. Walking from a host to
a destination and back spreads mass to the hosts that share that destination, in
inverse proportion to how many hosts do, so the walk is a label-free, seed-driven
form of guilt by association.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def bipartite(flows: pd.DataFrame, hosts: list[str], weight: str = "count") -> sparse.csr_matrix:
    f = flows[flows["src"].astype(str).isin(hosts)]
    g = f.groupby([f["src"].astype(str), f["dst"].astype(str)]).size()
    hidx = pd.Index(hosts)
    dsts = pd.Index(g.index.get_level_values(1).unique())
    rows = hidx.get_indexer(g.index.get_level_values(0))
    cols = dsts.get_indexer(g.index.get_level_values(1))
    w = g.to_numpy(dtype=float) if weight == "count" else np.ones(len(g))
    if weight == "log":
        w = np.log1p(g.to_numpy(dtype=float))
    return sparse.csr_matrix((w, (rows, cols)), shape=(len(hosts), len(dsts)))


def rwr(B: sparse.csr_matrix, seeds: list[int], restart: float = 0.15,
        iters: int = 200, tol: float = 1e-12) -> np.ndarray:
    """Stationary visiting probabilities of host nodes for each seed (hosts x seeds)."""
    nh, nd = B.shape
    A = sparse.bmat([[None, B], [B.T, None]], format="csr")
    deg = np.asarray(A.sum(axis=0)).ravel()
    deg[deg == 0] = 1.0
    P = A @ sparse.diags(1.0 / deg)
    E = np.zeros((nh + nd, len(seeds)))
    E[seeds, np.arange(len(seeds))] = 1.0
    X = E.copy()
    for _ in range(iters):
        Xn = (1 - restart) * (P @ X) + restart * E
        if np.abs(Xn - X).max() < tol:
            X = Xn
            break
        X = Xn
    return X[:nh]


def rare_destination_candidates(B: sparse.csr_matrix, k: int, cap: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Candidate pruning: score host pairs by the summed rarity weight of shared
    destinations contacted by at most ``cap`` hosts and keep the top ``k`` per host.

    Returns (candidate indices, scores), both hosts x k; -1 marks missing candidates.
    """
    Bb = (B > 0).astype(float).tocsc()
    df = np.asarray(Bb.sum(axis=0)).ravel()
    n = B.shape[0]
    keep = np.flatnonzero((df >= 2) & (df <= cap))
    idf = np.log((1 + n) / (1 + df[keep])) + 1
    R = Bb[:, keep].tocsr() @ sparse.diags(idf)
    M = (R @ Bb[:, keep].T).tocsr()          # sparse: only pairs sharing a rare destination
    M.setdiag(0)
    M.eliminate_zeros()
    cand = -np.ones((n, k), dtype=int)
    score = np.zeros((n, k))
    for i in range(n):
        lo, hi = M.indptr[i], M.indptr[i + 1]
        if hi == lo:
            continue
        idx, val = M.indices[lo:hi], M.data[lo:hi]
        top = np.argsort(-val, kind="stable")[:k]
        cand[i, :len(top)] = idx[top]
        score[i, :len(top)] = val[top]
    return cand, score
