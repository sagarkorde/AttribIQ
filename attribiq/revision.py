"""Experiments added in the revision.

The method modules (``profiles``, ``evidence``, ``fusion``, ``combine``) are unchanged
since the design freeze at commit 98637ce; this module only adds evaluation
conditions, a graph baseline, candidate pruning and the held-out incident.
``git diff 98637ce -- attribiq/profiles.py attribiq/evidence.py attribiq/fusion.py
attribiq/combine.py`` is empty.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evidence import EVIDENCE, EvidenceSpace
from .experiments import FAMILY, Config, _fit_logreg, _rng, rq1_pairs, rq2_decide, rq2_pairs
from .fusion import Calibrator, ReferenceCDF, fused_score
from .graph import bipartite, rare_destination_candidates, rwr
from .profiles import build_profiles
from .stress import blend, blend_session

RWR_SETTINGS = {f"rwr_{w}_{int(r * 100):02d}": (w, r)
                for w in ("count", "binary") for r in (0.1, 0.3, 0.5)}


def _profiles(cfg: Config, flows: dict, roles_in: dict, scen: tuple) -> tuple[dict, dict]:
    P, roles = {}, {}
    for s in scen:
        P[s] = build_profiles(flows[s], s, min_flows=cfg.min_flows, bin_seconds=cfg.bin_seconds)
        roles[s] = roles_in[s].reindex(P[s].hosts).fillna("background")
    return P, roles


def blended(cfg: Config, F: dict, R: dict, m: float, rep: int, scen: tuple,
            session: bool = False) -> tuple[dict, dict]:
    """Blend each scenario with the same random draws as the main experiment."""
    flows, roles = {}, {}
    for s in scen:
        fn = blend_session if session else blend
        f, r, _ = fn(F[s], R[s], m, _rng(cfg, m, rep, s))
        flows[s], roles[s] = f, r
    return flows, roles


def frozen_scores(new: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Calibrated probability and learned baseline for new pairs, fitted exactly as in
    the main experiment (leave-one-family-out on the main seed pairs, all levels)."""
    new = new.copy()
    new["p_attribiq"] = np.nan
    new["p_logreg"] = np.nan
    for s in sorted(new["sid"].unique()):
        test = new["sid"] == s
        train = main[main["sid"].map(FAMILY) != FAMILY[s]]
        cal = Calibrator().fit(train["S"].to_numpy(), train["y"].to_numpy())
        new.loc[test, "p_attribiq"] = cal.predict(new.loc[test, "S"].to_numpy())
        tr = train.dropna(subset=EVIDENCE)
        sc4, lr4 = _fit_logreg(tr, EVIDENCE)
        sc3, lr3 = _fit_logreg(tr, EVIDENCE[:3])
        te = new[test]
        has_t = te["E_tmp"].notna().to_numpy()
        p = np.empty(len(te))
        if has_t.any():
            p[has_t] = lr4.predict_proba(sc4.transform(te.loc[has_t, EVIDENCE]))[:, 1]
        if (~has_t).any():
            p[~has_t] = lr3.predict_proba(sc3.transform(te.loc[~has_t, EVIDENCE[:3]]))[:, 1]
        new.loc[test, "p_logreg"] = p
    return new


# ---------------------------------------------------------------------------
def graph_and_pruning(cfg: Config, F: dict, R: dict, m: float, rep: int,
                      k: int = 50, cap: int = 50) -> pd.DataFrame:
    """Random-walk-with-restart scores and rare-destination pruning ranks for every
    seed pair of the main experiment (same blending draws)."""
    flows, roles0 = blended(cfg, F, R, m, rep, cfg.multi)
    P, roles = _profiles(cfg, flows, roles0, cfg.multi)
    out = []
    for s in cfg.multi:
        p, r = P[s], roles[s]
        hosts = list(p.hosts)
        bots = [h for h in hosts if r[h] == "bot"]
        if len(bots) < 2:
            continue
        ib = [hosts.index(b) for b in bots]
        rows = {"a": [], "b": []}
        for a in bots:
            for b in hosts:
                if b != a:
                    rows["a"].append(a)
                    rows["b"].append(b)
        df = pd.DataFrame(rows)
        ia = df["a"].map({h: i for i, h in enumerate(bots)}).to_numpy()
        jb = df["b"].map({h: i for i, h in enumerate(hosts)}).to_numpy()
        for name, (w, rs) in RWR_SETTINGS.items():
            B = bipartite(flows[s], hosts, weight=w)
            X = rwr(B, ib, restart=rs)
            df[name] = X[jb, ia]
        B = bipartite(flows[s], hosts, weight="count")
        cand, _ = rare_destination_candidates(B, k=k, cap=cap)
        rank = np.full(len(df), np.nan)
        pos = {(i, c): q for i in ib for q, c in enumerate(cand[i]) if c >= 0}
        for t, (a, b) in enumerate(zip(df["a"], df["b"])):
            q = pos.get((hosts.index(a), hosts.index(b)))
            if q is not None:
                rank[t] = q + 1
        df["prune_rank"] = rank
        df["m"], df["rep"], df["sid"] = m, rep, s
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def session_pairs(cfg: Config, F: dict, R: dict, m: float, rep: int) -> pd.DataFrame:
    flows, roles0 = blended(cfg, F, R, m, rep, cfg.multi, session=True)
    P, roles = _profiles(cfg, flows, roles0, cfg.multi)
    return rq1_pairs(cfg, P, roles, flows, m, rep, with_variants=False, with_v1=False)


def contamination_pairs(cfg: Config, F: dict, R: dict, m: float, rep: int, keep: float) -> pd.DataFrame:
    """Keep only a fraction of the non-infected eligible hosts, so infected hosts make
    up a larger share of the reference population used for the percentiles."""
    flows, roles0 = blended(cfg, F, R, m, rep, cfg.multi)
    for s in cfg.multi:
        f, r = flows[s], roles0[s]
        cnt = f.groupby(f["src"].astype(str)).size()
        cand = sorted(h for h in cnt.index[cnt >= cfg.min_flows] if r.get(h, "background") != "bot")
        rng = np.random.default_rng([cfg.seed, 777, int(round(keep * 100)), rep, s])
        drop = set(rng.choice(cand, size=len(cand) - int(round(keep * len(cand))), replace=False))
        flows[s] = f[~f["src"].astype(str).isin(drop)].reset_index(drop=True)
    P, roles = _profiles(cfg, flows, roles0, cfg.multi)
    pairs = rq1_pairs(cfg, P, roles, flows, m, rep, with_variants=False, with_v1=False)
    if len(pairs):
        pairs["keep"] = keep
        pairs["n_hosts"] = pairs["sid"].map({s: len(P[s].hosts) for s in cfg.multi})
        pairs["n_infected"] = pairs["sid"].map({s: int((roles[s] == "bot").sum()) for s in cfg.multi})
    return pairs


def heldout_t2(cfg: Config, F: dict, R: dict, m: float, rep: int, rq1_main: pd.DataFrame) -> pd.DataFrame:
    flows, roles0 = blended(cfg, F, R, m, rep, cfg.scenarios)
    P, roles = _profiles(cfg, flows, roles0, cfg.scenarios)
    pairs = rq2_pairs(cfg, P, roles, m, rep)
    dec = rq2_decide(cfg, pairs, rq1_main)
    dec["n_lib"] = dec["sid"].map(lambda s: sum(int((roles[t] == "bot").sum()) for t in cfg.scenarios if t != s))
    return dec


# ---------------------------------------------------------------------------
def scaling(cfg: Config, flows_all: pd.DataFrame, sizes: list[int], k: int = 50, cap: int = 50,
            full_max: int = 2000, ref_block: int = 300) -> list[dict]:
    """Wall-clock cost of exhaustive versus pruned scoring for growing host counts."""
    cnt = flows_all.groupby(flows_all["src"].astype(str)).size()
    eligible = sorted(cnt.index[cnt >= cfg.min_flows])
    out = []
    sizes = [n for n in sizes if n < len(eligible)] + [len(eligible)]
    for n in sizes:
        rng = np.random.default_rng([cfg.seed, 999, n])
        hosts = sorted(rng.choice(eligible, size=n, replace=False))
        f = flows_all[flows_all["src"].astype(str).isin(hosts)]
        t = time.perf_counter()
        p = build_profiles(f, 0, min_flows=cfg.min_flows, bin_seconds=cfg.bin_seconds)
        t_prof = time.perf_counter() - t
        ES = EvidenceSpace([p], infra_mode=cfg.infra_mode)
        row = {"n_hosts": len(p.hosts), "profiles_s": t_prof}
        if n <= full_max:
            t = time.perf_counter()
            ref = ES.pair_table(p, p.hosts, p)
            ReferenceCDF().fit(ref).transform(ref)
            row["exhaustive_s"] = time.perf_counter() - t
            row["exhaustive_pairs"] = len(ref)
            del ref
        t = time.perf_counter()
        B = bipartite(f, list(p.hosts), weight="count")
        cand, _ = rare_destination_candidates(B, k=k, cap=cap)
        t_cand = time.perf_counter() - t
        t = time.perf_counter()
        hs = list(p.hosts)
        rb = list(rng.choice(hs, size=min(ref_block, len(hs)), replace=False))
        cb = list(rng.choice(hs, size=min(ref_block, len(hs)), replace=False))
        cdf = ReferenceCDF().fit(ES.pair_table(p, rb, p, cb))
        n_pairs = 0
        for i, h in enumerate(hs):
            c = [hs[j] for j in cand[i] if j >= 0]
            if not c:
                continue
            pt = ES.pair_table(p, [h], p, c)
            fused_score(cdf.transform(pt))
            n_pairs += len(pt)
        row.update({"candidates_s": t_cand, "pruned_scoring_s": time.perf_counter() - t,
                    "pruned_pairs": n_pairs})
        out.append(row)
    return out
