"""Experiment drivers for RQ1-RQ4 (see README and the paper, Section 5).

Every driver writes long-format CSV/parquet files to ``results/``; the paper's
tables and figures are produced from those files by ``scripts/make_report.py``.

Protocol summary
----------------
* Units: internal hosts with at least ``min_flows`` initiated flows.
* Labels are used only as analyst inputs (the confirmed seed host in RQ1, the
  library of confirmed hosts from *other* incidents in RQ2) and for scoring.
* Anything fitted with labels (calibrator, learned baseline) is fitted on
  scenarios of *other malware families* than the test scenario
  (leave-one-family-out).
* Blending condition m in {1.0, 0.5, 0.2, 0.1, 0.05}; R repetitions with
  independent random partner assignment and subsampling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .baselines import hdbscan_comembership, membership_similarity, v1_cluster, v1_fingerprints
from .combine import v1_score
from .data import host_roles, load_scenario
from .evidence import EVIDENCE, EvidenceSpace
from .fusion import Calibrator, ReferenceCDF, fused_score
from .profiles import build_profiles
from .scenarios import MULTI_HOST_SCENARIOS, PAPER_SCENARIOS, SCENARIOS
from .stress import blend, truncate

FAMILY = {s: SCENARIOS[s].family for s in SCENARIOS}


@dataclass
class Config:
    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    out_dir: str = "results"
    scenarios: tuple = PAPER_SCENARIOS
    multi: tuple = MULTI_HOST_SCENARIOS
    fractions: tuple = (1.0, 0.5, 0.2, 0.1, 0.05)
    reps: int = 5
    min_flows: int = 10
    bin_seconds: int = 60
    infra_mode: str = "dot"
    tau: float = 0.5
    seed: int = 20251109
    timings: list = field(default_factory=list)


def _rng(cfg: Config, m: float, rep: int, sid: int) -> np.random.Generator:
    return np.random.default_rng([cfg.seed, int(round(m * 1000)), rep, sid])


def load_all(cfg: Config) -> tuple[dict, dict]:
    F, R = {}, {}
    for s in cfg.scenarios:
        t = time.perf_counter()
        F[s] = load_scenario(s, cfg.raw_dir, cfg.cache_dir)
        R[s] = host_roles(F[s])
        cfg.timings.append({"stage": "load", "sid": s, "seconds": time.perf_counter() - t,
                            "flows": len(F[s])})
    return F, R


def condition(cfg: Config, F: dict, R: dict, m: float, rep: int,
              min_flows: int | None = None, bin_seconds: int | None = None,
              minutes: float | None = None) -> tuple[dict, dict, dict]:
    """Blend (and optionally truncate) every scenario and build profiles."""
    P, roles, flows = {}, {}, {}
    for s in cfg.scenarios:
        f, r, _ = blend(F[s], R[s], m, _rng(cfg, m, rep, s))
        if minutes is not None:
            f = truncate(f, r, minutes)
        t = time.perf_counter()
        P[s] = build_profiles(f, s, min_flows=min_flows or cfg.min_flows,
                              bin_seconds=bin_seconds or cfg.bin_seconds)
        cfg.timings.append({"stage": "profiles", "sid": s, "m": m, "rep": rep,
                            "seconds": time.perf_counter() - t, "hosts": len(P[s].hosts)})
        roles[s] = r.reindex(P[s].hosts).fillna("background")
        flows[s] = f
    return P, roles, flows


# ---------------------------------------------------------------------------
# RQ1: within-incident campaign expansion from one confirmed seed
# ---------------------------------------------------------------------------
def rq1_pairs(cfg: Config, P: dict, roles: dict, flows: dict, m: float, rep: int,
              with_variants: bool = True, with_v1: bool = True) -> pd.DataFrame:
    out = []
    for s in cfg.multi:
        p, r = P[s], roles[s]
        bots = [h for h in p.hosts if r[h] == "bot"]
        if len(bots) < 2:
            continue
        t = time.perf_counter()
        ES = EvidenceSpace([p], infra_mode=cfg.infra_mode)
        ref = ES.pair_table(p, p.hosts, p)                       # all pairs, label-free
        pairs = ES.pair_table(p, bots, p)                        # seed pairs
        cdf = ReferenceCDF().fit(ref)
        pct = cdf.transform(pairs)
        pairs = pd.concat([pairs, pct], axis=1)
        pairs["S"] = fused_score(pct)
        pairs["S3"] = fused_score(pct[["F_E_flow", "F_E_stat", "F_E_inf"]])
        cfg.timings.append({"stage": "evidence_rq1", "sid": s, "m": m, "rep": rep,
                            "seconds": time.perf_counter() - t, "pairs": len(ref)})
        pairs["y"] = (r.reindex(pairs["b"]).to_numpy() == "bot").astype(int)
        pairs["activity"] = np.log10(pd.Series(p.n_flows, index=p.hosts).reindex(pairs["b"]).to_numpy())
        es_cos = EvidenceSpace([p], infra_mode="cos")
        pairs["inf_cos"] = es_cos.infra_matrix(p, bots, p, p.hosts)[
            [bots.index(a) for a in pairs["a"]], [p.hosts.index(b) for b in pairs["b"]]]
        pairs["v1_weights"] = v1_score(pairs)
        if with_variants:
            for name, kw in {"inf_dot2": dict(infra_mode="dot2"), "inf_cos": dict(infra_mode="cos"),
                             "inf_noidf": dict(use_idf=False),
                             "inf_ext": dict(external_only=True)}.items():
                es2 = EvidenceSpace([p], **kw)
                pairs[name] = es2.infra_matrix(p, bots, p, p.hosts)[
                    [bots.index(a) for a in pairs["a"]], [p.hosts.index(b) for b in pairs["b"]]]
                refv = es2.infra_matrix(p, p.hosts, p, p.hosts)
                refv = refv[~np.eye(len(p.hosts), dtype=bool)]
                srt = np.sort(refv)
                x = pairs[name].to_numpy()
                pairs["F_" + name] = (np.searchsorted(srt, x, "left") + np.searchsorted(srt, x, "right")) / (2 * len(srt))
            # pooled (cross-incident) IDF instead of incident-local IDF
            esp = EvidenceSpace(list(P.values()), infra_mode=cfg.infra_mode)
            pairs["inf_pooled"] = esp.infra_matrix(p, bots, p, p.hosts)[
                [bots.index(a) for a in pairs["a"]], [p.hosts.index(b) for b in pairs["b"]]]
            refv = esp.infra_matrix(p, p.hosts, p, p.hosts)[~np.eye(len(p.hosts), dtype=bool)]
            srt = np.sort(refv)
            x = pairs["inf_pooled"].to_numpy()
            pairs["F_inf_pooled"] = (np.searchsorted(srt, x, "left") + np.searchsorted(srt, x, "right")) / (2 * len(srt))
        # density clustering baseline on standardised statistics
        Z = ES.z_stats(p, list(range(len(p.hosts))))
        co = hdbscan_comembership(Z)
        ia = [p.hosts.index(a) for a in pairs["a"]]
        ib = [p.hosts.index(b) for b in pairs["b"]]
        pairs["hdbscan"] = co[ia, ib] + 1e-3 * pairs["E_stat"].to_numpy()
        if with_v1:
            t = time.perf_counter()
            fp = v1_fingerprints(flows[s])
            best = v1_cluster(fp)
            sim = membership_similarity(fp, best["labels"], bots, p.hosts)
            pairs["v1_pipeline"] = sim[[bots.index(a) for a in pairs["a"]], ib]
            cfg.timings.append({"stage": "v1_pipeline", "sid": s, "m": m, "rep": rep,
                                "seconds": time.perf_counter() - t, "fingerprints": len(fp)})
        pairs["m"], pairs["rep"], pairs["sid"] = m, rep, s
        out.append(pairs)
    if not out:                      # e.g. a truncated window with < 2 eligible infected hosts
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def _fit_logreg(train: pd.DataFrame, feats: list[str]):
    sc = StandardScaler().fit(train[feats])
    lr = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(train[feats]), train["y"])
    return sc, lr


def rq1_score(cfg: Config, pairs: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-family-out calibration and learned baseline; adds p_* columns."""
    pairs = pairs.copy()
    pairs["p_attribiq"] = np.nan
    pairs["p_logreg"] = np.nan
    calib = {}
    for s in sorted(pairs["sid"].unique()):
        test = pairs["sid"] == s
        train = pairs[pairs["sid"].map(FAMILY) != FAMILY[s]]
        cal = Calibrator().fit(train["S"].to_numpy(), train["y"].to_numpy())
        calib[str(int(s))] = [float(cal.alpha), float(cal.beta)]
        pairs.loc[test, "p_attribiq"] = cal.predict(pairs.loc[test, "S"].to_numpy())
        tr = train.dropna(subset=EVIDENCE)
        sc4, lr4 = _fit_logreg(tr, EVIDENCE)
        sc3, lr3 = _fit_logreg(tr, EVIDENCE[:3])
        te = pairs[test]
        has_t = te["E_tmp"].notna().to_numpy()
        p = np.empty(len(te))
        p[has_t] = lr4.predict_proba(sc4.transform(te.loc[has_t, EVIDENCE]))[:, 1]
        p[~has_t] = lr3.predict_proba(sc3.transform(te.loc[~has_t, EVIDENCE[:3]]))[:, 1]
        pairs.loc[test, "p_logreg"] = p
    pairs.attrs["calibrators"] = calib
    return pairs


RQ1_METHODS = ["S", "p_logreg", "v1_weights", "v1_pipeline", "hdbscan", "activity",
               "E_flow", "E_stat", "E_inf", "E_tmp"]


def rq1_metrics(pairs: pd.DataFrame, methods: list[str] = RQ1_METHODS) -> pd.DataFrame:
    rows = []
    for (m, rep, s, seed), g in pairs.groupby(["m", "rep", "sid", "a"]):
        y = g["y"].to_numpy()
        base = {"m": m, "rep": rep, "sid": s, "seed": seed, "n_cand": len(g),
                "n_pos": int(y.sum()), "prevalence": y.mean()}
        for meth in methods:
            if meth not in g:
                continue
            sc = g[meth].fillna(-np.inf if meth == "E_tmp" else 0).to_numpy()
            sc = np.where(np.isinf(sc), -2.0, sc)
            rows.append({**base, "method": meth, "AP": M.ap(y, sc), "AUROC": M.auroc(y, sc),
                         "Rprec": M.r_precision(y, sc)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ2: open-set cross-incident attribution against a library of past incidents
# ---------------------------------------------------------------------------
def rq2_pairs(cfg: Config, P: dict, roles: dict, m: float, rep: int) -> pd.DataFrame:
    ES = EvidenceSpace([P[s] for s in cfg.scenarios], infra_mode=cfg.infra_mode)
    ES_cos = EvidenceSpace([P[s] for s in cfg.scenarios], infra_mode="cos")
    lib = {s: [h for h in P[s].hosts if roles[s][h] == "bot"] for s in cfg.scenarios}
    ref_parts, pair_parts = [], []
    for sa in cfg.scenarios:
        for sb in cfg.scenarios:
            if sa == sb:
                continue
            ref_parts.append(ES.pair_table(P[sa], P[sa].hosts, P[sb], P[sb].hosts, use_temporal=False)
                             [EVIDENCE[:3]])
            if lib[sb]:
                pt = ES.pair_table(P[sa], P[sa].hosts, P[sb], lib[sb], use_temporal=False)
                pt["inf_cos"] = ES_cos.infra_matrix(P[sa], P[sa].hosts, P[sb], lib[sb]).ravel()
                pt["role_a"] = roles[sa].reindex(pt["a"]).to_numpy()
                pair_parts.append(pt)
    ref = pd.concat(ref_parts, ignore_index=True)
    pairs = pd.concat(pair_parts, ignore_index=True)
    cdf = ReferenceCDF(EVIDENCE[:3]).fit(ref)
    pct = cdf.transform(pairs)
    pairs = pd.concat([pairs, pct], axis=1)
    pairs["S3"] = fused_score(pct)
    pairs["v1_weights"] = v1_score(pairs)
    pairs["fam_a"] = pairs["sid_a"].map(FAMILY)
    pairs["fam_b"] = pairs["sid_b"].map(FAMILY)
    pairs["m"], pairs["rep"] = m, rep
    return pairs


def rq2_decide(cfg: Config, pairs: pd.DataFrame, rq1_pairs_all: pd.DataFrame) -> pd.DataFrame:
    """Per-query decisions. Calibrators are fitted on within-incident seed pairs
    of families other than the query's family (3-line fused score S3)."""
    meths = ["S3", "v1_weights", "E_flow", "E_stat", "E_inf"]
    keys = ["m", "rep", "sid_a", "a"]
    fam_max = pairs.groupby(keys + ["fam_b"])[meths].max().reset_index()
    lib_fams = pairs.groupby(keys)["fam_b"].agg(lambda s: frozenset(s))
    role = pairs.groupby(keys)["role_a"].first()
    out = role.rename("role").reset_index()
    out["lib_fams"] = lib_fams.to_numpy()
    for meth in meths:
        idx = fam_max.groupby(keys)[meth].idxmax()
        best = fam_max.loc[idx.to_numpy(), keys + ["fam_b", meth]].set_index(keys)
        out = out.join(best.rename(columns={"fam_b": f"{meth}_top_family", meth: f"{meth}_top_score"}),
                       on=keys)
    out["fam_q"] = out["sid_a"].map(FAMILY)
    known = (out["role"] == "bot").to_numpy() & np.array([f in L for f, L in zip(out["fam_q"], out["lib_fams"])])
    out["truth"] = np.where(known, out["fam_q"], "unknown")
    out["kind"] = np.where(known, "bot_known_family",
                           np.where(out["role"] == "bot", "bot_unknown_family", out["role"]))
    out["p_top"] = np.nan
    for fam_q in out["fam_q"].unique():
        tr = rq1_pairs_all[rq1_pairs_all["sid"].map(FAMILY) != fam_q]
        cal = Calibrator().fit(tr["S3"].to_numpy(), tr["y"].to_numpy())
        sel = out["fam_q"] == fam_q
        out.loc[sel, "p_top"] = cal.predict(out.loc[sel, "S3_top_score"].to_numpy())
    out["decision"] = np.where(out["p_top"] >= cfg.tau, out["S3_top_family"], "unknown")
    out["v1_decision"] = np.where(out["v1_weights_top_score"] >= 0.5,
                                  out["v1_weights_top_family"], "unknown")
    # Neyman-Pearson style thresholds: the (1 - alpha) quantile of the top score
    # among hosts confirmed as not infected in incidents of *other* families
    for alpha in (0.01, 0.05):
        for meth in ["S3", "v1_weights", "E_flow", "E_stat", "E_inf"]:
            col = f"tau{int(alpha * 100):02d}_{meth}"
            out[col] = np.nan
            for (m, rep, fam_q), g in out.groupby(["m", "rep", "fam_q"]):
                null = out[(out["m"] == m) & (out["rep"] == rep) & (out["fam_q"] != fam_q)
                           & (out["role"] != "bot")][f"{meth}_top_score"]
                out.loc[g.index, col] = float(np.quantile(null, 1 - alpha))
            out[f"dec{int(alpha * 100):02d}_{meth}"] = np.where(
                out[f"{meth}_top_score"] > out[col], out[f"{meth}_top_family"], "unknown")
    out = out.rename(columns={"sid_a": "sid", "a": "query"}).drop(columns=["lib_fams"])
    return out


# ---------------------------------------------------------------------------
# RQ4 helpers: replication of the original pipeline's clustering statistics
# ---------------------------------------------------------------------------
def v1_replication(cfg: Config, sid: int, all_sources: bool = True) -> list[dict]:
    from .data import read_binetflow
    from .scenarios import SCENARIOS as S
    import pathlib
    f = read_binetflow(pathlib.Path(cfg.raw_dir) / S[sid].filename)
    if not all_sources:
        from .data import is_internal
        f = f[is_internal(f["src"])]
    fp = v1_fingerprints(f)
    rows = []
    for thr in (1, 10):
        sub = fp[fp["conn_count"] >= thr].reset_index(drop=True)
        best = v1_cluster(sub)
        lab = best["labels"]
        mal = sub["malicious"].to_numpy()
        clusters = pd.Series(lab).value_counts()
        mal_cluster = pd.Series(lab[mal]).value_counts()
        top = int(mal_cluster.index[0]) if len(mal_cluster) else -1
        rows.append({"sid": sid, "all_sources": all_sources, "min_flows": thr,
                     "fingerprints": len(sub), "single_flow_share": float((sub["conn_count"] == 1).mean()),
                     "malicious_fingerprints": int(mal.sum()), "k": best["k"],
                     "silhouette": best["silhouette"], "davies_bouldin": best["davies_bouldin"],
                     "largest_cluster_share": float(clusters.iloc[0] / len(sub)),
                     "single_flow_share_largest": float((sub["conn_count"].to_numpy()[lab == clusters.index[0]] == 1).mean()),
                     "single_flow_share_rest": float((sub["conn_count"].to_numpy()[lab != clusters.index[0]] == 1).mean())
                     if (lab != clusters.index[0]).any() else float("nan"),
                     "bot_cluster_size": int(clusters.get(top, 0)),
                     "bot_cluster_precision": float(mal_cluster.iloc[0] / clusters.get(top, 1)) if top >= 0 else np.nan,
                     "bot_cluster_recall": float(mal_cluster.iloc[0] / max(mal.sum(), 1)) if top >= 0 else np.nan})
    return rows
