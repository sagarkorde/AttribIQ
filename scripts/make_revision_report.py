#!/usr/bin/env python3
"""Analyses added in the revision; writes results/report/revision/.

  * scenario-level (hierarchical) bootstrap for T1 differences
  * graph baseline (random walk with restart), tuned in-sample and nested
  * nested leave-one-scenario-out selection over the AttribIQ design space and over
    tuned weighted sums (selection optimism and baseline parity)
  * calibration diagnostics (adaptive bins, calibration slope, informative range,
    scenario-level intervals) for T1 and T2
  * open-set linkage over seven incidents incl. held-out scenario 2, with a threshold
    frozen on the development incidents and exact binomial intervals
  * per-family diagnostics, session-aware blending, reference-population size,
    candidate pruning and scaling

Usage: python scripts/make_revision_report.py --results results
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attribiq import metrics as M  # noqa: E402
from attribiq.conformal import conformal_threshold, min_labelled  # noqa: E402
from attribiq.experiments import FAMILY  # noqa: E402
from attribiq.scenarios import HELDOUT_SCENARIOS, PAPER_SCENARIOS  # noqa: E402

FR = [1.0, 0.5, 0.2, 0.1, 0.05]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NUM: dict = {}
UNIT = ["rep", "sid", "a"]


def key(m: float) -> str:
    return str(float(m))


# ---------------------------------------------------------------------------- helpers
def fast_ap(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision identical to sklearn's (ties form one threshold)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    npos = y.sum()
    if npos == 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    tp = np.cumsum(y)
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp_t = tp[last]
    prec = tp_t / (last + 1)
    rec = tp_t / npos
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def unit_ap(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(UNIT, sort=True).apply(lambda g: fast_ap(g["y"].to_numpy(), g[col].to_numpy()),
                                             include_groups=False)


def holm(p: dict) -> dict:
    items = sorted(p.items(), key=lambda kv: kv[1])
    out, run = {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (len(items) - i) * v))
        out[k] = run
    return out


def hier_boot(d: pd.Series, n_boot: int = 2000, seed: int = 1) -> tuple[float, float]:
    """Two-stage bootstrap: scenarios with replacement, then units within scenario."""
    rng = np.random.default_rng(seed)
    groups = [g.to_numpy() for _, g in d.groupby(level="sid")]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        parts = [groups[i][rng.integers(0, len(groups[i]), len(groups[i]))] for i in pick]
        vals[b] = np.concatenate(parts).mean()
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def ece_eqmass(y, p, n_bins=15) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    o = np.argsort(p, kind="mergesort")
    tot = 0.0
    for b in np.array_split(o, n_bins):
        if len(b):
            tot += len(b) / len(p) * abs(p[b].mean() - y[b].mean())
    return float(tot)


def cal_slope(y, p) -> tuple[float, float]:
    """Logistic recalibration of logit(p): slope < 1 means over-confident spread."""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=5000).fit(x, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def cp_ci(k: int, n: int) -> tuple[float, float]:
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--results", default="results")
    args = ap_.parse_args()
    R = Path(args.results)
    RV = R / "revision"
    out = R / "report" / "revision"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(R / "rq1_scored_pairs.parquet")
    g = pd.read_parquet(RV / "t1_graph_pruning.parquet")
    p = p.merge(g, on=["m", "rep", "sid", "a", "b"], how="left", validate="one_to_one")
    rwr_cols = [c for c in g.columns if c.startswith("rwr_")]
    assert p[rwr_cols[0]].notna().all(), "graph scores missing for some pairs"

    # sanity: fast AP equals sklearn AP
    t = p[(p.m == 0.2) & (p.rep == 0)]
    for (_, h) in list(t.groupby(UNIT))[:20]:
        assert abs(fast_ap(h.y.to_numpy(), h.S.to_numpy()) - M.ap(h.y.to_numpy(), h.S.to_numpy())) < 1e-12

    # ------------------------------------------------------------------ T1 unit APs
    base_meths = ["S", "v1_weights", "p_logreg", "hdbscan", "v1_pipeline", "activity"] + rwr_cols
    UA = {m: pd.DataFrame({c: unit_ap(p[p.m == m], c) for c in base_meths}) for m in FR}
    # graph baseline: best setting in-sample (averaged over levels) and nested by scenario
    mean_by_setting = {c: np.mean([UA[m][c].mean() for m in FR]) for c in rwr_cols}
    rwr_best = max(mean_by_setting, key=mean_by_setting.get)
    NUM["rwr_settings_mean_ap"] = mean_by_setting
    NUM["rwr_best"] = rwr_best
    sids = sorted(p["sid"].unique())
    for m in FR:
        nested = []
        for s in sids:
            tr = {c: np.mean([UA[mm][c][UA[mm].index.get_level_values("sid") != s].mean() for mm in FR]) for c in rwr_cols}
            pick = max(tr, key=tr.get)
            nested.append(UA[m][pick][UA[m].index.get_level_values("sid") == s])
        UA[m]["rwr_nested"] = pd.concat(nested).reindex(UA[m].index)
        UA[m]["rwr_best"] = UA[m][rwr_best]
    NUM["t1_ap"] = {c: {key(m): float(UA[m][c].mean()) for m in FR}
                    for c in ["S", "v1_weights", "p_logreg", "hdbscan", "v1_pipeline", "activity", "rwr_best", "rwr_nested"]}
    NUM["t1_prevalence"] = {key(m): float(p[p.m == m].groupby(UNIT)["y"].mean().mean()) for m in FR}

    # ------------------------------------------------------------------ unit-level and scenario-level tests
    comp = ["v1_weights", "p_logreg", "hdbscan", "rwr_best", "v1_pipeline", "activity"]
    tests = {}
    for m in FR:
        pv, rows = {}, {}
        for c in comp:
            d = (UA[m]["S"] - UA[m][c]).dropna()
            pv[c] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            lo, hi = hier_boot(d)
            per = d.groupby(level="sid").mean()
            rows[c] = {"delta": float(d.mean()), "hier_lo": lo, "hier_hi": hi,
                       "scen_pos": int((per > 0).sum()), "scen_n": int(len(per)),
                       "per_scenario": {str(int(k)): float(v) for k, v in per.items()}}
        ph = holm(pv)
        for c in comp:
            rows[c]["p_holm_unit"] = ph[c]
        tests[key(m)] = rows
    NUM["t1_tests"] = tests

    # ------------------------------------------------------------------ nested selection over the design space
    Fcols = {"E_flow": "F_E_flow", "E_stat": "F_E_stat", "E_tmp": "F_E_tmp"}
    infra = {"E_inf": "F_E_inf", "inf_dot2": "F_inf_dot2", "inf_cos": "F_inf_cos",
             "inf_noidf": "F_inf_noidf", "inf_ext": "F_inf_ext", "inf_pooled": "F_inf_pooled"}

    def agg(X: np.ndarray, rule: str) -> np.ndarray:
        with np.errstate(all="ignore"):
            if rule == "mean":
                return np.nanmean(X, axis=1)
            if rule == "median":
                return np.nanmedian(X, axis=1)
            if rule == "max":
                return np.nanmax(X, axis=1)
            if rule == "fisher":
                return np.nansum(-2 * np.log(np.clip(1 - X, 1e-9, 1)), axis=1)
            if rule == "drop_lowest":
                n = np.sum(~np.isnan(X), axis=1)
                tot = np.nansum(X, axis=1) - np.where(n > 1, np.nanmin(X, axis=1), 0)
                return tot / np.maximum(n - 1, 1)
        raise ValueError(rule)

    configs = []
    for r in range(1, 5):
        for sub in itertools.combinations(["E_flow", "E_stat", "E_inf", "E_tmp"], r):
            for iv in (infra if "E_inf" in sub else {"E_inf": None}):
                for rule in ("mean", "median", "drop_lowest", "fisher", "max"):
                    if len(sub) == 1 and rule != "mean":
                        continue
                    configs.append((sub, iv, rule))
    print(f"design space: {len(configs)} configurations", flush=True)
    grp = {m: p[p.m == m].groupby(UNIT, sort=True) for m in FR}
    idx_units = {m: list(grp[m].groups.keys()) for m in FR}
    unit_rows = {m: [grp[m].indices[u] for u in idx_units[m]] for m in FR}
    Y = {m: p[p.m == m]["y"].to_numpy() for m in FR}
    design = {}
    for sub, iv, rule in configs:
        cols = [infra[iv] if c == "E_inf" else Fcols[c] for c in sub]
        name = f"{'+'.join(sub)}|{iv}|{rule}"
        design[name] = {}
        for m in FR:
            X = p[p.m == m][cols].to_numpy(dtype=float)
            sc = agg(X, rule)
            design[name][m] = pd.Series([fast_ap(Y[m][r_], sc[r_]) for r_ in unit_rows[m]],
                                        index=pd.MultiIndex.from_tuples(idx_units[m], names=UNIT))
    fixed = "E_flow+E_stat+E_inf+E_tmp|E_inf|mean"
    # tuned weighted sums of raw evidence (original-style score), weights on a 0.1 simplex grid
    raw = {"E_flow": "E_flow", "E_stat": "E_stat", "inf_cos": "inf_cos", "E_tmp": "E_tmp"}
    grid = [w for w in itertools.product(range(11), repeat=4) if sum(w) == 10]
    wsum = {}
    for w in grid:
        name = "w" + "".join(map(str, w))
        wsum[name] = {}
        for m in FR:
            X = p[p.m == m][list(raw.values())].to_numpy(dtype=float, copy=True)
            X[:, 3] = np.nan_to_num(np.clip(X[:, 3], 0, 1))
            sc = X @ (np.array(w) / 10)
            wsum[name][m] = pd.Series([fast_ap(Y[m][r_], sc[r_]) for r_ in unit_rows[m]],
                                      index=pd.MultiIndex.from_tuples(idx_units[m], names=UNIT))
    print("design and weight grids evaluated", flush=True)

    def nested(space: dict, m: float) -> tuple[pd.Series, list]:
        parts, picks = [], []
        for s in sids:
            score = {n: np.mean([v[mm][v[mm].index.get_level_values("sid") != s].mean() for mm in FR])
                     for n, v in space.items()}
            pick = max(score, key=score.get)
            picks.append(pick)
            parts.append(space[pick][m][space[pick][m].index.get_level_values("sid") == s])
        return pd.concat(parts), picks

    insample_best = max(design, key=lambda n: np.mean([design[n][m].mean() for m in FR]))
    insample_w = max(wsum, key=lambda n: np.mean([wsum[n][m].mean() for m in FR]))
    sel = {}
    for m in FR:
        nd, picks_d = nested(design, m)
        nw, picks_w = nested(wsum, m)
        sel[key(m)] = {
            "fixed": float(design[fixed][m].mean()),
            "design_nested": float(nd.mean()),
            "design_insample_best": float(design[insample_best][m].mean()),
            "wsum_nested": float(nw.mean()),
            "wsum_insample_best": float(wsum[insample_w][m].mean()),
            "orig_weights": float(UA[m]["v1_weights"].mean()),
            "rwr_nested": float(UA[m]["rwr_nested"].mean()),
            "rwr_insample_best": float(UA[m]["rwr_best"].mean()),
        }
        UA[m]["design_nested"] = nd.reindex(UA[m].index)
        UA[m]["wsum_nested"] = nw.reindex(UA[m].index)
    NUM["selection"] = sel
    NUM["selection_meta"] = {"n_design": len(design), "n_weights": len(grid),
                             "insample_best_design": insample_best, "insample_best_weights": insample_w,
                             "nested_design_picks": nested(design, 0.2)[1], "nested_weight_picks": nested(wsum, 0.2)[1]}
    # tuned comparisons with scenario-level intervals
    tt = {}
    for m in FR:
        tt[key(m)] = {}
        for c in ["wsum_nested", "rwr_nested", "design_nested"]:
            d = (UA[m]["S"] - UA[m][c]).dropna()
            lo, hi = hier_boot(d)
            tt[key(m)][c] = {"delta": float(d.mean()), "hier_lo": lo, "hier_hi": hi,
                             "p_unit": float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0,
                             "scen_pos": int((d.groupby(level="sid").mean() > 0).sum())}
    for m in FR:
        d = (UA[m]["design_nested"] - UA[m]["wsum_nested"]).dropna()
        lo, hi = hier_boot(d)
        tt[key(m)]["nested_design_vs_nested_wsum"] = {"delta": float(d.mean()), "hier_lo": lo, "hier_hi": hi,
                                                      "scen_pos": int((d.groupby(level="sid").mean() > 0).sum())}
    NUM["t1_tuned_tests"] = tt

    # ------------------------------------------------------------------ calibration (T1)
    cal = {}
    for m in FR:
        q = p[p.m == m]
        cal[key(m)] = {}
        for c in ["p_attribiq", "p_logreg", "v1_weights"]:
            y, pr = q["y"].to_numpy(), q[c].clip(0, 1).to_numpy()
            hi = pr >= 0.1
            boots = []
            rng = np.random.default_rng(3)
            gl = [np.flatnonzero(q["sid"].to_numpy() == s) for s in sids]
            for _ in range(500):
                pick = np.concatenate([gl[i] for i in rng.integers(0, len(gl), len(gl))])
                boots.append(M.ece(y[pick], pr[pick]))
            sl, ic = cal_slope(y, pr)
            cal[key(m)][c] = {"ece": M.ece(y, pr), "ece_lo": float(np.quantile(boots, .025)),
                              "ece_hi": float(np.quantile(boots, .975)), "ece_adaptive": ece_eqmass(y, pr),
                              "ece_hi_range": M.ece(y[hi], pr[hi]) if hi.sum() >= 50 else None,
                              "share_hi_range": float(hi.mean()), "slope": sl, "intercept": ic,
                              "mean_p_pos": float(pr[y == 1].mean()), "mean_p_neg": float(pr[y == 0].mean()),
                              "brier": M.brier(y, pr)}
    NUM["t1_calibration"] = cal

    # ------------------------------------------------------------------ per-family diagnostics
    diag = {}
    for m in (1.0, 0.2):
        q = p[p.m == m]
        for s in sids:
            qs = q[q.sid == s]
            pos = qs[qs.y == 1]
            ua = {c: float(UA[m][c][UA[m].index.get_level_values("sid") == s].mean()) for c in ["S"]}
            for c in ["E_flow", "E_stat", "E_inf", "E_tmp"]:
                ua[c] = float(qs.groupby(UNIT).apply(lambda h: fast_ap(h.y.to_numpy(), h[c].fillna(-1).to_numpy()),
                                                    include_groups=False).mean())
            diag.setdefault(str(int(s)), {})[key(m)] = {
                "family": FAMILY[s], "seeds_per_rep": float(qs.groupby("rep")["a"].nunique().mean()),
                "candidates": float(qs.groupby(["rep", "a"]).size().mean()), "ap": ua,
                "true_pair_pct": {c: float(pos[f"F_{c}"].mean()) for c in ["E_flow", "E_stat", "E_inf", "E_tmp"]},
                "true_pair_tmp_available": float(pos["E_tmp"].notna().mean())}
    NUM["per_family"] = diag

    # ------------------------------------------------------------------ session-aware blending
    ss = pd.read_parquet(RV / "t1_session_pairs.parquet")
    ses = {}
    for m in (0.5, 0.2, 0.1):
        q = ss[ss.m == m]
        base = p[p.m == m]
        ses[key(m)] = {}
        for c in ["S", "S3", "v1_weights", "p_logreg", "hdbscan", "E_flow", "E_stat", "E_inf", "E_tmp"]:
            a_s = unit_ap(q.assign(**{c: q[c].fillna(-1)}), c)
            a_b = unit_ap(base.assign(**{c: base[c].fillna(-1)}), c)
            ses[key(m)][c] = {"session": float(a_s.mean()), "standard": float(a_b.mean())}
        dS = unit_ap(q, "S") - unit_ap(q, "v1_weights")
        lo, hi = hier_boot(dS.dropna())
        ses[key(m)]["delta_S_vs_v1"] = {"delta": float(dS.mean()), "hier_lo": lo, "hier_hi": hi,
                                        "p_unit": float(wilcoxon(dS.dropna()).pvalue)}
        ses[key(m)]["ece_attribiq"] = M.ece(q["y"].to_numpy(), q["p_attribiq"].to_numpy())
    NUM["session"] = ses

    # ------------------------------------------------------------------ reference-population size
    cc = pd.read_parquet(RV / "t1_contamination_pairs.parquet")
    con = {}
    for (m, keep), q in cc.groupby(["m", "keep"]):
        a = unit_ap(q, "S")
        con.setdefault(key(m), {})[str(keep)] = {
            "ap": float(a.mean()), "prevalence": float(q.groupby(UNIT)["y"].mean().mean()),
            "infected_share_hosts": float((q.groupby("sid")["n_infected"].first() / q.groupby("sid")["n_hosts"].first()).mean()),
            "ece": M.ece(q["y"].to_numpy(), q["p_attribiq"].to_numpy()),
            "mean_p_pos": float(q.loc[q.y == 1, "p_attribiq"].mean()),
            "mean_p_neg": float(q.loc[q.y == 0, "p_attribiq"].mean()),
            "ap_v1": float(unit_ap(q, "v1_weights").mean())}
    NUM["reference_population"] = con

    # ------------------------------------------------------------------ pruning and scaling
    pr = {}
    for m in FR:
        q = p[p.m == m]
        pos = q[q.y == 1]
        pr[key(m)] = {f"recall@{k_}": float((pos["prune_rank"] <= k_).mean()) for k_ in (10, 25, 50)}
        for k_ in (25, 50):
            s2 = np.where(q["prune_rank"] <= k_, q["S"], -1.0)
            pr[key(m)][f"ap_pruned@{k_}"] = float(unit_ap(q.assign(Sp=s2), "Sp").mean())
    NUM["pruning"] = pr
    sc = pd.read_csv(RV / "scaling.csv")
    NUM["scaling"] = sc.to_dict(orient="records")
    ex = sc.dropna(subset=["exhaustive_s"])
    coef = np.polyfit(ex["n_hosts"].to_numpy(float) ** 2, ex["exhaustive_s"].to_numpy(float), 1)
    NUM["scaling_exhaustive_fit"] = {"sec_per_million_pairs": float(coef[0] * 1e6), "intercept": float(coef[1])}

    # ------------------------------------------------------------------ T2 over seven incidents incl. held-out scenario 2
    d = pd.read_parquet(RV / "t2_heldout_decisions.parquet")
    dev = set(PAPER_SCENARIOS)
    t2 = {}
    for m in FR:
        q = d[d.m == m]
        known = q[q.kind == "bot_known_family"]
        unk = q[q.kind == "bot_unknown_family"]
        ben = q[q.role != "bot"]
        rec = (known["dec01_S3"] == known["truth"]).mean()
        kb = int((ben["dec01_S3"] != "unknown").sum())
        t2[key(m)] = {
            "n_known_per_rep": len(known) / q.rep.nunique(), "known_families": sorted(known["fam_q"].unique()),
            "top1": float((known["S3_top_family"] == known["truth"]).mean()),
            "auroc": M.auroc(np.r_[np.ones(len(known)), np.zeros(len(ben))],
                             np.r_[known["S3_top_score"], ben["S3_top_score"]]),
            "recall01": float(rec), "unknown_attr01": float((unk["dec01_S3"] != "unknown").mean()),
            "benign_attr01": kb / len(ben), "benign_attr01_ci": cp_ci(kb, len(ben)), "n_benign": len(ben),
            "recall01_by_family": {f: float((g_["dec01_S3"] == g_["truth"]).mean()) for f, g_ in known.groupby("fam_q")},
            "top1_by_family": {f: float((g_["S3_top_family"] == g_["truth"]).mean()) for f, g_ in known.groupby("fam_q")},
            "v1_recall01": float((known["dec01_v1_weights"] == known["truth"]).mean()),
            "v1_top1": float((known["v1_weights_top_family"] == known["truth"]).mean()),
        }
        # held-out incident: threshold frozen on development incidents only
        h = q[q.sid.isin(HELDOUT_SCENARIOS)]
        hb = h[h.role != "bot"]
        hk = h[h.kind == "bot_known_family"]
        taus = []
        for rep, g_ in h.groupby("rep"):
            null = q[(q.rep == rep) & q.sid.isin(dev) & (q.role != "bot") & (q.fam_q != "Neris")]["S3_top_score"]
            taus.append((rep, float(np.quantile(null, 0.99))))
        tau = dict(taus)
        hb_att = sum(int((g_["S3_top_score"] > tau[r]).sum()) for r, g_ in hb.groupby("rep"))
        hk_ok = sum(int(((g_["S3_top_score"] > tau[r]) & (g_["S3_top_family"] == g_["truth"])).sum())
                    for r, g_ in hk.groupby("rep"))
        cal_ok = known.dropna(subset=["p_top"])
        t2[key(m)]["heldout"] = {
            "benign_n": len(hb), "benign_attr": hb_att, "benign_rate": hb_att / max(len(hb), 1),
            "benign_ci": cp_ci(hb_att, len(hb)), "known_n": len(hk), "known_correct_frozen": hk_ok,
            "known_top1": float((hk["S3_top_family"] == hk["truth"]).mean()) if len(hk) else None,
            "known_rank_family": None}
        # calibration of the T2 probability: is the top family correct?
        allq = q.dropna(subset=["p_top"])
        corr = ((allq["S3_top_family"] == allq["truth"]) & (allq["kind"] == "bot_known_family")).astype(int)
        t2[key(m)]["p_top_ece"] = M.ece(corr.to_numpy(), allq["p_top"].to_numpy())
        t2[key(m)]["p_top_mean_known"] = float(cal_ok["p_top"].mean())
    NUM["t2_extended"] = t2

    # per-incident split-conformal thresholds: a labelled sample of the incident's own
    # non-infected hosts sets the threshold; evaluated on the incident's other hosts
    rng = np.random.default_rng(11)
    conf = {}
    for eps in (0.01, 0.05):
        for n_cal in (50, 100, 150):
            for m in FR:
                q = d[d.m == m]
                rows_far, rows_rec, rows_ho, rows_ho_rec, n_inc = [], [], [], [], set()
                for (sid_, rep), g_ in q.groupby(["sid", "rep"]):
                    ben = g_[g_.role != "bot"]["S3_top_score"].to_numpy()
                    kn = g_[g_.kind == "bot_known_family"]
                    if len(ben) < n_cal + 50:
                        continue
                    n_inc.add(int(sid_))
                    for _ in range(100):
                        perm = rng.permutation(len(ben))
                        tau = conformal_threshold(ben[perm[:n_cal]], eps)
                        far = float((ben[perm[n_cal:]] > tau).mean())
                        rows_far.append(far)
                        if int(sid_) in HELDOUT_SCENARIOS:
                            rows_ho.append(far)
                        if len(kn):
                            ok = ((kn["S3_top_score"] > tau) & (kn["S3_top_family"] == kn["truth"])).mean()
                            rows_rec.append(float(ok))
                            if int(sid_) in HELDOUT_SCENARIOS:
                                rows_ho_rec.append(float(ok))
                conf.setdefault(str(eps), {}).setdefault(str(n_cal), {})[key(m)] = {
                    "far_mean": float(np.mean(rows_far)) if rows_far else None,
                    "far_p95": float(np.quantile(rows_far, 0.95)) if rows_far else None,
                    "recall_mean": float(np.mean(rows_rec)) if rows_rec else None,
                    "heldout_far_mean": float(np.mean(rows_ho)) if rows_ho else None,
                    "heldout_far_p95": float(np.quantile(rows_ho, 0.95)) if rows_ho else None,
                    "incidents": sorted(n_inc)}
    NUM["t2_conformal"] = conf
    NUM["t2_conformal_min_labelled"] = {"0.01": min_labelled(0.01), "0.05": min_labelled(0.05)}
    NUM["t2_benign_per_incident"] = {str(int(s_)): float(g_.groupby("rep").apply(lambda z: (z.role != "bot").sum(), include_groups=False).mean())
                                     for s_, g_ in d[d.m == 1.0].groupby("sid")}
    NUM["t2_counts"] = {"queries_per_condition": int(len(d[(d.m == 1.0)]) / d.rep.nunique()),
                        "incidents": sorted(int(s) for s in d["sid"].unique())}

    # ------------------------------------------------------------------ analyst budget: precision@k and recall@k
    def topk_hits(y: np.ndarray, sc: np.ndarray, k_: int) -> float:
        """Expected related hosts among the top k under random tie-breaking."""
        o = np.argsort(-sc, kind="mergesort")
        s_sorted, y_sorted = sc[o], y[o]
        if k_ >= len(sc):
            return float(y.sum())
        thr = s_sorted[k_ - 1]
        above = s_sorted > thr
        tie = s_sorted == thr
        return float(y_sorted[above].sum() + (k_ - above.sum()) * y_sorted[tie].mean())

    budget = {}
    for m in FR:
        q = p[p.m == m]
        budget[key(m)] = {}
        for c in ["S", "v1_weights", "rwr_best", "hdbscan", "p_logreg"]:
            col = rwr_best if c == "rwr_best" else c
            rec, prec = {}, {}
            for k_ in (5, 10, 20):
                h = [topk_hits(g["y"].to_numpy(float), g[col].to_numpy(float), k_) for _, g in q.groupby(UNIT)]
                npos = [g["y"].sum() for _, g in q.groupby(UNIT)]
                rec[str(k_)] = float(np.mean(np.array(h) / np.array(npos)))
                prec[str(k_)] = float(np.mean(np.array(h) / k_))
            budget[key(m)][c] = {"recall": rec, "precision": prec}
        budget[key(m)]["related_per_seed"] = float(q.groupby(UNIT)["y"].sum().mean())
        budget[key(m)]["candidates_per_seed"] = float(q.groupby(UNIT).size().mean())
    NUM["analyst_budget"] = budget

    # ------------------------------------------------------------------ isotonic recalibration (leave-one-family-out)
    from sklearn.isotonic import IsotonicRegression
    iso = np.full(len(p), np.nan)
    for s_ in sids:
        test = (p["sid"] == s_).to_numpy()
        train = (p["sid"].map(FAMILY) != FAMILY[s_]).to_numpy()
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p.loc[train, "S"], p.loc[train, "y"])
        iso[test] = ir.predict(p.loc[test, "S"])
    p["p_isotonic"] = iso
    isoc = {}
    for m in FR:
        q = p[p.m == m]
        y = q["y"].to_numpy()
        for c in ["p_attribiq", "p_isotonic"]:
            pr_ = q[c].clip(0, 1).to_numpy()
            hi = pr_ >= 0.1
            isoc.setdefault(key(m), {})[c] = {"ece": M.ece(y, pr_), "ece_hi_range": M.ece(y[hi], pr_[hi]) if hi.sum() >= 50 else None,
                                              "share_hi_range": float(hi.mean()), "brier": M.brier(y, pr_)}
    NUM["isotonic"] = isoc

    # ------------------------------------------------------------------ is the fixed design an outlier in any scenario?
    rank = {}
    for m in FR:
        rank[key(m)] = {}
        for s_ in sids:
            vals = np.array([design[n][m][design[n][m].index.get_level_values("sid") == s_].mean() for n in design])
            f_ = design[fixed][m][design[fixed][m].index.get_level_values("sid") == s_].mean()
            rank[key(m)][str(int(s_))] = float((vals <= f_ + 1e-12).mean())
    NUM["fixed_design_percentile_by_scenario"] = rank

    # ------------------------------------------------------------------ partner-pool sensitivity of blending
    pp = pd.read_parquet(RV / "t1_partner_pairs.parquet")
    base3 = p[(p.m == 0.2) & (p.rep < 3)]
    part = {"50": {c: float(unit_ap(base3, c).mean()) for c in ["S", "v1_weights", "p_logreg", "hdbscan"]}}
    for mf, q in pp.groupby("partner_min_flows"):
        part[str(int(mf))] = {c: float(unit_ap(q, c).mean()) for c in ["S", "v1_weights", "p_logreg", "hdbscan"]}
    NUM["partner_pool"] = part

    # ------------------------------------------------------------------ tables
    rows = []
    for m in FR:
        s = sel[key(m)]
        rows.append({"share": m, **{k_: round(v, 3) for k_, v in s.items()}})
    pd.DataFrame(rows).to_csv(out / "tables" / "table_selection_parity.csv", index=False)
    pd.DataFrame([{"share": m, "method": c, **{k_: v for k_, v in tests[key(m)][c].items() if k_ != "per_scenario"}}
                  for m in FR for c in comp]).to_csv(out / "tables" / "table_t1_tests_hierarchical.csv", index=False)
    pd.DataFrame([{"share": m, **{k_: v for k_, v in t2[key(m)].items() if not isinstance(v, (dict, list, tuple))}}
                  for m in FR]).to_csv(out / "tables" / "table_t2_extended.csv", index=False)

    # ------------------------------------------------------------------ figures (IEEE column widths)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    xs = np.arange(len(FR))
    lab = ["100%", "50%", "20%", "10%", "5%"]
    # Fig. 1: T1 average precision (single column)
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    lines = [("S", "AttribIQ", SERIES[0], "o"), ("v1_weights", "Original weights", SERIES[1], "s"),
             ("p_logreg", "Learned logistic", SERIES[2], "^"), ("hdbscan", "HDBSCAN", SERIES[3], "D"),
             ("rwr_best", "Random walk (tuned)", SERIES[4], "v")]
    for c, l, col, mk in lines:
        ax.plot(xs, [UA[m][c].mean() for m in FR], marker=mk, color=col, label=l, linewidth=1.3, markersize=3.5)
    sx = [i for i, m in enumerate(FR) if key(m) in ses]
    ax.plot(sx, [ses[key(FR[i])]["S"]["session"] for i in sx], marker="o", color=SERIES[0], linestyle="--",
            linewidth=1.0, markersize=3.2, mfc="white", label="AttribIQ, session-aware")
    ax.plot(sx, [ses[key(FR[i])]["v1_weights"]["session"] for i in sx], marker="s", color=SERIES[1], linestyle="--",
            linewidth=1.0, markersize=3.2, mfc="white", label="Original weights, session-aware")
    ax.plot(xs, [NUM["t1_prevalence"][key(m)] for m in FR], color=INK2, linestyle=":", linewidth=1, label="Random ranking")
    ax.set_xticks(xs, lab)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Average precision (T1)", color=INK)
    ax.set_xlabel("Malicious share of an infected host's traffic", color=INK)
    style(ax)
    ax.legend(fontsize=5.4, frameon=False, loc="upper right", handlelength=2.2, ncol=1, labelspacing=0.25)
    fig.savefig(out / "figures" / "fig1_t1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    # Fig. 2: (a) T2 over seven incidents, (b) calibration in the informative range
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={"wspace": 0.42})
    ax = axs[0]
    ax.plot(xs, [t2[key(m)]["auroc"] for m in FR], marker="o", color=SERIES[0], linewidth=1.2, markersize=3, label="AUROC")
    ax.plot(xs, [t2[key(m)]["top1"] for m in FR], marker="s", color=SERIES[6], linewidth=1.2, markersize=3, label="Top-1 family")
    ax.plot(xs, [t2[key(m)]["recall01"] for m in FR], marker="^", color=SERIES[1], linewidth=1.2, markersize=3, label="Recall, 1% FAR")
    ax.axhline(0.5, color=INK2, linestyle=":", linewidth=0.9)
    ax.set_xticks(xs, ["100", "50", "20", "10", "5"])
    ax.set_xlabel("Malicious share (%)", color=INK)
    ax.set_ylim(0, 1.03)
    ax.set_title("(a) T2, seven incidents", fontsize=6.8, color=INK, loc="left")
    style(ax)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.2, frameon=False, loc="lower left", handlelength=1.6)
    ax = axs[1]
    for i, m in enumerate((1.0, 0.2)):
        q = p[p.m == m]
        y, pr_ = q["y"].to_numpy(), q["p_attribiq"].to_numpy()
        o = np.argsort(pr_, kind="mergesort")
        top = o[-int(0.1 * len(o)):]
        bins = np.array_split(top, 8)
        ax.plot([pr_[b].mean() for b in bins], [y[b].mean() for b in bins], marker="o" if i == 0 else "s",
                color=SERIES[0] if i == 0 else SERIES[1], linewidth=1.1, markersize=2.8,
                label="100% malicious" if i == 0 else "20% malicious")
    ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Predicted p (top 10% of pairs)", color=INK)
    ax.set_ylabel("Observed related", color=INK)
    ax.set_title("(b) T1 calibration", fontsize=6.8, color=INK, loc="left")
    style(ax)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left", handlelength=1.6)
    fig.savefig(out / "figures" / "fig2_t2_calibration.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    json.dump(clean(NUM), open(out / "revision_numbers.json", "w"), indent=1, allow_nan=False)
    print("revision report written", flush=True)


if __name__ == "__main__":
    main()
