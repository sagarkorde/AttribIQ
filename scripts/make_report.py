#!/usr/bin/env python3
"""Build every table, figure and reported number from results/ (no recomputation
of evidence). Writes results/report/{tables,figures} and results/report/numbers.json.

    python scripts/make_report.py --results results --raw data/raw --cache data/cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attribiq import metrics as M  # noqa: E402
from attribiq.data import host_roles, load_scenario  # noqa: E402
from attribiq.scenarios import PAPER_SCENARIOS, SCENARIOS  # noqa: E402

FAM = {s: SCENARIOS[s].family for s in SCENARIOS}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FR = [1.0, 0.5, 0.2, 0.1, 0.05]
FR_LABEL = ["100%", "50%", "20%", "10%", "5%"]
METHOD_LABEL = {
    "S": "AttribIQ (fused)", "v1_weights": "Original weights", "v1_pipeline": "Original pipeline",
    "p_logreg": "Learned logistic", "hdbscan": "HDBSCAN co-membership", "activity": "Activity rank",
    "E_flow": "Flow mix (E_flow)", "E_stat": "Statistics (E_stat)", "E_inf": "Shared rare infra. (E_inf)",
    "E_tmp": "Co-activity (E_tmp)",
}
NUM: dict = {}


def style(ax, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK, fontsize=9)


def ci_row(v, groups=None):
    mean, lo, hi = M.bootstrap_ci(np.asarray(v, dtype=float), n_boot=2000, groups=groups)
    return mean, lo, hi


def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    n = len(items)
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (n - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


# ---------------------------------------------------------------------------
def table_dataset(args, out):
    rows = []
    for s in PAPER_SCENARIOS:
        f = load_scenario(s, args.raw, args.cache)
        r = host_roles(f)
        n = f.groupby("src", observed=True).size()
        elig = n[n >= 10].index
        rows.append({
            "Scenario": s, "Family": FAM[s], "Flows": SCENARIOS[s].n_flows,
            "Internal-source flows": len(f),
            "Duration (h)": round((f["ts"].max() - f["ts"].min()) / 3600, 2),
            "Internal hosts": len(n), "Eligible hosts (>=10 flows)": len(elig),
            "Infected hosts (eligible/all)": f"{int((r.reindex(elig) == 'bot').sum())}/{int((r == 'bot').sum())}",
            "Normal-labelled hosts (eligible)": int((r.reindex(elig) == "normal").sum()),
        })
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "table_dataset.csv", index=False)
    NUM["dataset"] = t.to_dict(orient="records")
    NUM["dataset_totals"] = {"flows": int(t["Flows"].sum()), "internal_flows": int(t["Internal-source flows"].sum()),
                             "eligible_hosts": int(t["Eligible hosts (>=10 flows)"].sum())}
    return t


def rq1(args, out):
    met = pd.read_csv(Path(args.results) / "rq1_metrics.csv")
    met["unit"] = met["rep"].astype(str) + "|" + met["sid"].astype(str) + "|" + met["seed"]
    NUM["rq1_units_per_condition"] = int(met[(met.m == 1.0) & (met.method == "S")].shape[0])
    NUM["rq1_seeds"] = int(met[(met.m == 1.0) & (met.method == "S") & (met.rep == 0)].shape[0])
    rows = []
    for m in FR:
        g = met[met.m == m]
        prev = g[g.method == "S"]["prevalence"]
        row = {"m": m, "Random (prevalence)": f"{prev.mean():.3f}"}
        for meth in ["S", "v1_weights", "v1_pipeline", "p_logreg", "hdbscan", "activity",
                     "E_flow", "E_stat", "E_inf", "E_tmp"]:
            v = g[g.method == meth].sort_values("unit")
            mean, lo, hi = ci_row(v["AP"].to_numpy())
            row[METHOD_LABEL[meth]] = fmt(mean, lo, hi)
            NUM.setdefault("rq1_ap", {}).setdefault(meth, {})[str(m)] = [mean, lo, hi]
        NUM.setdefault("rq1_prevalence", {})[str(m)] = float(prev.mean())
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "table_rq1_ap.csv", index=False)

    # paired tests: AttribIQ vs each baseline, Wilcoxon signed-rank, Holm within m
    trows = []
    for m in FR:
        g = met[met.m == m].pivot_table(index="unit", columns="method", values="AP")
        pv, eff = {}, {}
        for b in ["v1_weights", "v1_pipeline", "p_logreg", "hdbscan", "activity",
                  "E_flow", "E_stat", "E_inf", "E_tmp"]:
            d = (g["S"] - g[b]).dropna()
            stat = wilcoxon(d) if (d != 0).any() else None
            pv[b] = float(stat.pvalue) if stat is not None else 1.0
            mean, lo, hi, _ = M.paired_bootstrap(g["S"].to_numpy(), g[b].to_numpy())
            wins = float((d > 0).mean())
            eff[b] = (mean, lo, hi, wins, len(d))
        adj = holm(pv)
        for b in pv:
            mean, lo, hi, wins, n = eff[b]
            trows.append({"m": m, "baseline": METHOD_LABEL[b], "delta_AP": round(mean, 3),
                          "CI95": f"[{lo:.3f}, {hi:.3f}]", "AttribIQ_better_share": round(wins, 2),
                          "n": n, "p_wilcoxon": pv[b], "p_holm": adj[b]})
            NUM.setdefault("rq1_tests", {}).setdefault(str(m), {})[b] = {
                "delta": mean, "lo": lo, "hi": hi, "p_holm": adj[b], "wins": wins, "n": n}
    pd.DataFrame(trows).to_csv(out / "tables" / "table_rq1_tests.csv", index=False)

    # per-scenario breakdown for AttribIQ
    ps = met[met.method.isin(["S", "v1_weights", "E_flow", "E_inf"])].groupby(
        ["m", "sid", "method"])["AP"].mean().unstack(["sid"]).round(3)
    ps.to_csv(out / "tables" / "table_rq1_per_scenario.csv")
    NUM["rq1_per_scenario_S"] = met[met.method == "S"].groupby(["m", "sid"])["AP"].mean().round(3).unstack().to_dict()

    # figure 2: methods (a) and evidence lines (b)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    x = np.arange(len(FR))
    for ax, meths, title in ((axes[0], ["S", "v1_weights", "p_logreg", "hdbscan", "v1_pipeline", "activity"],
                              "(a) AttribIQ and baselines"),
                             (axes[1], ["S", "E_flow", "E_stat", "E_inf", "E_tmp"], "(b) Single evidence lines")):
        for i, meth in enumerate(meths):
            vals = np.array([NUM["rq1_ap"][meth][str(m)] for m in FR])
            ax.errorbar(x + (i - len(meths) / 2) * 0.05, vals[:, 0],
                        yerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]],
                        color=SERIES[i], marker=MARKERS[i], markersize=5, linewidth=1.6,
                        capsize=2, elinewidth=0.8, label=METHOD_LABEL[meth])
        prev = [NUM["rq1_prevalence"][str(m)] for m in FR]
        ax.plot(x, prev, color=INK2, linestyle=":", linewidth=1.2, label="Random (prevalence)")
        ax.set_xticks(x, FR_LABEL)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax, "Malicious share of infected host's traffic", "Average precision" if ax is axes[0] else None)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig2_rq1_ap_vs_blending.png", dpi=300)
    plt.close(fig)
    return t


def rq2(args, out):
    d = pd.read_csv(Path(args.results) / "rq2_decisions.csv")
    rows = []
    for m in FR:
        g = d[d.m == m]
        known = g[g.kind == "bot_known_family"]
        unk = g[g.kind == "bot_unknown_family"]
        benign = g[g.kind.isin(["normal", "background"])]
        row = {"m": m, "n_known": len(known), "n_unknown_bot": len(unk), "n_benign": len(benign)}
        for meth in ["S3", "v1_weights", "E_flow", "E_stat", "E_inf"]:
            top1 = (known[f"{meth}_top_family"] == known["truth"]).mean()
            y = np.r_[np.ones(len(known)), np.zeros(len(benign))]
            s = np.r_[known[f"{meth}_top_score"], benign[f"{meth}_top_score"]]
            au_b = M.auroc(y, s)
            y2 = np.r_[np.ones(len(known)), np.zeros(len(unk))]
            s2 = np.r_[known[f"{meth}_top_score"], unk[f"{meth}_top_score"]]
            au_u = M.auroc(y2, s2)
            rec01 = (known[f"dec01_{meth}"] == known["truth"]).mean()
            rec05 = (known[f"dec05_{meth}"] == known["truth"]).mean()
            far_u01 = (unk[f"dec01_{meth}"] != "unknown").mean()
            far_b01 = (benign[f"dec01_{meth}"] != "unknown").mean()
            far_u05 = (unk[f"dec05_{meth}"] != "unknown").mean()
            far_b05 = (benign[f"dec05_{meth}"] != "unknown").mean()
            NUM.setdefault("rq2", {}).setdefault(meth, {})[str(m)] = {
                "top1": top1, "auroc_vs_benign": au_b, "auroc_vs_unknown_bots": au_u,
                "recall_far01": rec01, "recall_far05": rec05, "far_unknown_bots_01": far_u01,
                "far_benign_01": far_b01, "far_unknown_bots_05": far_u05, "far_benign_05": far_b05}
            if meth in ("S3", "v1_weights"):
                lab = "AttribIQ" if meth == "S3" else "Original weights"
                row[f"{lab}: top-1"] = round(top1, 3)
                row[f"{lab}: AUROC vs benign"] = round(au_b, 3)
                row[f"{lab}: AUROC vs unknown-family bots"] = round(au_u, 3)
                row[f"{lab}: recall@FAR1%"] = round(rec01, 3)
                row[f"{lab}: false attr. unknown bots@FAR1%"] = round(far_u01, 3)
                row[f"{lab}: false attr. benign@FAR1%"] = round(far_b01, 4)
        # calibrated tau=0.5 decision
        row["AttribIQ p>=0.5: known attributed"] = round((known["decision"] == known["truth"]).mean(), 3)
        row["AttribIQ p>=0.5: benign attributed"] = round((benign["decision"] != "unknown").mean(), 4)
        NUM.setdefault("rq2_calibrated", {})[str(m)] = {
            "known_correct": float((known["decision"] == known["truth"]).mean()),
            "unknown_bot_attributed": float((unk["decision"] != "unknown").mean()),
            "benign_attributed": float((benign["decision"] != "unknown").mean()),
            "mean_p_known": float(known["p_top"].mean()), "mean_p_benign": float(benign["p_top"].mean())}
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "table_rq2.csv", index=False)
    NUM["rq2_counts"] = {"known": int(t["n_known"].iloc[0]), "unknown_bots": int(t["n_unknown_bot"].iloc[0]),
                         "benign": int(t["n_benign"].iloc[0])}
    NUM["rq2_counts_by_m"] = {str(r.m): {"known": int(r.n_known), "unknown_bots": int(r.n_unknown_bot),
                                         "benign": int(r.n_benign)} for r in t.itertuples()}
    # per-family top-1 and query families
    fam_tab = d[d.kind.str.startswith("bot")].groupby(["m", "fam_q", "kind"]).agg(
        n=("query", "size"), top_family=("S3_top_family", lambda s: s.value_counts().index[0])).reset_index()
    fam_tab.to_csv(out / "tables" / "table_rq2_families.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(FR))
    for i, (meth, lab) in enumerate((("S3", "AttribIQ (fused)"), ("v1_weights", "Original weights"),
                                     ("E_flow", "Flow mix only"), ("E_inf", "Shared rare infra. only"))):
        au = [NUM["rq2"][meth][str(m)]["auroc_vs_benign"] for m in FR]
        rc = [NUM["rq2"][meth][str(m)]["recall_far01"] for m in FR]
        axes[0].plot(x, au, color=SERIES[i], marker=MARKERS[i], markersize=5, linewidth=1.6, label=lab)
        axes[1].plot(x, rc, color=SERIES[i], marker=MARKERS[i], markersize=5, linewidth=1.6, label=lab)
    axes[0].axhline(0.5, color=INK2, linestyle=":", linewidth=1.0)
    for ax, title, yl in ((axes[0], "(a) Known-family bots vs benign hosts", "AUROC of top score"),
                          (axes[1], "(b) Correct attribution at 1% false-attribution rate", "Recall")):
        ax.set_xticks(x, FR_LABEL)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        style(ax, "Malicious share of infected host's traffic", yl)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig3_rq2_open_set.png", dpi=300)
    plt.close(fig)
    return t


def rq3(args, out):
    p = pd.read_parquet(Path(args.results) / "rq1_scored_pairs.parquet")
    rows = []
    for m in FR:
        g = p[p.m == m]
        for col, lab in (("p_attribiq", "AttribIQ (calibrated)"), ("p_logreg", "Learned logistic"),
                         ("v1_weights", "Original weights (score read as probability)")):
            prob = g[col].clip(0, 1).to_numpy()
            rows.append({"m": m, "method": lab, "Brier": M.brier(g.y, prob), "ECE": M.ece(g.y, prob),
                         "mean_p_pos": float(prob[g.y == 1].mean()), "mean_p_neg": float(prob[g.y == 0].mean())})
            NUM.setdefault("rq3_calibration", {}).setdefault(col, {})[str(m)] = rows[-1]
    t = pd.DataFrame(rows)
    t.to_csv(out / "tables" / "table_rq3_calibration.csv", index=False)

    # reliability diagrams at 100% and 20%
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, m in zip(axes, (1.0, 0.2)):
        g = p[p.m == m]
        ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1.0)
        for i, (col, lab) in enumerate((("p_attribiq", "AttribIQ (calibrated)"), ("p_logreg", "Learned logistic"),
                                        ("v1_weights", "Original weights"))):
            rel = M.reliability(g.y, g[col].clip(0, 1), n_bins=10)
            xs = [r[3] for r in rel if r[2] >= 20]
            ys = [r[4] for r in rel if r[2] >= 20]
            ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], markersize=5, linewidth=1.6, label=lab)
        ax.set_title(f"({'a' if m == 1.0 else 'b'}) malicious share {int(m * 100)}%", fontsize=9, color=INK, loc="left")
        style(ax, "Predicted probability", "Observed frequency" if m == 1.0 else None)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig4_rq3_reliability.png", dpi=300)
    plt.close(fig)

    for m in FR:
        g = p[p.m == m]
        rel = M.reliability(g.y, g["p_attribiq"].clip(0, 1), n_bins=10)
        NUM.setdefault("rq3_reliability", {})[str(m)] = {
            "bins": [{"pred": r[3], "obs": r[4], "n": int(r[2])} for r in rel if r[2] > 0],
            "share_below_0.1": float((g["p_attribiq"] < 0.1).mean()),
            "n_pairs": int(len(g))}

    # worked decomposition example: scenario 12 without blending, seed 147.32.84.165
    ex = p[(p.m == 1.0) & (p.rep == 0) & (p.sid == 12) & (p.a == "147.32.84.165")].sort_values("S", ascending=False)
    if len(ex):
        cols = ["b", "y", "S", "p_attribiq", "F_E_flow", "F_E_stat", "F_E_inf", "F_E_tmp", "E_tmp"]
        sel = pd.concat([ex.head(3), ex[ex.y == 1]]).drop_duplicates(subset=["b"])
        NUM["example_sc12"] = sel[cols].round(4).to_dict(orient="records")
        NUM["example_sc12_rank_of_true"] = [int(i) + 1 for i, y in enumerate(ex["y"].to_numpy()) if y == 1]

    # explanations: which lines independently support each correct top-k attribution
    F = ["F_E_flow", "F_E_stat", "F_E_inf", "F_E_tmp"]
    support, robust = [], []
    for (m, rep, sid, seed), g in p.groupby(["m", "rep", "sid", "a"]):
        k = int(g.y.sum())
        top = g.nlargest(k, "S")
        hits = top[top.y == 1]
        for _, h in hits.iterrows():
            vals = h[F].astype(float)
            row = {"m": m, "n_support": int((vals >= 0.95).sum())}
            row.update({c.replace("F_", ""): bool(vals[c] >= 0.95) for c in F})
            support.append(row)
        for drop in F:
            keep = [c for c in F if c != drop]
            s2 = g[keep].mean(axis=1, skipna=True)
            top2 = g.loc[s2.nlargest(k).index]
            kept = len(set(hits.index) & set(top2[top2.y == 1].index)) / max(len(hits), 1)
            robust.append({"m": m, "dropped": drop.replace("F_", ""), "kept": kept, "n_hits": len(hits)})
    e = pd.DataFrame(support)
    share = e.groupby("m")[["E_flow", "E_stat", "E_inf", "E_tmp"]].mean().reindex(FR)
    share["two_or_more_lines"] = e.assign(t=e.n_support >= 2).groupby("m")["t"].mean().reindex(FR)
    share["hits"] = e.groupby("m").size().reindex(FR)
    share.to_csv(out / "tables" / "table_rq3_support.csv")
    NUM["rq3_support"] = share.round(3).to_dict(orient="index")
    rb = pd.DataFrame(robust)
    rbt = rb[rb.n_hits > 0].groupby(["m", "dropped"])["kept"].mean().unstack().reindex(FR)
    rbt.to_csv(out / "tables" / "table_rq3_single_line_robustness.csv")
    NUM["rq3_robustness"] = rbt.round(3).to_dict(orient="index")

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    order = ["E_flow", "E_stat", "E_inf", "E_tmp"]
    w = 0.18
    xx = np.arange(len(FR))
    for i, c in enumerate(order):
        ax.bar(xx + (i - 1.5) * w, share[c].to_numpy(), width=w - 0.02, color=SERIES[i],
               label={"E_flow": "Flow-type mix", "E_stat": "Statistics", "E_inf": "Shared rare infra.",
                      "E_tmp": "Co-activity"}[c], linewidth=0)
    ax.plot(xx, share["two_or_more_lines"].to_numpy(), color=INK, marker="o", markersize=4,
            linewidth=1.2, label="Supported by two or more lines")
    ax.set_xticks(xx, FR_LABEL)
    ax.set_ylim(0, 1.05)
    style(ax, "Malicious share of infected host's traffic", "Share of correct top-k attributions")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig5_rq3_evidence_support.png", dpi=300)
    plt.close(fig)
    return t


def rq4(args, out):
    p = pd.read_parquet(Path(args.results) / "rq1_scored_pairs.parquet")
    F = ["F_E_flow", "F_E_stat", "F_E_inf", "F_E_tmp"]
    variants = {"AttribIQ (all four lines)": F}
    for drop in F:
        variants[f"without {drop.replace('F_', '')}"] = [c for c in F if c != drop]
    for v in ("inf_dot2", "inf_cos", "inf_noidf", "inf_ext", "inf_pooled"):
        variants[f"E_inf as {v}"] = ["F_E_flow", "F_E_stat", "F_" + v, "F_E_tmp"]
    X = p[F].to_numpy(dtype=float)
    k = (~np.isnan(X)).sum(axis=1)
    pv = np.clip(1.0 - X, 1e-4, 1.0)
    aggregations = {
        "aggregation: median of percentiles": np.nanmedian(X, axis=1),
        "aggregation: mean without lowest line": (np.nansum(X, axis=1) - np.nanmin(X, axis=1)) / (k - 1),
        "aggregation: Fisher combination": np.nansum(-2 * np.log(pv), axis=1),
        "aggregation: maximum percentile": np.nanmax(X, axis=1),
    }
    rows, unit_ap = [], {}
    scores = {name: s for name, s in aggregations.items()}
    scores.update({name: p[cols].mean(axis=1, skipna=True) for name, cols in variants.items()})
    for name, s in scores.items():
        q = p.assign(score=np.asarray(s, dtype=float))
        for m in FR:
            aps = np.array([M.ap(g.y, g.score) for _, g in q[q.m == m].groupby(["rep", "sid", "a"])])
            unit_ap[(name, m)] = aps
            mean, lo, hi = ci_row(aps)
            rows.append({"variant": name, "m": m, "AP": mean, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    # paired Wilcoxon of every variant against the method, Holm-corrected within each level
    trows = []
    for m in FR:
        base = unit_ap[("AttribIQ (all four lines)", m)]
        pv, eff = {}, {}
        for name in scores:
            if name == "AttribIQ (all four lines)":
                continue
            d = unit_ap[(name, m)] - base
            pv[name] = float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            eff[name] = float(d.mean())
        adj = holm(pv)
        for name in pv:
            trows.append({"variant": name, "m": m, "delta_AP": round(eff[name], 4), "p_wilcoxon": pv[name], "p_holm": adj[name]})
            NUM.setdefault("rq4_ablation_tests", {}).setdefault(name, {})[str(m)] = {"delta": eff[name], "p_holm": adj[name]}
    pd.DataFrame(trows).to_csv(out / "tables" / "table_rq4_ablation_tests.csv", index=False)
    piv = t.pivot(index="variant", columns="m", values="AP")[FR].round(3)
    piv.to_csv(out / "tables" / "table_rq4_ablation.csv")
    NUM["rq4_ablation"] = piv.to_dict(orient="index")

    # sensitivity runs
    sens_path = Path(args.results) / "rq4_sensitivity.parquet"
    if sens_path.exists():
        s = pd.read_parquet(sens_path)
        base = p[(p.m.isin([1.0, 0.2])) & (p.rep < 3)].copy()
        rows = []
        for (setting, value, m), g in s.groupby(["setting", "value", "m"]):
            gg = g.dropna(subset=["y"]) if "y" in g else g.iloc[0:0]
            aps = [M.ap(h.y, h.S) for _, h in gg.groupby(["rep", "sid", "a"]) if h.y.sum() > 0]
            aps_v1 = [M.ap(h.y, h.v1_weights) for _, h in gg.groupby(["rep", "sid", "a"]) if h.y.sum() > 0]
            n_bots = (gg.groupby(["rep", "sid"])["a"].nunique().groupby(level=0).sum()
                      .reindex(sorted(g["rep"].unique()), fill_value=0).mean())
            rows.append({"setting": setting, "value": value, "m": m,
                         "AP": float(np.mean(aps)) if aps else float("nan"),
                         "AP_v1": float(np.mean(aps_v1)) if aps_v1 else float("nan"),
                         "seed_hosts": n_bots,
                         "eligible_infected": float(g.groupby("rep")["eligible_infected"].first().mean())})
        for m in (1.0, 0.2):
            g = base[base.m == m]
            aps = [M.ap(h.y, h.S) for _, h in g.groupby(["rep", "sid", "a"])]
            aps_v1 = [M.ap(h.y, h.v1_weights) for _, h in g.groupby(["rep", "sid", "a"])]
            n_bots = g.groupby(["rep", "sid"])["a"].nunique().groupby(level=0).sum().mean()
            for setting, value in (("min_flows", 10), ("bin_seconds", 60), ("minutes", "full")):
                rows.append({"setting": setting, "value": value, "m": m, "AP": float(np.mean(aps)),
                             "AP_v1": float(np.mean(aps_v1)), "seed_hosts": n_bots, "eligible_infected": n_bots})
        st = pd.DataFrame(rows)
        st["value"] = st["value"].astype(str)
        st.to_csv(out / "tables" / "table_rq4_sensitivity.csv", index=False)
        NUM["rq4_sensitivity"] = st.round(3).to_dict(orient="records")

    rep_path = Path(args.results) / "rq4_v1_replication.csv"
    if rep_path.exists():
        r = pd.read_csv(rep_path)
        NUM["v1_replication"] = r.round(4).to_dict(orient="records")
        r.to_csv(out / "tables" / "table_v1_replication.csv", index=False)

    # cost
    tfiles = list(Path(args.results).glob("timings_*.csv"))
    if tfiles:
        tm = pd.concat([pd.read_csv(f) for f in tfiles], ignore_index=True)
        cost = tm.groupby("stage")["seconds"].agg(["count", "mean", "median", "max", "sum"]).round(2)
        cost.to_csv(out / "tables" / "table_cost.csv")
        NUM["cost"] = cost.to_dict(orient="index")
        prof = tm[tm.stage == "profiles"].groupby("sid")["seconds"].mean().round(2)
        prof.index = prof.index.astype(int)
        NUM["cost_profiles_per_scenario"] = prof.to_dict()
    envs = list(Path(args.results).glob("environment_*.json"))
    NUM["environment"] = [{**json.load(open(e)), "file": e.stem} for e in sorted(envs)]
    return t


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f2ee"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=INK2, linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.6, color=INK, wrap=True)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=INK2, linewidth=0.9))
    box(1, 15, 12, 10, "Bidirectional\nflow records\n(one incident)")
    box(17, 15, 13, 10, "Host profiles\n(>= 10 flows):\nflow types, 16 stats,\ndestinations, series")
    box(34, 29, 16, 7, "E_flow  flow-type mix", "#e8f0fb")
    box(34, 21, 16, 7, "E_stat  statistics", "#e8f0fb")
    box(34, 13, 16, 7, "E_inf  shared rare infra.", "#e8f0fb")
    box(34, 5, 16, 7, "E_tmp  co-activity\n(same incident only)", "#e8f0fb")
    box(54, 15, 14, 10, "Empirical CDF\nvs. all host pairs\n(label-free)")
    box(72, 15, 12, 10, "Mean of available\npercentiles = S\n+ 2-parameter\ncalibration")
    box(87, 25, 12, 9, "T1: rank incident\nhosts for a seed")
    box(87, 14, 12, 9, "T2: link to library\nor abstain")
    box(87, 3, 12, 9, "Exact per-line\nlog-odds\ncontributions")
    arrow(13, 20, 17, 20)
    for y in (32.5, 24.5, 16.5, 8.5):
        arrow(30, 20, 34, y)
        arrow(50, y, 54, 20)
    arrow(68, 20, 72, 20)
    for y in (29.5, 18.5, 7.5):
        arrow(84, 20, 87, y)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig1_pipeline.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/cache")
    args = ap.parse_args()
    out = Path(args.results) / "report"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig_pipeline(out)
    table_dataset(args, out)
    rq1(args, out)
    rq2(args, out)
    rq3(args, out)
    rq4(args, out)
    calib = Path(args.results) / "rq1_calibrators.json"
    if calib.exists():
        NUM["calibrators"] = json.load(open(calib))
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x
    json.dump(clean(NUM), open(out / "numbers.json", "w"), indent=1, default=float, allow_nan=False)
    print("report written to", out)


if __name__ == "__main__":
    main()
