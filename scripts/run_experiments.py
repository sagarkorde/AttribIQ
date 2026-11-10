#!/usr/bin/env python3
"""Run every experiment of the paper and write raw results to results/.

Usage:
    python scripts/run_experiments.py --raw data/raw --cache data/cache --out results

Stages (each is skipped when its output already exists, so the script resumes):
  main      blending conditions x repetitions -> RQ1 seed pairs and RQ2 query-library pairs
  score     leave-one-family-out calibration, metrics, open-set decisions
  extras    RQ4 sensitivity runs (activity threshold, time bin, observation window)
  replicate original AttribIQ clustering statistics on Scenarios 10 and 12 (all sources and internal only)
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribiq import experiments as X  # noqa: E402


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--stages", default="main,score,extras,replicate")
    ap.add_argument("--force", action="store_true", help="recompute stages even if outputs exist")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    cfg = X.Config(raw_dir=args.raw, cache_dir=args.cache, out_dir=args.out, reps=args.reps)
    t_start = time.perf_counter()
    F, R = X.load_all(cfg)

    if "main" in stages:
        for m in cfg.fractions:
            for rep in range(cfg.reps):
                f1 = out / "raw" / f"rq1_m{m:.2f}_r{rep}.parquet"
                f2 = out / "raw" / f"rq2_m{m:.2f}_r{rep}.parquet"
                if f1.exists() and f2.exists() and not args.force:
                    continue
                t = time.perf_counter()
                P, roles, flows = X.condition(cfg, F, R, m, rep)
                p1 = X.rq1_pairs(cfg, P, roles, flows, m, rep)
                t1 = time.perf_counter()
                p2 = X.rq2_pairs(cfg, P, roles, m, rep)
                cfg.timings.append({"stage": "evidence_rq2", "m": m, "rep": rep,
                                    "seconds": time.perf_counter() - t1, "pairs": len(p2)})
                p1.to_parquet(f1, index=False)
                p2.to_parquet(f2, index=False)
                print(f"m={m:.2f} rep={rep} rq1={len(p1)} rq2={len(p2)} "
                      f"{time.perf_counter() - t:.0f}s rss={peak_rss_mb():.0f}MB", flush=True)

    if "score" in stages:
        p1 = pd.concat([pd.read_parquet(f) for f in sorted((out / "raw").glob("rq1_m*.parquet"))],
                       ignore_index=True)
        p1 = X.rq1_score(cfg, p1)
        p1.to_parquet(out / "rq1_scored_pairs.parquet", index=False)
        json.dump({str(k): v for k, v in p1.attrs.get("calibrators", {}).items()},
                  open(out / "rq1_calibrators.json", "w"), indent=2)
        X.rq1_metrics(p1).to_csv(out / "rq1_metrics.csv", index=False)
        p2 = pd.concat([pd.read_parquet(f) for f in sorted((out / "raw").glob("rq2_m*.parquet"))],
                       ignore_index=True)
        d = X.rq2_decide(cfg, p2, p1)
        d.to_csv(out / "rq2_decisions.csv", index=False)
        print("scored", flush=True)

    if "extras" in stages:
        rows = []
        f_ex = out / "rq4_sensitivity.parquet"
        if args.force or not f_ex.exists():
            for m in (1.0, 0.2):
                for rep in range(3):
                    for kind, values in (("min_flows", (5, 20, 50)), ("bin_seconds", (10, 300)),
                                         ("minutes", (5, 15, 30, 60))):
                        for v in values:
                            kw = {kind: v}
                            P, roles, flows = X.condition(cfg, F, R, m, rep, **kw)
                            p1 = X.rq1_pairs(cfg, P, roles, flows, m, rep,
                                             with_variants=False, with_v1=False)
                            elig = {s_: int((roles[s_] == "bot").sum()) for s_ in cfg.multi}
                            if p1.empty:
                                p1 = pd.DataFrame({"m": [m], "rep": [rep]})
                            p1["setting"], p1["value"] = kind, v
                            p1["eligible_infected"] = sum(elig.values())
                            rows.append(p1)
                            print(f"extras {kind}={v} m={m} rep={rep}", flush=True)
            pd.concat(rows, ignore_index=True).to_parquet(f_ex, index=False)

    if "replicate" in stages:
        f_rep = out / "rq4_v1_replication.csv"
        if args.force or not f_rep.exists():
            rows = []
            for sid in (10, 12):
                for all_src in (True, False):
                    t = time.perf_counter()
                    rows += X.v1_replication(cfg, sid, all_sources=all_src)
                    print(f"replicated sc{sid} all_sources={all_src} {time.perf_counter() - t:.0f}s",
                          flush=True)
            pd.DataFrame(rows).to_csv(f_rep, index=False)

    timings = pd.DataFrame(cfg.timings)
    timings.to_csv(out / f"timings_{'_'.join(sorted(stages))}.csv", index=False)
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "processor": platform.processor(), "cpu_count": __import__("os").cpu_count(),
           "peak_rss_mb": peak_rss_mb(), "wall_seconds": time.perf_counter() - t_start,
           "numpy": np.__version__, "pandas": pd.__version__,
           "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}
    json.dump(env, open(out / f"environment_{'_'.join(sorted(stages))}.json", "w"), indent=2)
    print("done", json.dumps(env), flush=True)


if __name__ == "__main__":
    main()
