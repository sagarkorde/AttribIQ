#!/usr/bin/env python3
"""Experiments added in the revision (run after run_experiments.py).

Stages
  heldout   open-set linkage over seven incidents incl. the held-out scenario 2
  graph     random-walk-with-restart baseline and rare-destination pruning ranks
  session   session-aware blending (malicious flows move into the user's active minutes)
  contam    smaller reference populations (fewer non-infected hosts)
  scale     cost of exhaustive versus pruned scoring for growing host counts

Usage:
  python scripts/run_revision.py --raw data/raw --cache data/cache --out results
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attribiq import revision as RV  # noqa: E402
from attribiq.data import read_binetflow  # noqa: E402
from attribiq.experiments import Config, load_all  # noqa: E402
from attribiq.scenarios import SCENARIOS, T2_EXTENDED_SCENARIOS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--stages", default="heldout,graph,session,contam,scale")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) / "revision"
    out.mkdir(parents=True, exist_ok=True)
    stages = args.stages.split(",")
    t_start = time.time()
    cfg = Config(raw_dir=args.raw, cache_dir=args.cache, out_dir=args.out)
    main_pairs = pd.read_parquet(Path(args.out) / "rq1_scored_pairs.parquet")

    def todo(name: str) -> bool:
        return args.force or not (out / name).exists()

    if "heldout" in stages and todo("t2_heldout_decisions.parquet"):
        c2 = replace(cfg, scenarios=T2_EXTENDED_SCENARIOS)
        F, R = load_all(c2)
        parts = []
        for m in c2.fractions:
            for rep in range(c2.reps):
                t = time.perf_counter()
                parts.append(RV.heldout_t2(c2, F, R, m, rep, main_pairs))
                print(f"heldout m={m} rep={rep} {time.perf_counter() - t:.0f}s", flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(out / "t2_heldout_decisions.parquet")
        del F, R

    if {"graph", "session", "contam"} & set(stages):
        cm = replace(cfg, scenarios=cfg.multi)
        F, R = load_all(cm)
        if "graph" in stages and todo("t1_graph_pruning.parquet"):
            parts = []
            for m in cm.fractions:
                for rep in range(cm.reps):
                    t = time.perf_counter()
                    parts.append(RV.graph_and_pruning(cm, F, R, m, rep))
                    print(f"graph m={m} rep={rep} {time.perf_counter() - t:.0f}s", flush=True)
            pd.concat(parts, ignore_index=True).to_parquet(out / "t1_graph_pruning.parquet")
        if "session" in stages and todo("t1_session_pairs.parquet"):
            parts = []
            for m in (0.5, 0.2, 0.1):
                for rep in range(cm.reps):
                    t = time.perf_counter()
                    parts.append(RV.session_pairs(cm, F, R, m, rep))
                    print(f"session m={m} rep={rep} {time.perf_counter() - t:.0f}s", flush=True)
            pairs = RV.frozen_scores(pd.concat(parts, ignore_index=True), main_pairs)
            pairs.to_parquet(out / "t1_session_pairs.parquet")
        if "contam" in stages and todo("t1_contamination_pairs.parquet"):
            parts = []
            for m in (1.0, 0.2):
                for rep in range(3):
                    for keep in (1.0, 0.5, 0.25, 0.1):
                        t = time.perf_counter()
                        parts.append(RV.contamination_pairs(cm, F, R, m, rep, keep))
                        print(f"contam m={m} rep={rep} keep={keep} {time.perf_counter() - t:.0f}s", flush=True)
            pairs = RV.frozen_scores(pd.concat(parts, ignore_index=True), main_pairs)
            pairs.to_parquet(out / "t1_contamination_pairs.parquet")
        del F, R

    if "scale" in stages and todo("scaling.csv"):
        # A synthetic large network for timing only: all source addresses of the seven
        # captures, overlaid on a common start time and kept distinct per capture.
        parts = []
        for sid in T2_EXTENDED_SCENARIOS:
            f = read_binetflow(Path(args.raw) / SCENARIOS[sid].filename)
            f["ts"] = f["ts"] - f["ts"].min()
            f["src"] = str(sid) + ":" + f["src"].astype(str)
            parts.append(f)
        flows = pd.concat(parts, ignore_index=True)
        del parts
        rows = RV.scaling(cfg, flows, [250, 500, 1000, 2000, 4000, 8000, 16000])
        pd.DataFrame(rows).to_csv(out / "scaling.csv", index=False)
        print(pd.DataFrame(rows).to_string(), flush=True)

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "wall_seconds": round(time.time() - t_start, 1),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
           "stages": stages}
    json.dump(env, open(out / f"environment_{'_'.join(stages)}.json", "w"), indent=1)
    print("done", env, flush=True)


if __name__ == "__main__":
    main()
