# AttribIQ

Calibrated, decomposable campaign attribution from bidirectional network flows, evaluated
when malicious traffic is only a minority of an infected host's traffic.

Given flow records of a confirmed incident, AttribIQ answers two analyst questions:

* **T1, seeded expansion:** given one confirmed infected host, rank all other hosts of the
  incident so that members of the same campaign come first.
* **T2, open-set linkage:** link a host to a malware family in a library of earlier
  incidents, or answer *unknown* for families absent from the library and for benign hosts.

For every pair of hosts it computes four evidence lines (flow-type mix, 16 behavioural
statistics, shared rare destinations, temporal co-activity), maps each line to a label-free
empirical percentile over all host pairs, averages the available lines, calibrates the result
with two parameters fitted on other malware families, and decomposes every score exactly
into additive per-line log-odds contributions. A threshold set on confirmed non-infected
hosts fixes the false-attribution rate of T2.

This repository holds the code, data checksums, tests and every result behind the paper
(the manuscript itself is not included).

## Repository layout

```
attribiq/            library
  scenarios.py       CTU-13 scenario metadata and SHA-256 checksums
  data.py            reading, verification and caching of binetflow files
  profiles.py        host profiles (flow types, statistics, destinations, per-minute series)
  evidence.py        the four evidence lines
  fusion.py          label-free percentile fusion, calibration, exact decomposition
  combine.py         original weighted score and fusion helpers
  stress.py          blending stress test and observation-window truncation
  baselines.py       original AttribIQ pipeline, HDBSCAN co-membership
  metrics.py         AP, AUROC, Brier, ECE, bootstrap intervals
  experiments.py     T1 and T2 protocols with leave-one-family-out fitting
scripts/
  fetch_data.py      download and verify the six CTU-13 flow files
  run_experiments.py all experiments (stages main, score, extras, replicate)
  make_report.py     every table, figure and number of the paper
tests/               unit tests (pytest)
data/SHA256SUMS      checksums of the exact flow files used
results/             per-condition results, report/tables, report/figures, report/numbers.json
```

## Installation

Tested with Python 3.13.18 and the pinned package versions.

```bash
pip install -r requirements.txt
```

## Data

The experiments use CTU-13 scenarios 5, 7, 9, 10, 11 and 12 (García et al., *Computers &
Security* 45, 2014, doi:10.1016/j.cose.2014.05.011), distributed by the Stratosphere
Laboratory, Czech Technical University in Prague. The script tries the official repository
first and then public mirrors, and accepts a file only if its SHA-256 equals the value in
`attribiq/scenarios.py` (also listed in `data/SHA256SUMS`):

```bash
python scripts/fetch_data.py --out data/raw
python scripts/fetch_data.py --out data/raw --verify-only     # check existing files
```

Scenarios 9–12 have identical checksums across two independent mirrors (three for 12);
scenario 5 matches a second mirror record for record; scenario 7 comes from one mirror.

## Reproducing the paper

```bash
python scripts/run_experiments.py --raw data/raw --cache data/cache --out results
python scripts/make_report.py --results results --raw data/raw --cache data/cache
python -m pytest -q
```

The first command runs all stages and resumes if interrupted (`--force` recomputes). On a
2-vCPU virtual machine the main experiment (5 malicious shares × 5 repetitions × 6 scenarios,
all baselines) took 12 minutes with 2.1 GB peak memory;
the sensitivity and replication stages took about 13 minutes more. All seeds are fixed and
repeated runs give identical results. `results/report/numbers.json` holds every number
quoted in the paper.

## Main results

Data: 4,073,930 flows, 1,151 eligible hosts. The blending stress
test overlays each infected host on the traffic of a real benign host so that malicious
traffic makes up the stated share.

**T1, average precision** (mean over seeds and repetitions; 95% intervals in
`results/report/tables/table_rq1_ap.csv`):

| Method | 100% | 50% | 20% | 10% | 5% |
|---|---|---|---|---|---|
| AttribIQ (fused percentiles) | 0.960 | 0.704 | 0.329 | 0.246 | 0.203 |
| Original hand-set weights | 1.000 | 0.735 | 0.291 | 0.195 | 0.188 |
| Learned logistic regression | 0.993 | 0.498 | 0.158 | 0.127 | 0.107 |
| HDBSCAN co-membership | 0.986 | 0.490 | 0.223 | 0.169 | 0.172 |
| Original K-means pipeline | 0.421 | 0.045 | 0.040 | 0.040 | 0.038 |
| Activity rank | 0.298 | 0.081 | 0.067 | 0.066 | 0.082 |
| Random ranking | 0.028 | 0.029 | 0.029 | 0.029 | 0.029 |

Without blending every evidence-based method is near perfect. From 20% malicious share
downwards AttribIQ has the highest mean; at 20% and 10% it is significantly better than
every baseline (Wilcoxon signed-rank, Holm-corrected), at 5% the differences from the
original weights and HDBSCAN are not significant.

**Calibration (ECE of pair probabilities):** AttribIQ 0.028 | 0.019 | 0.016 | 0.021 | 0.023; original weights read as a
probability 0.105 | 0.115 | 0.113 | 0.120 | 0.118.

**T2, AttribIQ open-set linkage at a 1% false-attribution threshold:**

| | 100% | 50% | 20% | 10% | 5% |
|---|---|---|---|---|---|
| Top-1 family (known-family queries) | 1.000 | 0.923 | 0.692 | 0.492 | 0.646 |
| Recall at 1% false attribution | 0.917 | 0.138 | 0.000 | 0.031 | 0.031 |
| Unknown-family hosts attributed | 0.000 | 0.000 | 0.000 | 0.027 | 0.000 |
| Benign hosts attributed | 0.013 | 0.015 | 0.012 | 0.014 | 0.014 |

Cross-incident linkage works without blending and fails for every method under heavy
blending, where AttribIQ abstains rather than misattributes.

## Limitations

The evidence lines, parameters and aggregation rule were developed on the same six
scenarios (no held-out scenario); the captures are from 2011; only one family (Rbot)
occurs in two incidents, so T2 rests on 12–13 known-family queries per repetition; the
blending model overlays traffic uniformly. See the paper's threats-to-validity section.

## Licence

MIT (see `LICENSE`). CTU-13 is subject to the terms of its publishers.
