import numpy as np
import pandas as pd

from attribiq.data import host_roles
from attribiq.graph import bipartite, rare_destination_candidates, rwr
from attribiq.stress import blend, blend_session


def test_session_blend_keeps_draws_and_moves_malicious_flows_into_user_minutes(toy_flows):
    roles = host_roles(toy_flows)
    a, ra, ma = blend(toy_flows, roles, 0.5, np.random.default_rng(1), min_benign_flows=50)
    b, rb, mb = blend_session(toy_flows, roles, 0.5, np.random.default_rng(1), min_benign_flows=50)
    assert ma == mb and len(a) == len(b)                       # same partners and subsampling
    t0 = b["ts"].min()
    for bot in mb:
        f = b[b["src"] == bot]
        user_min = set(np.floor((f.loc[f["role"] != "bot", "ts"] - t0) / 60))
        mal_min = set(np.floor((f.loc[f["role"] == "bot", "ts"] - t0) / 60))
        assert mal_min <= user_min                              # malware only active with the user


def test_random_walk_ranks_co_campaign_host_first(toy_flows):
    hosts = sorted(toy_flows["src"].unique())
    B = bipartite(toy_flows, hosts)
    X = rwr(B, [hosts.index("147.32.84.165")], restart=0.3)
    scores = pd.Series(X[:, 0], index=hosts).drop("147.32.84.165")
    assert scores.idxmax() == "147.32.84.191"


def test_rare_destination_pruning_finds_shared_c2(toy_flows):
    hosts = sorted(toy_flows["src"].unique())
    cand, score = rare_destination_candidates(bipartite(toy_flows, hosts), k=2, cap=2)
    i, j = hosts.index("147.32.84.165"), hosts.index("147.32.84.191")
    assert j in cand[i] and i in cand[j]


def test_conformal_threshold_controls_false_attribution():
    from attribiq.conformal import conformal_threshold, min_labelled
    rng = np.random.default_rng(0)
    assert min_labelled(0.01) == 99 and np.isinf(conformal_threshold(rng.random(98), 0.01))
    far = []
    for _ in range(2000):
        cal, test = rng.random(200), rng.random(500)
        far.append((test > conformal_threshold(cal, 0.05)).mean())
    assert np.mean(far) <= 0.05 + 0.005           # marginal guarantee (exchangeable scores)
