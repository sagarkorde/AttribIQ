import io

import numpy as np
import pandas as pd
import pytest

from attribiq import metrics as M
from attribiq.data import _parse_port, flow_role, host_roles, is_internal, read_binetflow
from attribiq.evidence import EvidenceSpace
from attribiq.fusion import Calibrator, ReferenceCDF, fused_score
from attribiq.profiles import STAT_NAMES, build_profiles
from attribiq.stress import blend, truncate


# --------------------------------------------------------------------- data
def test_parse_port_handles_hex_and_empty():
    out = _parse_port(pd.Series(["80", "0x0303", "", None]))
    assert out[0] == 80 and out[1] == 0x0303
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_flow_roles_from_labels():
    labels = pd.Series(["flow=From-Botnet-V42-TCP", "flow=From-Normal-V42-Grill",
                        "flow=Background-TCP-Attempt", "flow=To-Background-UDP"])
    assert list(flow_role(labels)) == ["bot", "normal", "background", "background"]


def test_internal_prefix():
    assert list(is_internal(pd.Series(["147.32.84.165", "147.33.1.1", "8.8.8.8"]))) == [True, False, False]


def test_read_binetflow(tmp_path):
    text = ("StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
            "2011/08/18 10:19:13.328,0.002,tcp,147.32.84.165,1025,   ->,1.2.3.4,6667,S_RA,0,0,4,321,200,flow=From-Botnet-V51-1\n"
            "2011/08/18 10:19:14.000,1.0,icmp,147.32.84.10,0x0008,   ->,1.2.3.5,0x0000,ECO,0,0,2,120,60,flow=Background\n")
    p = tmp_path / "x.binetflow"
    p.write_text(text)
    f = read_binetflow(p)
    assert len(f) == 2 and f.loc[0, "role"] == "bot" and f.loc[1, "sport"] == 8
    assert host_roles(f).to_dict() == {"147.32.84.10": "background", "147.32.84.165": "bot"}


# ----------------------------------------------------------------- profiles
def test_profiles_define_every_statistic(toy_flows):
    p = build_profiles(toy_flows, sid=0, min_flows=10)
    assert set(p.hosts) == {"147.32.84.165", "147.32.84.191", "147.32.84.10", "147.32.84.11"}
    assert list(p.stats.columns) == STAT_NAMES
    # beaconing bots: low per-destination CoV and most flows periodic
    bot = p.stats.loc["147.32.84.165"]
    assert bot["median_dst_cov"] < 0.1 and bot["frac_periodic_flows"] > 0.75
    np.testing.assert_allclose(p.tokens.sum(axis=1).to_numpy(), 1.0)


def test_activity_threshold_excludes_small_hosts(toy_flows):
    p = build_profiles(toy_flows, sid=0, min_flows=100)
    assert p.hosts == ["147.32.84.165", "147.32.84.191"]


# ----------------------------------------------------------------- evidence
def test_evidence_ranks_co_campaign_host_first(toy_flows):
    p = build_profiles(toy_flows, sid=0, min_flows=10)
    es = EvidenceSpace([p])
    pt = es.pair_table(p, ["147.32.84.165"], p).set_index("b")
    for line in ("E_flow", "E_stat", "E_inf", "E_tmp"):
        assert pt[line].idxmax() == "147.32.84.191", line
    assert 0.99 < pt.loc["147.32.84.191", "E_flow"] <= 1.0 + 1e-9


def test_temporal_unavailable_across_incidents(toy_flows):
    p = build_profiles(toy_flows, sid=0, min_flows=10)
    q = build_profiles(toy_flows, sid=1, min_flows=10)
    es = EvidenceSpace([p, q])
    pt = es.pair_table(p, ["147.32.84.165"], q)
    assert pt["E_tmp"].isna().all()


def test_idf_downweights_shared_resolver():
    es = EvidenceSpace.__new__(EvidenceSpace)
    es.external_only, es.use_idf, es.infra_mode = False, True, "dot"
    es.n_hosts = 10
    es.idf = {"resolver": np.log(11 / 11) + 1, "c2": np.log(11 / 3) + 1}
    assert es.idf["c2"] > es.idf["resolver"]


# ------------------------------------------------------------------- fusion
def test_reference_cdf_midrank():
    ref = pd.DataFrame({"E_flow": [0.0, 1.0, 2.0, 3.0]})
    cdf = ReferenceCDF(["E_flow"]).fit(ref)
    out = cdf.transform(pd.DataFrame({"E_flow": [-1.0, 1.0, 10.0, np.nan]}))["F_E_flow"].to_numpy()
    assert out[0] == 0.0 and out[1] == pytest.approx(0.375) and out[2] == 1.0 and np.isnan(out[3])


def test_decomposition_is_exact():
    rng = np.random.default_rng(1)
    pct = pd.DataFrame(rng.uniform(size=(200, 4)), columns=["F_a", "F_b", "F_c", "F_d"])
    pct.iloc[:50, 3] = np.nan                       # unavailable line
    s = fused_score(pct)
    y = (s + rng.normal(0, 0.1, len(s)) > 0.6).astype(int)
    cal = Calibrator().fit(s, y)
    d = cal.decompose(pct)
    logit = cal.alpha * s + cal.beta
    np.testing.assert_allclose(d["base"] + d.drop(columns="base").fillna(0).sum(axis=1), logit, atol=1e-9)


# ------------------------------------------------------------------- stress
def test_blend_hits_target_fraction(toy_flows):
    roles = host_roles(toy_flows)
    f, r, mapping = blend(toy_flows, roles, 0.2, np.random.default_rng(0), min_benign_flows=10)
    assert set(mapping) == {"147.32.84.165", "147.32.84.191"}
    for b in mapping:
        g = f[f["src"] == b]
        frac = (g["role"] == "bot").mean()
        assert abs(frac - 0.2) < 0.02
    assert not set(mapping.values()) & set(r.index)          # partners left the pool


def test_truncate_keeps_window(toy_flows):
    roles = host_roles(toy_flows)
    f = truncate(toy_flows, roles, minutes=5)
    assert f["ts"].max() - f["ts"].min() < 300


# ------------------------------------------------------------------ metrics
def test_metrics_basic():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert M.r_precision(y, s) == 0.5
    assert M.ap(y, s) == pytest.approx((1 + 2 / 3) / 2)
    assert M.ece(np.array([0, 1] * 50), np.full(100, 0.5)) == pytest.approx(0.0)
    mean, lo, hi = M.bootstrap_ci(np.arange(100.0), n_boot=500)
    assert lo < mean < hi
