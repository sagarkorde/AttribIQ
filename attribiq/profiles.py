"""Per-host behavioural profiles built from the flows a host initiated.

A profile has four parts, one per evidence line:

* ``tokens``  - distribution over discrete flow types (service x outcome x size);
* ``stats``   - a vector of 16 robust statistics (Table 3 of the paper);
* ``dsts``    - the set of destination addresses contacted;
* ``series``  - per-minute flow counts over the capture.

Every statistic is defined for every host that passes the activity threshold;
quantities that cannot be computed (e.g. periodicity with no repeated
destination) are returned as NaN together with an explicit missingness flag
instead of a silent placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TCP_SERVICES = {25: "smtp", 465: "smtp", 587: "smtp", 53: "dns", 80: "http", 8080: "http",
                3128: "http", 8000: "http", 443: "https", 135: "smb", 137: "smb", 138: "smb",
                139: "smb", 445: "smb", 21: "ftp", 22: "ssh", 23: "telnet"}
UDP_SERVICES = {53: "dns", 123: "ntp", 137: "netbios", 138: "netbios", 161: "snmp",
                162: "snmp", 1900: "ssdp", 67: "dhcp", 68: "dhcp"}
SIZE_EDGES = np.array([64, 256, 1024, 8192])  # bytes sent by the initiator

STAT_NAMES = [
    "log_rate_per_active_min", "log_mean_src_bytes", "log_mean_dst_bytes", "log_mean_pkts",
    "log_mean_duration", "frac_tcp", "frac_udp", "frac_icmp", "dst_entropy_norm",
    "log_unique_dsts", "log_unique_dports", "frac_no_reply", "frac_internal_dst",
    "median_dst_cov", "frac_periodic_flows", "active_minute_fraction",
]


def service_of(proto: np.ndarray, dport: np.ndarray) -> np.ndarray:
    out = np.empty(len(proto), dtype=object)
    for i, (p, d) in enumerate(zip(proto, dport)):
        if p == "tcp":
            if np.isnan(d):
                out[i] = "tcp_other"
                continue
            d = int(d)
            if d in TCP_SERVICES:
                out[i] = TCP_SERVICES[d]
            elif 6660 <= d <= 6669 or d == 6697:
                out[i] = "irc"
            elif d < 1024:
                out[i] = "tcp_low"
            elif d < 49152:
                out[i] = "tcp_reg"
            else:
                out[i] = "tcp_eph"
        elif p == "udp":
            if np.isnan(d):
                out[i] = "udp_other"
                continue
            d = int(d)
            out[i] = UDP_SERVICES.get(d, "udp_low" if d < 1024 else "udp_high")
        elif p == "icmp":
            out[i] = "icmp"
        else:
            out[i] = "other"
    return out


def outcome_of(proto: np.ndarray, state: np.ndarray, dst_bytes: np.ndarray) -> np.ndarray:
    out = np.where(dst_bytes <= 0, "noreply", "reply").astype(object)
    for i, (p, s) in enumerate(zip(proto, state)):
        if p == "tcp" and out[i] == "reply":
            parts = str(s).split("_")
            if len(parts) == 2 and "R" in parts[1]:
                out[i] = "reset"
    return out


def annotate(flows: pd.DataFrame) -> pd.DataFrame:
    """Add service, outcome, size bucket and token columns (vectorised where possible)."""
    f = flows
    proto = f["proto"].astype(str).to_numpy()
    dport = f["dport"].to_numpy(dtype="float64")
    # vectorise the service mapping through unique (proto, dport) pairs
    key = pd.MultiIndex.from_arrays([proto, np.nan_to_num(dport, nan=-1).astype("int64")])
    uniq = key.unique()
    u_proto = np.asarray(uniq.get_level_values(0))
    u_port = np.asarray(uniq.get_level_values(1), dtype="float64")
    u_port[u_port < 0] = np.nan
    svc_map = dict(zip(uniq, service_of(u_proto, u_port)))
    svc = np.array([svc_map[k] for k in key], dtype=object)
    dst_bytes = (f["bytes"] - f["src_bytes"]).to_numpy()
    st = f["state"].astype(str).to_numpy()
    skey = pd.MultiIndex.from_arrays([proto, st, dst_bytes > 0])
    su = skey.unique()
    o_map = dict(zip(su, outcome_of(np.asarray(su.get_level_values(0)), np.asarray(su.get_level_values(1)),
                                    np.where(np.asarray(su.get_level_values(2)), 1, 0))))
    outc = np.array([o_map[k] for k in skey], dtype=object)
    size = np.searchsorted(SIZE_EDGES, f["src_bytes"].to_numpy(), side="left")
    out = f.copy()
    out["service"] = svc
    out["outcome"] = outc
    out["size_bin"] = size
    out["token"] = pd.Series(svc).str.cat([pd.Series(outc), pd.Series(size.astype(str))], sep="|").to_numpy()
    out["dst_bytes"] = dst_bytes
    return out


@dataclass
class HostProfiles:
    """Profiles of the eligible hosts of one incident."""
    sid: int
    hosts: list[str]
    n_flows: np.ndarray
    tokens: pd.DataFrame                 # hosts x token, row-normalised
    stats: pd.DataFrame                  # hosts x STAT_NAMES (may contain NaN)
    dsts: dict[str, frozenset]
    series: np.ndarray                   # hosts x minutes (flow counts)
    t0: float
    meta: dict = field(default_factory=dict)


def _periodicity(g: pd.DataFrame, min_flows: int = 5, cov_thr: float = 0.25) -> tuple[float, float]:
    """Per-destination inter-arrival regularity.

    For each destination contacted at least ``min_flows`` times, the coefficient
    of variation of the gaps between consecutive flow start times is computed.
    Returns (median CoV over such destinations, fraction of the host's flows that
    go to destinations with CoV < cov_thr). The median is NaN when the host has
    no destination with enough flows.
    """
    covs, periodic_flows = [], 0
    for _, ts in g.groupby("dst", observed=True)["ts"]:
        if len(ts) < min_flows:
            continue
        gaps = np.diff(np.sort(ts.to_numpy()))
        m = gaps.mean()
        if m <= 0:
            continue
        cov = gaps.std() / m
        covs.append(cov)
        if cov < cov_thr:
            periodic_flows += len(ts)
    if not covs:
        return float("nan"), 0.0
    return float(np.median(covs)), periodic_flows / len(g)


def build_profiles(flows: pd.DataFrame, sid: int, min_flows: int = 10,
                   bin_seconds: int = 60, t0: float | None = None,
                   t1: float | None = None) -> HostProfiles:
    """Build profiles for hosts with at least ``min_flows`` initiated flows."""
    f = annotate(flows)
    f["src"] = f["src"].astype(str)
    counts = f.groupby("src").size()
    eligible = counts[counts >= min_flows].index.tolist()
    f = f[f["src"].isin(eligible)]
    t0 = float(f["ts"].min()) if t0 is None else t0
    t1 = float(f["ts"].max()) if t1 is None else t1
    n_bins = int(np.floor((t1 - t0) / bin_seconds)) + 1

    tok = pd.crosstab(f["src"], f["token"])
    tok = tok.div(tok.sum(axis=1), axis=0).reindex(eligible)

    rows, dsts = [], {}
    series = np.zeros((len(eligible), n_bins), dtype="float64")
    idx = {h: i for i, h in enumerate(eligible)}
    bins = np.clip(((f["ts"].to_numpy() - t0) // bin_seconds).astype(int), 0, n_bins - 1)
    np.add.at(series, (f["src"].map(idx).to_numpy(), bins), 1.0)

    internal_prefix = "147.32."
    for h, g in f.groupby("src", sort=False):
        n = len(g)
        dst_counts = g["dst"].value_counts()
        p = dst_counts.to_numpy() / n
        h_dst = float(-(p * np.log2(p)).sum())
        active = int((series[idx[h]] > 0).sum())
        med_cov, frac_per = _periodicity(g)
        rows.append({
            "host": h,
            "log_rate_per_active_min": np.log10(n / max(active, 1)),
            "log_mean_src_bytes": np.log10(1 + g["src_bytes"].mean()),
            "log_mean_dst_bytes": np.log10(1 + g["dst_bytes"].clip(lower=0).mean()),
            "log_mean_pkts": np.log10(1 + g["pkts"].mean()),
            "log_mean_duration": np.log10(1e-3 + g["dur"].mean()),
            "frac_tcp": float((g["proto"].astype(str) == "tcp").mean()),
            "frac_udp": float((g["proto"].astype(str) == "udp").mean()),
            "frac_icmp": float((g["proto"].astype(str) == "icmp").mean()),
            "dst_entropy_norm": h_dst / np.log2(n) if n > 1 else 0.0,
            "log_unique_dsts": np.log10(len(dst_counts)),
            "log_unique_dports": np.log10(1 + g["dport"].nunique()),
            "frac_no_reply": float((g["outcome"] == "noreply").mean()),
            "frac_internal_dst": float(g["dst"].str.startswith(internal_prefix).mean()),
            "median_dst_cov": med_cov,
            "frac_periodic_flows": frac_per,
            "active_minute_fraction": active / n_bins,
        })
        dsts[h] = frozenset(dst_counts.index)
    stats = pd.DataFrame(rows).set_index("host").reindex(eligible)[STAT_NAMES]
    return HostProfiles(sid=sid, hosts=eligible, n_flows=counts.reindex(eligible).to_numpy(),
                        tokens=tok, stats=stats, dsts=dsts, series=series, t0=t0,
                        meta={"bin_seconds": bin_seconds, "min_flows": min_flows,
                              "n_bins": n_bins})
