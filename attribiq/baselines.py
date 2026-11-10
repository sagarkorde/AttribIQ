"""Baselines: activity ranking, the original AttribIQ pipeline, density clustering.

``v1_fingerprints`` re-implements the 19-feature source-IP x hour fingerprint
of the original AttribIQ paper. Two quantities were not defined there and are
fixed here explicitly: the burst score (share of the hour's flows that fall in
its busiest minute) and the value of undefined statistics for fingerprints
with fewer than three flows (beacon CoV = 1.0 and payload asymmetry computed
with +1 in the denominator), which reproduces the placeholder behaviour
reported in that paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

IOT_PORTS = {23, 2323, 7547, 8080, 8443, 5555, 48101, 37215}
V1_FEATURES = [
    "conn_count", "avg_duration", "avg_bytes", "bytes_per_packet", "night_ratio",
    "burst_score", "unique_dst_ips", "unique_dst_ports", "proto_entropy",
    "protocol_diversity", "scan_score", "scan_entropy", "c2_beacon_score",
    "payload_asymmetry", "iot_port_affinity", "horizontal_scan_flag",
    "vertical_scan_flag", "device_role_score", "src_bytes_ratio",
]
V1_MINMAX = V1_FEATURES[:11]


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def v1_fingerprints(flows: pd.DataFrame) -> pd.DataFrame:
    """One row per (source IP, clock hour) with the 19 original features."""
    f = flows[["ts", "dur", "proto", "src", "dst", "sport", "dport", "pkts", "bytes",
               "src_bytes", "role"]].copy()
    f["src"] = f["src"].astype(str)
    f["proto"] = f["proto"].astype(str)
    f["hour"] = (f["ts"] // 3600).astype("int64")
    f["minute"] = (f["ts"] // 60).astype("int64")
    hod = (f["ts"] % 86400) // 3600
    f["night"] = ((hod >= 22) | (hod < 6)).astype(float)
    f["iot"] = f["dport"].isin(IOT_PORTS).astype(float)
    f["low_sport"] = (f["sport"] < 1024).astype(float)
    f["dst_bytes"] = (f["bytes"] - f["src_bytes"]).clip(lower=0)
    g = f.groupby(["src", "hour"], sort=False)
    base = g.agg(conn_count=("ts", "size"), avg_duration=("dur", "mean"),
                 avg_bytes=("bytes", "mean"), sum_bytes=("bytes", "sum"),
                 sum_pkts=("pkts", "sum"), night_ratio=("night", "mean"),
                 unique_dst_ips=("dst", "nunique"), unique_dst_ports=("dport", "nunique"),
                 protocol_diversity=("proto", "nunique"), iot_port_affinity=("iot", "mean"),
                 device_role_score=("low_sport", "mean"), mean_src=("src_bytes", "mean"),
                 mean_dst=("dst_bytes", "mean"), sum_src=("src_bytes", "sum"),
                 malicious=("role", lambda s: bool((s == "bot").any())))
    base["bytes_per_packet"] = base["sum_bytes"] / base["sum_pkts"].clip(lower=1)
    base["scan_score"] = base["unique_dst_ips"] / base["conn_count"]
    base["payload_asymmetry"] = base["mean_src"] / (base["mean_dst"] + 1.0)
    base["src_bytes_ratio"] = base["sum_src"] / base["sum_bytes"].clip(lower=1)
    base["horizontal_scan_flag"] = ((base["unique_dst_ips"] > 50) & (base["unique_dst_ports"] < 3)).astype(float)
    base["vertical_scan_flag"] = ((base["unique_dst_ports"] > 20) & (base["unique_dst_ips"] < 5)).astype(float)
    base["burst_score"] = (f.groupby(["src", "hour", "minute"]).size().groupby(level=[0, 1]).max()
                           / base["conn_count"])
    base["proto_entropy"] = f.groupby(["src", "hour", "proto"]).size().groupby(level=[0, 1]).apply(
        lambda c: _entropy(c.to_numpy()))
    base["scan_entropy"] = f.groupby(["src", "hour", "dst"]).size().groupby(level=[0, 1]).apply(
        lambda c: _entropy(c.to_numpy()))

    def cov(ts: pd.Series) -> float:
        if len(ts) < 3:
            return 1.0
        d = np.diff(np.sort(ts.to_numpy()))
        m = d.mean()
        return float(d.std() / m) if m > 0 else 0.0
    base["c2_beacon_score"] = g["ts"].apply(cov)
    return base.reset_index()


def v1_cluster(fp: pd.DataFrame, k_range=range(2, 11), sample: int = 20000,
               seed: int = 42) -> dict:
    """Original AttribIQ campaign discovery: K-means with k chosen by silhouette."""
    X = fp[V1_FEATURES].to_numpy(dtype=float).copy()
    j = [V1_FEATURES.index(c) for c in V1_MINMAX]
    X[:, j] = MinMaxScaler().fit_transform(X[:, j])
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    best = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X)
        sil = silhouette_score(X[sub], km.labels_[sub])
        if best is None or sil > best["silhouette"]:
            best = {"k": k, "silhouette": float(sil), "labels": km.labels_,
                    "davies_bouldin": float(davies_bouldin_score(X[sub], km.labels_[sub]))}
    return best


def membership_similarity(fp: pd.DataFrame, labels: np.ndarray, rows: list[str],
                          cols: list[str]) -> np.ndarray:
    """Cosine similarity of hosts' cluster-membership histograms."""
    k = int(labels.max()) + 1
    M = pd.crosstab(fp["src"], labels).reindex(columns=range(k), fill_value=0)
    M = M.div(np.sqrt((M ** 2).sum(axis=1)), axis=0)
    A = M.reindex(rows).fillna(0).to_numpy()
    B = M.reindex(cols).fillna(0).to_numpy()
    return A @ B.T


def hdbscan_comembership(Z: np.ndarray, min_cluster_size: int = 2) -> np.ndarray:
    """1 if two hosts share a non-noise HDBSCAN cluster, else 0."""
    lab = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(Z)
    same = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    return same.astype(float)
