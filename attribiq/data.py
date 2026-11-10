"""Loading CTU-13 bidirectional NetFlow (Argus ``.binetflow``) files.

The loader normalises one capture into a typed table, keeps only flows
initiated by hosts inside the monitored prefix, and derives host roles from
the CTU-13 labels. Labels are used for evaluation and for the analyst-provided
inputs defined in the paper (a confirmed seed host, a library of previously
confirmed incidents); they never enter a feature.
"""
from __future__ import annotations

import hashlib
import ipaddress
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from .scenarios import SCENARIOS

MONITORED_PREFIX = ipaddress.ip_network("147.32.0.0/16")

_COLUMNS = ["StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir", "DstAddr",
            "Dport", "State", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes", "Label"]


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_port(values: pd.Series) -> np.ndarray:
    """Ports are decimal, hexadecimal (ICMP type/code, e.g. '0x0303') or empty."""
    s = values.fillna("").astype(str).str.strip()
    out = pd.to_numeric(s, errors="coerce")
    hexmask = s.str.startswith("0x") & out.isna()
    if hexmask.any():
        out.loc[hexmask] = s[hexmask].map(lambda v: int(v, 16))
    return out.to_numpy(dtype="float64")


def is_internal(ips: pd.Series) -> np.ndarray:
    """Vectorised membership test for the monitored /16 (string prefix match)."""
    net = str(MONITORED_PREFIX.network_address).split(".")[:2]
    prefix = ".".join(net) + "."
    return ips.astype(str).str.startswith(prefix).to_numpy()


def flow_role(labels: pd.Series) -> np.ndarray:
    lab = labels.astype(str)
    role = np.full(len(lab), "background", dtype=object)
    role[lab.str.contains("From-Normal", regex=False).to_numpy()] = "normal"
    role[lab.str.contains("From-Botnet", regex=False).to_numpy()] = "bot"
    return role


def read_binetflow(path: str | Path) -> pd.DataFrame:
    """Read an Argus .binetflow file into a normalised DataFrame (all flows)."""
    opts = pacsv.ConvertOptions(column_types={c: pa.string() for c in _COLUMNS},
                                include_columns=_COLUMNS)
    tbl = pacsv.read_csv(str(path), convert_options=opts,
                         parse_options=pacsv.ParseOptions(newlines_in_values=False))
    df = tbl.to_pandas()
    ts = pd.to_datetime(df["StartTime"].str.strip(), format="%Y/%m/%d %H:%M:%S.%f")
    out = pd.DataFrame({
        "ts": (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy(),
        "dur": pd.to_numeric(df["Dur"], errors="coerce").fillna(0.0).to_numpy(),
        "proto": df["Proto"].str.strip().str.lower().astype("category"),
        "src": df["SrcAddr"].str.strip(),
        "sport": _parse_port(df["Sport"]),
        "dir": df["Dir"].str.strip().astype("category"),
        "dst": df["DstAddr"].str.strip(),
        "dport": _parse_port(df["Dport"]),
        "state": df["State"].fillna("").str.strip().astype("category"),
        "pkts": pd.to_numeric(df["TotPkts"], errors="coerce").fillna(0).astype("int64").to_numpy(),
        "bytes": pd.to_numeric(df["TotBytes"], errors="coerce").fillna(0).astype("int64").to_numpy(),
        "src_bytes": pd.to_numeric(df["SrcBytes"], errors="coerce").fillna(0).astype("int64").to_numpy(),
    })
    out["role"] = pd.Categorical(flow_role(df["Label"]), categories=["background", "normal", "bot"])
    return out


def load_scenario(sid: int, raw_dir: str | Path, cache_dir: str | Path,
                  verify: bool = True) -> pd.DataFrame:
    """Return flows initiated inside the monitored prefix, cached as parquet."""
    sc = SCENARIOS[sid]
    cache = Path(cache_dir) / f"sc{sid:02d}_internal.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    path = Path(raw_dir) / sc.filename
    if verify and sc.sha256 is not None:
        got = sha256_of(path)
        if got != sc.sha256:
            raise ValueError(f"checksum mismatch for scenario {sid}: {got} != {sc.sha256}")
    flows = read_binetflow(path)
    if sc.n_flows is not None and len(flows) != sc.n_flows:
        raise ValueError(f"scenario {sid}: {len(flows)} flows, expected {sc.n_flows}")
    flows = flows[is_internal(flows["src"])].reset_index(drop=True)
    flows["src"] = flows["src"].astype("category")
    cache.parent.mkdir(parents=True, exist_ok=True)
    flows.to_parquet(cache, index=False)
    return flows


def host_roles(flows: pd.DataFrame) -> pd.Series:
    """Role of each internal source host: 'bot' > 'normal' > 'background'."""
    r = flows.groupby("src", observed=True)["role"].agg(
        lambda s: "bot" if (s == "bot").any() else ("normal" if (s == "normal").any() else "background"))
    return r.astype(str)
