import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _flows(rows):
    df = pd.DataFrame(rows, columns=["ts", "dur", "proto", "src", "sport", "dir", "dst", "dport",
                                     "state", "pkts", "bytes", "src_bytes", "role"])
    df["proto"] = df["proto"].astype("category")
    df["state"] = df["state"].astype("category")
    df["role"] = pd.Categorical(df["role"], categories=["background", "normal", "bot"])
    return df


@pytest.fixture
def toy_flows():
    """Two bots that beacon to the same C2 every 60 s and two benign web clients."""
    rng = np.random.default_rng(0)
    rows = []
    for bot in ("147.32.84.165", "147.32.84.191"):
        for k in range(60):
            rows.append([1000.0 + 60 * k + rng.uniform(0, 0.5), 0.1, "tcp", bot, 1025, "->",
                         "6.6.6.6", 6667, "SPA_SPA", 6, 400, 150, "bot"])
            rows.append([1010.0 + 60 * k, 0.0, "udp", bot, 1026, "<->", "147.32.80.9", 53,
                         "CON", 2, 150, 70, "bot"])
        for minute in (10, 30, 50):                 # synchronised command bursts
            for j in range(10):
                rows.append([1000.0 + 60 * minute + 2 * j, 0.1, "tcp", bot, 3000 + j, "->",
                             f"5.5.5.{j}", 80, "S_RA", 2, 120, 60, "bot"])
    for i, host in enumerate(("147.32.84.10", "147.32.84.11")):
        for k in range(80):
            rows.append([1000.0 + 37 * k + 11 * i, 2.0, "tcp", host, 2000 + k, "->",
                         f"93.184.{i}.{k % 20}", 443, "FSPA_FSPA", 20, 9000, 1200, "background"])
    return _flows(rows)
