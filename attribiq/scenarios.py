"""Registry of the CTU-13 scenarios used by AttribIQ.

Each entry records the malware family, the official file name, the SHA-256 of
the exact file used in the experiments and the number of flow records it
contains. ``fetch_data.py`` refuses to use a file whose checksum differs.
"""
from __future__ import annotations

from dataclasses import dataclass

OFFICIAL_BASE = "https://mcfp.felk.cvut.cz/publicDatasets"


@dataclass(frozen=True)
class Scenario:
    sid: int
    family: str
    capture: str          # CTU-Malware-Capture-Botnet-NN directory name
    filename: str         # bidirectional NetFlow (Argus) file
    sha256: str | None    # checksum of the file used in the paper (None = not verified yet)
    n_flows: int | None   # number of flow records (header excluded)

    @property
    def official_url(self) -> str:
        return f"{OFFICIAL_BASE}/{self.capture}/detailed-bidirectional-flow-labels/{self.filename}"


SCENARIOS: dict[int, Scenario] = {
    1: Scenario(1, "Neris", "CTU-Malware-Capture-Botnet-42", "capture20110810.binetflow", None, None),
    2: Scenario(2, "Neris", "CTU-Malware-Capture-Botnet-43", "capture20110811.binetflow",
                "366b87099b5cbbf41b7464eefccc224b10497101a4597f4f5dab6b8259aa3cf8", 1808122),
    3: Scenario(3, "Rbot", "CTU-Malware-Capture-Botnet-44", "capture20110812.binetflow", None, None),
    4: Scenario(4, "Rbot", "CTU-Malware-Capture-Botnet-45", "capture20110815.binetflow", None, None),
    5: Scenario(5, "Virut", "CTU-Malware-Capture-Botnet-46", "capture20110815-2.binetflow",
                "ec67980705e59d794eb2cea5a39d6ea23507a531036e1a8a6cc16c8985d15da9", 129832),
    6: Scenario(6, "Menti", "CTU-Malware-Capture-Botnet-47", "capture20110816.binetflow", None, None),
    7: Scenario(7, "Sogou", "CTU-Malware-Capture-Botnet-48", "capture20110816-2.binetflow",
                "df0b5338190b967bd340a0d6c1bb3c34d1bbfb4b7ffa764c2dd26f77f1a26680", 114077),
    8: Scenario(8, "Murlo", "CTU-Malware-Capture-Botnet-49", "capture20110816-3.binetflow", None, None),
    9: Scenario(9, "Neris", "CTU-Malware-Capture-Botnet-50", "capture20110817.binetflow",
                "7974f3cb3f4b707d7543194b6cd96c52af804d4a11aad2645036fd6f3e635abe", 2087508),
    10: Scenario(10, "Rbot", "CTU-Malware-Capture-Botnet-51", "capture20110818.binetflow",
                 "8ae44d50e48fdff5b810b741b7824999ccf474081676562a54aa2b23d55d556f", 1309791),
    11: Scenario(11, "Rbot", "CTU-Malware-Capture-Botnet-52", "capture20110818-2.binetflow",
                 "d761b4513ee42be57193fe8f442e9234e73ac81f4fbd01248e37c23ca8ec1cf2", 107251),
    12: Scenario(12, "NSIS.ay", "CTU-Malware-Capture-Botnet-53", "capture20110819.binetflow",
                 "615e28233cb3c494b74c1016af940394c2765df9a285413256e65b0ebd11bbca", 325471),
    13: Scenario(13, "Virut", "CTU-Malware-Capture-Botnet-54", "capture20110815-3.binetflow", None, None),
}

# Scenarios evaluated in the paper: every CTU-13 scenario with more than one
# infected host (9-12) plus two single-host scenarios of further families (5, 7).
PAPER_SCENARIOS: tuple[int, ...] = (5, 7, 9, 10, 11, 12)
MULTI_HOST_SCENARIOS: tuple[int, ...] = (9, 10, 11, 12)


# Scenario 2 was obtained only after the method design was frozen (commit 98637ce)
# and is used exclusively for held-out cross-incident evaluation.
HELDOUT_SCENARIOS = (2,)
T2_EXTENDED_SCENARIOS = (2, 5, 7, 9, 10, 11, 12)

# SHA-256 of the files in the official CTU-13 archive as listed by two independent
# public manifests (EnriqueGonzt/ctu13-host-window-graph-builder,
# metadata/ctu13_binetflow_sha256.txt; clustercoder/nidra, ml/artifacts/metrics/ctu13_audit.json).
# Scenarios 2, 5, 9, 10, 11 and 12 used here match; scenario 7 is a re-ordered copy.
OFFICIAL_ARCHIVE_SHA256 = {
    1: "d5fafadc606100b3dc4080dbf851a40d78c522c222576a0bcc375cbf13e0c106",
    2: "366b87099b5cbbf41b7464eefccc224b10497101a4597f4f5dab6b8259aa3cf8",
    3: "a5e7311e25ff280e302ee5786b1aabf48dacfa6edaf25412f3aa1adbe1006039",
    4: "a2feb2d6253c2560fbffdfcc3e92b6c9235f1ea7c43a5f8d7902d7e352e84171",
    5: "ec67980705e59d794eb2cea5a39d6ea23507a531036e1a8a6cc16c8985d15da9",
    6: "b1b49932c7ce0c4e344431fbc229e4f91be06772abfe7a2e6e7d4b0d439d61d7",
    7: "f224baab443b56b2f0812cf2f9cdbb0f118b8af70095c196d9ead6a92c72ef0f",
    8: "c2ff96c8908f1b24b2ff9dc2033ca9fd034b994f3a59e0cc4f1f53a0615c4ec0",
    9: "7974f3cb3f4b707d7543194b6cd96c52af804d4a11aad2645036fd6f3e635abe",
    10: "8ae44d50e48fdff5b810b741b7824999ccf474081676562a54aa2b23d55d556f",
    11: "d761b4513ee42be57193fe8f442e9234e73ac81f4fbd01248e37c23ca8ec1cf2",
    12: "615e28233cb3c494b74c1016af940394c2765df9a285413256e65b0ebd11bbca",
    13: "a4dbb317f53eaaef2538d1cc1cd228bda3743a84779032601d5762742c4daad1",
}
