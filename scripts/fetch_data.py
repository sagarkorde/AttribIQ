#!/usr/bin/env python3
"""Download the CTU-13 bidirectional flow files used in the paper and verify them.

    python scripts/fetch_data.py --out data/raw            # official site first, mirrors second
    python scripts/fetch_data.py --out data/raw --verify-only

Every file is accepted only if its SHA-256 equals the value in
attribiq/scenarios.py, which is the checksum of the exact file used in the paper.
The official files are published by the Stratosphere Laboratory (Czech Technical
University in Prague); the mirrors below are public copies whose content was found
identical (same SHA-256) to each other. Scenario 7 is only available to us as a
time-sorted copy, whose checksum is recorded separately.
"""
from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attribiq.data import sha256_of  # noqa: E402
from attribiq.scenarios import HELDOUT_SCENARIOS, PAPER_SCENARIOS, SCENARIOS  # noqa: E402

LFS = "https://media.githubusercontent.com/media"
MIRRORS = {
    2: [f"{LFS}/ahmdzzzki/botnet-detection-deep-learning/main/data/2/capture20110811.binetflow"],
    5: [f"https://raw.githubusercontent.com/Awadnawafleh/ctu13/main/capture20110815-2.binetflow"],
    7: [f"https://raw.githubusercontent.com/mincuties/Botnet-Netflow-Scenario-Analyzer/main/data/scenarios/scenario07.binetflow.gz"],
    9: [f"{LFS}/kbaran1998/CS4035-cyber-data/main/lab3/data/dataset_9/capture20110817.binetflow",
        f"{LFS}/kponichtera/cs4035-cda/main/lab3/data/dataset_9/capture20110817.binetflow"],
    10: [f"{LFS}/kbaran1998/CS4035-cyber-data/main/lab3/data/dataset_10/capture20110818.binetflow",
         f"{LFS}/kponichtera/cs4035-cda/main/lab3/data/dataset_10/capture20110818.binetflow"],
    11: [f"{LFS}/kbaran1998/CS4035-cyber-data/main/lab3/data/dataset_11/capture20110818-2.binetflow",
         f"{LFS}/kponichtera/cs4035-cda/main/lab3/data/dataset_11/capture20110818-2.binetflow"],
    12: [f"{LFS}/kbaran1998/CS4035-cyber-data/main/lab3/data/dataset_12/capture20110819.binetflow",
         f"{LFS}/kponichtera/cs4035-cda/main/lab3/data/dataset_12/capture20110819.binetflow"],
}


def download(url: str, dest: Path) -> bool:
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, length=1 << 20)
    except Exception as exc:  # noqa: BLE001
        print(f"  failed: {url} ({exc})")
        tmp.unlink(missing_ok=True)
        return False
    if url.endswith(".gz"):
        with gzip.open(tmp, "rb") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        tmp.unlink()
    else:
        tmp.replace(dest)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    for sid in tuple(PAPER_SCENARIOS) + tuple(HELDOUT_SCENARIOS):
        sc = SCENARIOS[sid]
        dest = out / sc.filename
        if dest.exists() and sha256_of(dest) == sc.sha256:
            print(f"scenario {sid:2d}: present and verified")
            continue
        if args.verify_only:
            print(f"scenario {sid:2d}: MISSING or checksum mismatch")
            ok = False
            continue
        for url in [sc.official_url] + MIRRORS.get(sid, []):
            print(f"scenario {sid:2d}: trying {url}")
            if download(url, dest) and sha256_of(dest) == sc.sha256:
                print(f"scenario {sid:2d}: verified")
                break
        else:
            print(f"scenario {sid:2d}: no source produced the expected checksum {sc.sha256}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
