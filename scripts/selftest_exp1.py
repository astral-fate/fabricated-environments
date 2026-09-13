"""End-to-end self-test of the exp1 pipeline. No API key, no GPU, no spend.

    python scripts/selftest_exp1.py

Runs the whole thing -- three arms, episode loop, pure-function scope detection, cell pairing, all
five validity gates, the cluster bootstrap over every contrast, and the verdict path -- with a
scripted actor and templated fabrication, into a throwaway directory.

It cannot detect a wrong *measurement*, and it is not meant to: every arm shares one fixed actor
script, so every contrast is identically zero by construction. What it detects is a broken
*wiring* -- a misaligned pairing, a gate that raises, a bootstrap that degenerates, a verdict path
that crashes on an edge case -- in seconds, rather than after a collection that costs hours.

The run must end in `OFFLINE -- NOT REPORTABLE`. A self-test that produced a verdict would be the
most dangerous artifact here: a well-formed result with no measurement behind it.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exp1-selftest"


def main() -> int:
    shutil.rmtree(OUT, ignore_errors=True)
    cmd = [sys.executable, "-u", str(ROOT / "experiments" / "exp1_behavioural_divergence.py"),
           "--offline", "--out", str(OUT), "--seeds", "1", "--max-turns", "8"]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, timeout=600)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-2000:])
        print(f"FAIL: exp1 self-test exited {proc.returncode}")
        return 1

    artifact = OUT / "exp1.json"
    if not artifact.exists():
        print("FAIL: the self-test produced no artifact")
        return 1
    got = json.loads(artifact.read_text(encoding="utf-8"))

    problems = []
    if not got.get("offline"):
        problems.append("artifact is not stamped offline")
    if got.get("verdict") != "OFFLINE -- NOT REPORTABLE":
        problems.append(f"offline run issued a verdict: {got.get('verdict')!r}")
    if not got.get("gates_all_pass"):
        problems.append(f"a validity gate failed on clean synthetic data: {got.get('gates')}")
    if got["pairing"]["n_cells"] != 3:
        problems.append(f"expected 3 cells, got {got['pairing']['n_cells']}")
    for key in ("real->sim", "real->sim_scripted", "sim_scripted->sim"):
        if key not in got["results"]:
            problems.append(f"contrast {key} missing from the results")

    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1

    print(f"  exp1 self-test ok: {got['pairing']['n_cells']} cells, "
          f"{len(got['results'])} contrasts, all gates pass, no verdict issued")
    shutil.rmtree(OUT, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
