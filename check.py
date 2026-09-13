"""Run every check in this repository. No GPU, no network, no API keys.

    python check.py

Suites, in dependency order. Each is independently runnable; this exists so a reviewer can
establish in one command that the repository agrees with itself, without reading the code first.

    vendor        the four files reused from paper 1 are byte-identical to their source
    scope         the violation detector is pure, and its path arithmetic is correct
    task          the difficulty classes are what they claim -- verified by exhaustive search
    real arm      sealed (no egress), grounded, and interchangeable at the tool surface
    resume        a killed run loses no work and double-counts nothing
    probe         the probe mathematics, on synthetic data with a known answer
    claims        every claim in the paper, recomputed from results/
    citations     no citation orphans between the manuscript and refs.bib

A non-zero exit means the repository disagrees with itself and nothing should be submitted until
it does not.

`claims` is skipped while no paper source exists. It becomes the suite that matters most once
there is one: paper 1 shipped a cost table that contradicted its own artifact and survived review
because the checker recomputed what the paper SAID and confirmed the paper said it consistently.
Here the manuscript carries no numbers of its own.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUITES: list[tuple[str, list[str]]] = [
    ("vendor", ["scripts/sync_from_project_v2.py"]),
    ("scope", ["-m", "pytest", "tests/test_scope.py", "-q"]),
    ("task", ["-m", "pytest", "tests/test_task.py", "-q"]),
    ("real arm", ["-m", "pytest", "tests/test_real_arm.py", "-q"]),
    ("resume", ["-m", "pytest", "tests/test_resume.py", "-q"]),
    ("probe", ["-m", "pytest", "tests/test_probe.py", "-q"]),
]


def run(name: str, argv: list[str]) -> bool:
    print(f"\n=== {name} " + "=" * (68 - len(name)))
    proc = subprocess.run([sys.executable, *argv], cwd=ROOT)
    ok = proc.returncode == 0
    print(f"--- {name}: {'ok' if ok else 'FAIL'}")
    return ok


def main() -> int:
    results = {name: run(name, argv) for name, argv in SUITES}

    paper = ROOT / "paper" / "render.py"
    if paper.exists():
        results["claims"] = run("claims", ["paper/render.py", "--check"])
        results["citations"] = run("citations", ["paper/check_citations.py"])
    else:
        print("\n=== claims " + "=" * 68)
        print("  no paper source yet; skipped")

    print("\n" + "=" * 78)
    failed = [n for n, ok in results.items() if not ok]
    for name, ok in results.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if failed:
        print(f"\n{len(failed)} suite(s) failed: {', '.join(failed)}")
        return 1
    print(f"\nall {len(results)} suite(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
