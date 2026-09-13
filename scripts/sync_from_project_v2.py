"""Verify that vendored files are byte-identical to their source in `project-v2`.

    python scripts/sync_from_project_v2.py            # check; non-zero exit on drift
    python scripts/sync_from_project_v2.py --update   # re-copy from source

Why vendor rather than import across repositories
--------------------------------------------------
`realness/` reuses four files from paper 1 unchanged. Importing them over a relative path would
keep one copy, but it would also mean the probe arm cannot be zipped and shipped to a rented GPU
without dragging the whole of `project-v2` along, and that packaging step is the difference
between exp0 running at 1.7B on this machine and running at 32B on a 4090.

So the files are copied. The cost of copying is drift -- a fix made in one repository and not the
other -- and drift is exactly the kind of thing that stays invisible until a number disagrees with
itself. This script makes the reuse boundary a mechanical check instead of a comment in a README:
`check.py` runs it, and a modified vendored file fails the build with the hash that changed.

If a vendored file genuinely needs to differ here, it should be moved out of VENDORED and given
its own module with a docstring saying what diverged and why.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT.parent / "project-v2"

#: destination (relative to realness/) -> source (relative to project-v2/)
VENDORED = {
    "src/substrate.py": "src/substrate.py",
    "src/srm.py": "src/srm.py",
    "src/providers.py": "runner/providers.py",
    "scripts/watchdog.py": "scripts/watchdog.py",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="re-copy from project-v2")
    ap.add_argument("--source", default=str(SOURCE))
    args = ap.parse_args(argv)
    source = Path(args.source)

    if not source.exists():
        # A rented GPU or a fresh clone will not have paper 1 alongside. That is not drift, and
        # failing the whole build for it would make the bundle unusable where it matters most.
        print(f"  source repository not present at {source}; skipping vendor check")
        return 0

    drift, missing = [], []
    for dst_rel, src_rel in sorted(VENDORED.items()):
        dst, src = ROOT / dst_rel, source / src_rel
        if not src.exists():
            missing.append(f"{src_rel} (source)")
            continue
        if not dst.exists():
            missing.append(f"{dst_rel} (vendored copy)")
            continue
        if args.update:
            shutil.copy2(src, dst)
            print(f"  updated {dst_rel}")
            continue
        h_dst, h_src = sha(dst), sha(src)
        status = "ok  " if h_dst == h_src else "DRIFT"
        print(f"  {status} {dst_rel:<24} {h_dst[:12]}")
        if h_dst != h_src:
            drift.append((dst_rel, src_rel, h_dst[:12], h_src[:12]))

    if missing:
        print("\nFAIL: missing file(s):")
        for m in missing:
            print(f"  {m}")
        return 1
    if drift:
        print("\nFAIL: vendored file(s) differ from source:")
        for dst_rel, src_rel, hd, hs in drift:
            print(f"  {dst_rel} ({hd}) != project-v2/{src_rel} ({hs})")
        print("\n  Re-copy with --update, or move the file out of VENDORED and document "
              "what diverged and why.")
        return 1
    if not args.update:
        print(f"\n  {len(VENDORED)} vendored file(s) identical to source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
