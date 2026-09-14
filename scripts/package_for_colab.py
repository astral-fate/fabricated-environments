"""Bundle this repository for a rented GPU, so exp0 can be run at a scale this machine cannot.

    python scripts/package_for_colab.py            # writes realness-code.zip
    python scripts/package_for_colab.py --with-results

Follows the pattern of `project-v2/colab/package_for_drive.py`. See `docs/RUNPOD.md` in this repository
for the full workflow; paper 1's `project-v2/colab/RUNPOD.md` covers the pod mechanics but its GPU
advice does NOT carry over, because exp0's workload is different:

  * Paper 1 was bound by HuggingFace's Python decode loop, so it recommended a 4090 and said an
    A100 bought bandwidth the software could not use. exp0 does no generation at all -- it runs
    forward passes for hidden states -- so it is bound by **weights fitting in VRAM** instead.
    Qwen3-32B in bf16 is ~64 GB, which a 4090 (24 GB) and even an A40 (48 GB) cannot hold. 32B
    needs an 80 GB card; 8B is comfortable on a 4090.
  * A 4090 reports ~23.6 GiB, *below* the 24 GiB auto-threshold that selects 4-bit. That matters
    more here than it did for paper 1: NF4 perturbs the residual stream, which is the entire
    measurement. `--dtype bfloat16` is passed explicitly and `quantised` is recorded in the result.
  * Attach a network volume at `/workspace`, or a destroyed pod keeps nothing.

Why the bundle is self-contained
--------------------------------
`realness/` vendors its four reused files from `project-v2` rather than importing across
repositories precisely so this zip does not have to carry paper 1 as well. `sync_from_project_v2`
skips its check when the source repository is absent, so the bundle runs on a fresh pod.

Secrets are never bundled. `.env` is excluded by name and the packer refuses to write a zip that
contains anything shaped like a live key.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "realness-code.zip"

INCLUDE = ["src", "analyze", "experiments", "tests", "scripts"]
INCLUDE_FILES = ["scripts/check.py", "requirements.txt", "README.md",
                 "PREREGISTRATION.md", "docs/RUNPOD.md"]

#: Cached stage-1 and stage-2 artifacts, shipped by default.
#:
#: This is what makes the rented-GPU run need NO API KEYS AND NO NETWORK. The episodes were
#: generated once, the simulated replays once; both are append-only caches keyed by episode id, so
#: the pod re-runs only stages 3-6 -- stimulus construction, extraction, analysis -- which are pure
#: GPU and pure arithmetic. Nothing secret travels, the scale comparison is against the SAME
#: trajectories rather than freshly sampled ones, and a difference between 1.7B and 32B is
#: therefore a difference in the model rather than in the data.
INCLUDE_CACHES = [
    "results/exp0/episodes.jsonl",
    "results/exp0/replays.jsonl",
    "results/exp0/provenance.json",
]

EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", "checkouts", "replay", ".venv"}
EXCLUDE_NAMES = {".env", "realness-code.zip"}

#: Patterns that must never appear in a bundled file. Cheap, and the failure it prevents -- a live
#: key uploaded to a rented machine and then to a Hub repo -- is not cheap.
SECRET_PATTERNS = [
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}"),          # Groq
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),           # OpenAI
    re.compile(r"\bnvapi-[A-Za-z0-9_\-]{20,}"),     # NVIDIA
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),           # Hugging Face
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),            # AWS
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def collect(with_results: bool) -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE_FILES + INCLUDE_CACHES:
        p = ROOT / name
        if p.exists():
            files.append(p)
    roots = list(INCLUDE) + (["results"] if with_results else [])
    for d in roots:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if p.name in EXCLUDE_NAMES or p.name.startswith(".env"):
                continue
            if p.suffix in {".pyc", ".zip", ".npz"}:
                continue            # activation cache: 410 MB and regenerated in minutes
            files.append(p)
    return files


def scan_for_secrets(files: list[Path]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                found.append((str(p.relative_to(ROOT)), pat.pattern))
    return found


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--with-results", action="store_true",
                    help="include results/ (large; needed only to re-analyse remotely)")
    args = ap.parse_args(argv)

    files = collect(args.with_results)
    leaks = scan_for_secrets(files)
    if leaks:
        print("FAIL: refusing to bundle -- something shaped like a live key was found:")
        for path, pat in leaks:
            print(f"  {path}  ({pat})")
        return 1

    out = Path(args.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())

    size = out.stat().st_size / 1024
    print(f"  {out.name}  {len(files)} files  {size:.0f} KiB")
    print("\nOn the pod -- NO API KEY AND NO NETWORK NEEDED (stages 1-2 ship cached):")
    print("  unzip realness-code.zip -d realness && cd realness")
    print("  pip install -r requirements.txt")
    print("  python scripts/check.py")
    print("")
    print("  # Heidari et al.'s range is 7B-49B. Run 8B first: it is the cheap point that")
    print("  # already enters that range, and it tells you whether 32B is worth the hour.")
    print("  python experiments/exp0_construct_separation.py \\")
    print("      --model Qwen/Qwen3-8B  --dtype bfloat16 --batch-size 2")
    print("  cp results/exp0/exp0.json results/exp0/exp0-8b.json")
    print("")
    print("  python experiments/exp0_construct_separation.py \\")
    print("      --model Qwen/Qwen3-32B --dtype bfloat16 --batch-size 1")
    print("  cp results/exp0/exp0.json results/exp0/exp0-32b.json")
    print("")
    print("Bring back only the two exp0-*.json files -- a few hundred KiB each. The activation")
    print("cache is ~0.4 GB at 1.7B and scales with hidden size; leave it on the pod.")
    print("")
    print("Before believing any number, check two fields in the result:")
    print("  \"quantised\": false   -- NF4 perturbs the residual stream, which IS the measurement.")
    print("                          A 4090 reports ~23.6 GiB and would auto-select 4-bit, so")
    print("                          --dtype bfloat16 is passed explicitly above.")
    print("  \"n_groups\": 24       -- same 24 episodes as the 1.7B run, so a difference between")
    print("                          scales is the model and not the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
