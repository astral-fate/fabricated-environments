"""Citation coverage: every key cited must exist in refs.bib, and vice versa.

Zero citation orphans is a stated quality standard, and the direction that matters most here is
`cited but missing from bib` -- a key with no entry renders as [?] in the PDF and is the visible
symptom of a fabricated reference. The reverse direction (an uncited entry) is reported too, since
an entry nobody cites is usually a reference that was removed from the prose and left behind.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent


def main() -> int:
    tmpl = (PAPER / "main.tex.tmpl").read_text(encoding="utf-8")
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")

    used: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", tmpl):
        used.update(k.strip() for k in m.group(1).split(",") if k.strip())
    defined = set(re.findall(r"@\w+\{([^,]+),", bib))

    missing = sorted(used - defined)
    uncited = sorted(defined - used)
    print(f"  cited in text: {len(used)}    in refs.bib: {len(defined)}")
    if missing:
        print(f"\nFAIL: {len(missing)} key(s) cited but absent from refs.bib: {', '.join(missing)}")
        print("  These render as [?] and are the visible symptom of a fabricated reference.")
        return 1
    if uncited:
        print(f"  {len(uncited)} entry(ies) in refs.bib never cited: {', '.join(uncited)}")
    print("  no citation orphans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
