"""One-shot repair: restore LaTeX control words mangled into ASCII control characters.

A shell heredoc collapsed `\\\\b` to `\\b` before Python parsed it, so `\\begin` became a backspace
followed by `egin`, `\\texttt` became a tab, `\\ref` became a carriage return, and `\\appendix`
became a bell. LaTeX then died on `^^H egin{figure}`.

The mapping is unambiguous in this direction because none of these control characters has any
legitimate place in the manuscript source: a tab is not used for indentation here, and the others
cannot be typed deliberately. After this runs, the file should contain no C0 control characters
except newline.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "main.tex.tmpl"

# control character -> the LaTeX control word whose backslash was eaten
REPAIRS = {
    "\x08": "\\b",     # \begin, \bibliographystyle
    "\x07": "\\a",     # \appendix
    "\t": "\\t",       # \texttt, \toprule, \textbf
    "\r": "\\r",       # \ref
    "\x0c": "\\f",     # \footnote
    "\x0b": "\\v",     # \vspace
}


def main() -> int:
    s = TARGET.read_text(encoding="utf-8")
    before = {c: s.count(c) for c in REPAIRS if c in s}
    if not before:
        print("  no control characters found; nothing to repair")
        return 0
    for ch, word in REPAIRS.items():
        s = s.replace(ch, word)
    TARGET.write_text(s, encoding="utf-8")

    leftover = [hex(ord(c)) for c in set(s) if ord(c) < 32 and c != "\n"]
    print(f"  repaired: { {hex(ord(c)): n for c, n in before.items()} }")
    print(f"  remaining control characters: {leftover or 'none'}")
    return 1 if leftover else 0


if __name__ == "__main__":
    raise SystemExit(main())
