"""Export `docs/index.html` to a scrollable landscape PDF, one slide per page.

    python scripts/export_slides_pdf.py

The deck's CSS already carries `@page { size:A4 landscape; margin:0 }` and a print block that
forces one slide per page, so the job here is to drive a real browser engine and tell it to honour
that rather than to re-lay-out anything.

Two details do the work, and both are easy to get wrong:

  print_background   off by default. The deck is dark with gradient panels, so without this every
                     page renders as black text on white and is unreadable.
  prefer_css_page_size  without it the browser imposes its own default page size and the
                     `@page` landscape rule is ignored, giving portrait pages with clipped slides.

Falls back to headless Chrome or Edge if Playwright has no browser binary installed, since the
CSS does the layout either way.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "index.html"
OUT = ROOT / "docs" / "fabricated-environments-slides.pdf"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]


def via_playwright() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(SRC.as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(OUT),
                landscape=True,
                print_background=True,       # the deck is dark; without this it prints blank
                prefer_css_page_size=True,   # honour the deck's own @page landscape rule
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            browser.close()
        return True
    except Exception as exc:                 # noqa: BLE001 - fall through to the browser binary
        print(f"  playwright unavailable ({type(exc).__name__}); trying a browser binary")
        return False


def via_chrome() -> bool:
    exe = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None) or shutil.which("chrome")
    if not exe:
        return False
    print(f"  using {Path(exe).name}")
    cmd = [
        exe, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=10000",       # let the images decode before the snapshot
        f"--print-to-pdf={OUT}",
        SRC.as_uri(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not OUT.exists():
        print(proc.stderr[-600:])
        return False
    return True


def main() -> int:
    if not SRC.exists():
        print(f"no deck at {SRC}. Run `python scripts/build_slides.py` first.")
        return 1
    OUT.unlink(missing_ok=True)

    if not (via_playwright() or via_chrome()):
        print("FAIL: no usable browser engine for PDF export.")
        return 1

    size_kb = OUT.stat().st_size // 1024
    pages = "?"
    try:
        info = subprocess.run(["pdfinfo", str(OUT)], capture_output=True, text=True, timeout=30)
        for line in info.stdout.splitlines():
            if line.lower().startswith("pages"):
                pages = line.split()[-1]
            if line.lower().startswith("page size"):
                print(f"  {line.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    print(f"  {OUT.relative_to(ROOT)}  {pages} pages  {size_kb} KiB")

    expected = SRC.read_text(encoding="utf-8").count('<section class="slide')
    if pages != "?" and int(pages) != expected:
        print(f"  WARNING: {expected} slides but {pages} PDF pages -- a slide has overflowed "
              f"onto a second page. Shorten it or reduce its figure height.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
