#!/usr/bin/env bash
# Regenerate the analyses and figures, render from results/, then build the PDF.
#
# Figures are regenerated every build for the same reason the numbers are: a figure committed once
# and never rebuilt can drift from the result it illustrates, and nobody notices because a stale
# PDF looks exactly like a fresh one.
set -euo pipefail
cd "$(dirname "$0")/.."
python analyze/error_analysis.py > /dev/null
python analyze/figures.py
python paper/render.py
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build1.log 2>&1 || { tail -30 build1.log; exit 1; }
bibtex main > build_bib.log 2>&1 || tail -20 build_bib.log
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build2.log 2>&1 || { tail -30 build2.log; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build3.log 2>&1 || { tail -30 build3.log; exit 1; }
echo "  main.pdf  $(wc -c < main.pdf) bytes"
