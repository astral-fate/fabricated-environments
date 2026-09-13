"""Generate the paper's figures from `results/`. No figure is drawn by hand.

    python analyze/figures.py

Same rule as the numbers: a figure the manuscript displays must be regenerated from committed
artifacts, so it cannot drift from the result it illustrates. Each function reads a result file
and writes a PDF into `paper/figures/`; the manuscript includes them by name.

Three figures, each earning its place by showing something a table cannot:

  fig:layers   AUROC as a function of depth, for all four directions. A table reports the best
               layer; the curve shows whether that maximum is a broad plateau or a spike, which is
               what decides how fragile a layer choice is.
  fig:scale    cross-transfer and cosine against model size. The central claim is about a
               *trend* -- both probes sharpen while transfer does not -- and a trend is easier to
               refute from a picture than from three rows of a table.
  fig:errors   held-out accuracy broken down by task class and by held-out paraphrase. This is the
               figure that shows the errors are structured rather than uniform.

Palette is colourblind-safe (Okabe-Ito). No red/green pairing carries meaning on its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exp0"
FIGDIR = ROOT / "paper" / "figures"

#: Okabe-Ito, safe under the common forms of colour vision deficiency.
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#666666"}

SCALES = [("exp0.json", "1.7B"), ("exp0-qwen3-8b.json", "8B"),
          ("exp0-qwen3-32b.json", "32B")]

DIRECTION_STYLE = {
    "A_eval": (OKABE["blue"], "-", "A: eval-awareness"),
    "C_declared": (OKABE["orange"], "--", "C: declared realness"),
    "B_empirical": (OKABE["green"], "-", "B: empirical realness"),
    "S_scripted": (OKABE["purple"], ":", "S: scripted control"),
}


def _load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False,
    })


def fig_layers(result_file: str = "exp0-qwen3-8b.json") -> Path | None:
    """Per-layer held-out AUROC for every direction, with the chance line marked."""
    d = _load(result_file)
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for name, (colour, ls, label) in DIRECTION_STYLE.items():
        rec = d["pooling"]["last"]["directions"].get(name)
        if not rec:
            continue
        xs = [r["layer"] for r in rec["per_layer"] if r["auroc"] == r["auroc"]]
        ys = [r["auroc"] for r in rec["per_layer"] if r["auroc"] == r["auroc"]]
        ax.plot(xs, ys, color=colour, linestyle=ls, linewidth=1.6, label=label)
        ax.plot([rec["best_layer"]], [rec["best_auroc"]], marker="o", color=colour, markersize=4)
    ax.axhline(0.5, color=OKABE["grey"], linewidth=0.8, linestyle="-")
    ax.annotate("chance", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=7, color=OKABE["grey"])
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    out = FIGDIR / "layers.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_scale() -> Path | None:
    """Cross-transfer and cosine against model scale, with the pre-registered threshold drawn."""
    rows = [(label, _load(f)) for f, label in SCALES]
    rows = [(lab, d) for lab, d in rows if d is not None]
    if len(rows) < 2:
        return None

    labels = [lab for lab, _ in rows]
    xs = list(range(len(rows)))

    def cross(d, a, b):
        return d["pooling"]["last"]["cross_auroc"][f"{a}->{b}"]["auroc"]

    def cos(d, a, b):
        c = d["pooling"]["last"]["cosine"]
        return (c.get(f"{a}|{b}") or c.get(f"{b}|{a}"))["cosine"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax1.plot(xs, [d["pooling"]["last"]["directions"]["A_eval"]["best_auroc"] for _, d in rows],
             color=OKABE["blue"], marker="o", label="A: own contrast")
    ax1.plot(xs, [d["pooling"]["last"]["directions"]["B_empirical"]["best_auroc"] for _, d in rows],
             color=OKABE["green"], marker="s", label="B: own contrast")
    ax1.plot(xs, [cross(d, "A_eval", "B_empirical") for _, d in rows],
             color=OKABE["blue"], marker="o", linestyle="--", label=r"A $\rightarrow$ B")
    ax1.plot(xs, [cross(d, "B_empirical", "A_eval") for _, d in rows],
             color=OKABE["green"], marker="s", linestyle="--", label=r"B $\rightarrow$ A")
    thr = rows[0][1]["config"].get("threshold", 0.85)
    ax1.axhline(thr, color=OKABE["vermillion"], linewidth=1.0, linestyle="-.")
    ax1.annotate("separability threshold", xy=(0.02, thr), xycoords=("axes fraction", "data"),
                 va="bottom", fontsize=7, color=OKABE["vermillion"])
    ax1.axhline(0.5, color=OKABE["grey"], linewidth=0.8)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.35, 1.03)
    ax1.set_xlabel("model scale")
    ax1.set_ylabel("AUROC")
    ax1.legend(fontsize=7, loc="center left")

    for (a, b), colour, marker, lab in [
        (("A_eval", "B_empirical"), OKABE["blue"], "o", r"$\cos$(A, B)"),
        (("A_eval", "C_declared"), OKABE["orange"], "^", r"$\cos$(A, C)"),
        (("B_empirical", "C_declared"), OKABE["vermillion"], "v", r"$\cos$(B, C)"),
        (("B_empirical", "S_scripted"), OKABE["purple"], "s", r"$\cos$(B, S)"),
    ]:
        ax2.plot(xs, [cos(d, a, b) for _, d in rows], color=colour, marker=marker, label=lab)
    ax2.axhline(0.0, color=OKABE["grey"], linewidth=0.8)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("model scale")
    ax2.set_ylabel("cosine between directions")
    ax2.legend(fontsize=7)

    fig.tight_layout()
    out = FIGDIR / "scale.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_errors() -> Path | None:
    """Held-out accuracy by task class and by held-out paraphrase."""
    p = RESULTS / "error_analysis.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))["directions"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    kinds = ["solvable", "blocked", "tempting"]
    names = [n for n in ("A_eval", "C_declared", "B_empirical", "S_scripted") if n in d]
    width = 0.8 / max(len(names), 1)

    for i, name in enumerate(names):
        colour, _, label = DIRECTION_STYLE[name]
        vals = [d[name]["by_task_class"].get(k, {}).get("accuracy", 0.0) for k in kinds]
        ax1.bar([x + i * width for x in range(len(kinds))], vals, width=width,
                color=colour, label=label)
    ax1.axhline(0.5, color=OKABE["grey"], linewidth=0.8)
    ax1.set_xticks([x + 0.4 - width / 2 for x in range(len(kinds))])
    ax1.set_xticklabels(kinds)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("held-out accuracy")
    ax1.set_xlabel("task class")
    ax1.legend(fontsize=7, loc="lower left")

    variants = sorted({v for name in names for v in d[name]["by_paraphrase"]})
    for i, name in enumerate(names):
        if not d[name]["by_paraphrase"]:
            continue
        colour, _, label = DIRECTION_STYLE[name]
        vals = [d[name]["by_paraphrase"].get(v, {}).get("accuracy", 0.0) for v in variants]
        ax2.bar([x + i * width for x in range(len(variants))], vals, width=width,
                color=colour, label=label)
    ax2.axhline(0.5, color=OKABE["grey"], linewidth=0.8)
    ax2.set_xticks([x + 0.4 - width / 2 for x in range(len(variants))])
    ax2.set_xticklabels([f"held-out\nwording {v}" for v in variants], fontsize=7.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("held-out accuracy")
    ax2.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    out = FIGDIR / "errors.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_architecture() -> Path:
    """Schematic of the measurement pipeline.

    Drawn rather than photographed from a whiteboard because the single most common
    misunderstanding of this design is where the manipulation enters. The figure exists to make
    one thing unmistakable: everything upstream of the arm boundary is shared, and the arms differ
    only in what a tool call returns.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, label, colour, fontsize=7.5, alpha=0.15):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=1.1, edgecolor=colour,
                                    facecolor=colour, alpha=alpha))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, linespacing=1.35)

    def arrow(x1, y1, x2, y2, colour=OKABE["grey"], style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=9, linewidth=0.9, color=colour))

    # shared upstream
    box(0.15, 3.5, 1.9, 1.0, "task family\n3 difficulty\nclasses", OKABE["grey"])
    box(2.35, 3.5, 1.9, 1.0, "agent\nepisode loop\n(shared prompt)", OKABE["grey"])

    # the seam
    ax.add_patch(FancyBboxPatch((4.5, 0.35), 2.3, 4.3, boxstyle="round,pad=0.08",
                                linewidth=1.3, edgecolor=OKABE["vermillion"],
                                facecolor="none", linestyle="--"))
    ax.text(5.65, 4.44, "the only seam", ha="center", va="center", fontsize=7.5,
            color=OKABE["vermillion"], style="italic")
    box(4.65, 3.25, 2.0, 0.95, "REAL arm\ninstrumented filesystem\n(no egress)", OKABE["green"])
    box(4.65, 2.05, 2.0, 0.95, "SIM arm\nLLM-fabricated\ntool outputs", OKABE["blue"])
    box(4.65, 0.85, 2.0, 0.95, "scripted arm\ntemplate-fabricated\n(transfer control)", OKABE["purple"])

    arrow(2.05, 4.0, 2.35, 4.0)
    arrow(4.25, 4.0, 4.65, 3.72)
    arrow(4.25, 4.0, 4.65, 2.52)
    arrow(4.25, 4.0, 4.65, 1.32)

    # downstream
    box(7.05, 2.6, 1.5, 1.4, "matched\ntranscript\nprefixes", OKABE["grey"])
    box(7.05, 0.85, 1.5, 1.3, "residual\nstream\nper layer", OKABE["orange"])
    box(8.75, 1.7, 1.1, 1.5, "mean-diff\ndirections\nA C B S\n+ gate", OKABE["vermillion"])

    arrow(6.65, 3.72, 7.05, 3.35)
    arrow(6.65, 2.52, 7.05, 3.15)
    arrow(6.65, 1.32, 7.05, 2.95)
    arrow(7.8, 2.6, 7.8, 2.15)
    arrow(8.55, 1.5, 8.75, 2.1)

    # the action log, which bypasses the probe path entirely
    box(7.05, 4.15, 2.8, 0.65, "action log $\\rightarrow$ scope detector (pure function)",
        OKABE["green"], fontsize=7)
    arrow(6.65, 4.0, 7.05, 4.4)

    fig.tight_layout()
    out = FIGDIR / "architecture.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    _style()
    made = [f for f in (fig_architecture(), fig_layers(), fig_scale(), fig_errors()) if f]
    for f in made:
        print(f"  {f.relative_to(ROOT)}  {f.stat().st_size // 1024} KiB")
    if not made:
        print("  no figures produced -- results/ is empty")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
