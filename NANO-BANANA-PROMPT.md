# Nano Banana prompt — Figure 5, Track 5 (Realness)

*Paper: "Evaluation-Awareness and Environment-Realness Are Separable Directions"*
Target file: `realness/paper/figures/architecture.png`
Verify with: `python paper/render.py --list`

Every number below was read from the live artifacts on 2026-09-13, not typed from memory. The
regeneration command that produces them is given in the checklist so you can re-check before
shipping.

---

## Read this before generating anything

Same rule as tracks 1–4, and it bites harder here. **Image models render text unreliably, and this
paper's entire discipline is that no number is hand-entered** — `render.py` fails the build if a
bare decimal appears in the manuscript source. Shipping a figure with an invented digit would
break that rule in the one place the automated check cannot see, because the check reads the LaTeX
source and not the pixels inside a PNG.

**So generate this diagram with NO NUMERIC LABELS AT ALL.** The prompt below deliberately contains
none. Ask for boxes, arrows and category names only; add every number yourself in Inkscape or
Figma afterwards, reading them from the checklist at the bottom. The layout is the hard part; the
numbers are the part you cannot afford to have invented.

**There is already a working figure.** `analyze/figures.py::fig_architecture` generates
`architecture.pdf` from code, and the build regenerates it every time. That version cannot drift
and needs no verification. Treat the Nano Banana output as a *presentation* upgrade for slides or
a hero image — and if you do swap it into the manuscript, keep the matplotlib source in the repo
so a reader can regenerate the layout.

**House style, shared with tracks 1–4** (paste this into the prompt):

> Flat vector style on a white background. Technical diagram, not an infographic. Restrained
> palette: charcoal `#2B2B2B` for text and strokes, muted slate blue `#3A6EA5` for actors and data
> flow, warm amber `#D98C2B` for the highlighted result, soft red `#B5453C` for closed or blocked
> states, light grey `#EFEFEF` for panel fills. Thin 1.5pt strokes. Generous white space. No
> gradients, no drop shadows, no 3D, no glossy effects, no photographic elements, no decorative
> background. Clean geometric sans-serif labels (Inter or Helvetica). Every label horizontal and
> legible at small print size. Do not add a title, a legend, an outer border, page numbers, or any
> text other than the labels specified.

---

## The prompt

> [paste house style]
>
> A wide landscape system-architecture diagram, roughly 16:9, flowing left to right in four
> columns. Thin light-grey vertical rules separate the columns. A small bold letter (A, B, C, D)
> sits at the top-left of each column.
>
> **Column A — "One task, one agent".**
> A rounded rectangle in slate-blue outline labelled **"agent episode loop"**, with a small
> head-and-shoulders glyph. Below it a light-grey rounded rectangle labelled **"task family"**
> containing three small stacked pills reading **"solvable"**, **"blocked"** and **"tempting"**.
> Beneath those, in smaller charcoal text, the line **"scope stated in the brief, enforced
> nowhere"**.
>
> **Column B — "The only seam" (the visual centre of the figure).**
> Enclose this entire column in a rounded rectangle drawn with a **dashed soft-red border**,
> labelled at its top in soft red italic **"the only seam"**. Inside it, three rounded rectangles
> stacked vertically, evenly spaced, each the same size:
> - top, in **green-grey outline**, labelled **"REAL arm"** with the sub-label **"instrumented
>   filesystem, no network"**;
> - middle, in **slate blue**, labelled **"SIM arm"** with the sub-label **"LLM fabricates every
>   tool output"**;
> - bottom, in **muted purple**, labelled **"scripted arm"** with the sub-label **"template
>   fabrication, transfer control"**.
>
> Three thin slate-blue arrows fan out from the agent box in Column A into the three arm boxes.
> Above the fan, a short horizontal charcoal bracket spanning the three arrows, labelled
> **"identical task, prompt, turn budget, tool schemas"**.
>
> **Column C — "Matched prefixes".**
> A light-grey rounded rectangle labelled **"transcript prefixes"**. Inside it, draw three short
> horizontal stacks of thin rounded bars suggesting a chat transcript — alternating wide and narrow
> bars — with only the **final bar of each stack filled in amber**. Beside the amber bars, a small
> charcoal caption reading **"read at the final token"**. Below the box, the line **"same actions,
> same position, different bytes"**.
>
> Three arrows, one from each arm box, converge into this single box.
>
> **Column D — "Directions and the gate".**
> Four small rounded rectangles in a two-by-two grid, each labelled with a single letter and a
> short name, in this order: **"A — eval framing"**, **"C — declared realness"**, **"B — empirical
> realness"**, **"S — scripted control"**. Box B is outlined in amber; the other three in charcoal.
> Below the grid, a single wider rounded rectangle in amber labelled **"cross-transfer gate"**.
> Two arrows run into it, one from A and one from B, crossing each other in an X shape to suggest
> each direction being tested on the other's contrast.
>
> **A separate path along the top of the figure.**
> From the agent box in Column A, a **single green-grey arrow** runs along the very top of the
> diagram, above all four columns, ending in a green-grey rounded rectangle at the far right
> labelled **"scope detector"** with the sub-label **"pure function over the action log"**. This
> arrow must visibly bypass Columns B, C and D rather than passing through them. Label the arrow
> itself, in small charcoal text, **"action log"**.

---

## Verification checklist

The prompt above requests **no numbers**, so there is nothing to verify in the generated image
except the labels. If you later overlay numbers by hand, every one must match this table.

Regenerate the source of truth with `python paper/render.py --list` from `realness/`.

| Label you might add | Must equal | Claim key |
|---|---|---|
| `Qwen3-8B` | headline model | `HEADLINE_MODEL` |
| `24 episodes` | clustering unit count | `N_GROUPS` |
| `192 pairs` (A, C) | sentence-swapped set size | `N_PAIRS_A` |
| `96 pairs` (B, S) | substrate-swapped set size | `N_PAIRS_B` |
| `0.992` (A own contrast) | A\_eval held-out AUROC | `AEVAL_AUROC` |
| `0.992` (B own contrast) | B\_empirical held-out AUROC | `BEMP_AUROC` |
| `0.605` (A→B) | cross-AUROC, A onto B | `CROSS_AB` |
| `0.579` (B→A) | cross-AUROC, B onto A | `CROSS_BA` |
| `0.85` (threshold) | pre-registered separability bound | `THRESHOLD` |

**Check the arm labels against the code, not against this file:** the three arms are
`arms/real.py::RealArm`, `arms/sim_stub.py::LLMSimArm` and `arms/sim_stub.py::ScriptedSimArm`. If
a future revision renames one, the figure is wrong and this table will not tell you.

---

## Three things this figure must not do

1. **It must not put the scope detector downstream of the probe path.** The behavioural
   measurement never passes through a model — that is the argument against circularity, and a
   diagram that routes the action log through Column C or D asserts the opposite of what the paper
   claims. The bypass arrow along the top is the whole point of including it.

2. **It must not draw the three arms as alternatives an agent chooses between.** They are
   experimental conditions applied to the *same* recorded action sequence; the agent never sees
   more than one, and the SIM and scripted arms are replays rather than live episodes. Anything
   suggesting a runtime branch — a decision diamond, a switch, a router — misrepresents the design.

3. **It must not show four directions feeding the gate.** Only A and B enter the pre-registered
   decision. C and S are controls that interpret the result and are reported alongside it; drawing
   them into the gate would imply a four-way test that was never pre-registered, and the
   preregistration is the document this paper is accountable to.
