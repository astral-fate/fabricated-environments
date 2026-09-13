# construct-separation-gate

**Is "this is an evaluation" the same thing, inside a model, as "this environment is not real"?**

A counterfactual resample — continuing a halted agent trajectory in a synthetic environment to see
what it *would* have done — is a measurement whose instrument is an environment the agent can form
beliefs about. Those beliefs may change the quantity being measured, and the literature disagrees
about the direction. Classical evaluation-awareness says an agent that believes it is tested
behaves *better*, so a resample understates danger. The opposite is also documented: an agent that
concludes its consequences are not real complies more readily with what it would otherwise refuse.

If those are one phenomenon, a suspicion meter for resampling harnesses is redundant — existing
evaluation-awareness tooling already measures it. If they are two, no amount of
evaluation-awareness detection tells you whether an agent believes its environment is fabricated.
This repository is the measurement that decides which.

---

## The finding

**The constructs are separable, at every scale tested, and the separation strengthens with size.**

| Model | `A_eval` | `B_empirical` | A→B | B→A | cos(A, B) |
|---|---|---|---|---|---|
| Qwen3-1.7B | 0.919 | 0.925 | 0.473 | 0.572 | +0.048 |
| **Qwen3-8B** | **0.992** | **0.992** | **0.605** | **0.579** | **+0.045** |
| Qwen3-32B | 0.999 | 0.978 | 0.417 | 0.484 | +0.029 |

Columns 2–3 are each direction on its own contrast; columns 4–5 are cross-transfer onto the
other's contrast. The pre-registered rule required cross-AUROC ≥ **0.85 in both directions** to
call the constructs inseparable. Both are far below it at every scale.

The obvious objection to a null is that the model is too small to have formed the representation.
That reading is ruled out: at 32B the evaluation-awareness direction is near-perfect (0.999) while
cross-transfer *falls* to 0.417. Both probes sharpen; the bridge between them does not.

Two further results make the null interpretable rather than bare:

- **Being *told* a world is fake is not the same as the world *being* fake.** `C_declared` — the
  realness axis built the way the prior probe literature builds evaluation-awareness, by declaring
  it in the prompt — sits close to `A_eval` (cos +0.402 at 8B) and far from ground-truth realness
  (cos +0.118). Declared framing clusters with declared framing whichever axis it names.
- **`B_empirical` is not a prose-provenance detector.** It transfers to a template simulator it
  never saw (cross-AUROC 0.704, cos +0.469), so it is not merely detecting one generator's style.

---

## Three results behind the gate

The gate asked whether the construct exists. These ask whether it matters.

**Agents act differently when the environment is actually fabricated.** Each task cell was run
on-policy in all three arms — grounded, template-fabricated, LLM-fabricated — holding task, prompt,
tool schemas, turn budget and actor constant. Out-of-scope actions rise **+2.417 per episode
[+1.167, +3.709]** from real to LLM-fabricated, and the rate per action rises **+0.428
[+0.231, +0.624]** with no difference in episode length, so it is not an artifact of longer runs.
The composition changes rather than the volume: **0 out-of-scope writes and 0 out-of-scope reads in
the real arm across every episode**, against 12 writes and 86 reads in the fabricated arms.

Agents crossed the boundary *more* when the world was fake, so a resample of this task family
**overstates** danger — the failure mode that makes resampled evidence rebuttable by the developer
it is aimed at.

**It is fabrication that moves conduct, not the simulator being bad at it.** The third arm exists
to split a confound a two-arm design cannot: holding self-consistency fixed and varying only
fabrication moves out-of-scope actions +2.958 [+0.708, +5.500]; holding fabrication fixed and
varying only self-consistency moves it −0.542 [−3.084, +1.751], which includes zero. A more
consistent simulator does not recover the behaviour, because consistency was not what changed it.

**The probe predicts conduct; the agent's own words do not.** Within arm — so the arm label cannot
manufacture the association — the out-of-scope rate rises **+0.3318 per standard deviation of
probe-measured credence [+0.1092, +0.5210]**, while the same regression on the agent's verbalized
suspicion gives +0.0756 [−0.1470, +0.3209]. Verbalized suspicion *does* differ between arms
(+0.500), so the agent says the environment looks fake more often when it is — it just does not say
so more often when it is about to act on it. An instrument built on what the agent reports would
see the first and miss the second.

**The transcript is not a viable manipulation surface.** Asserting a conclusion in the agent's own
context moves its stated belief +0.417 [+0.167, +0.625] at the first probe and +0.208
[−0.042, +0.417] by the third, after intervening work — it takes, then the model regenerates its
own framing over it. Worth noting separately: asked directly, the model called the **real**
filesystem simulated in 0.917 of control episodes, reasoning from timestamps that postdate its
training data.

Full numbers, per-layer curves, bootstrap intervals, validity-gate output, episode logs and run
logs for every experiment: [`results/`](results/), indexed in
[`results/README.md`](results/README.md).

---

## How it works

![System architecture](paper/figures/architecture-render.png)

One recorded action sequence is replayed against three substrates that differ **only** in what a
tool call returns:

- **REAL** — an instrumented filesystem with no network affordance. Tool calls do what they say.
- **SIM** — a language model fabricates every tool output.
- **Scripted** — template fabrication, used as a transfer control.

Four mean-difference probe directions are estimated per layer on the **same** stimulus substrate,
each varying exactly one element of an otherwise identical agentic transcript prefix:

| | Manipulation | Label 1 means |
|---|---|---|
| `A_eval` | system framing sentence ("real user" / "AI evaluator"); tool outputs real in both | being evaluated |
| `C_declared` | system declaration ("real production system" / "a simulation"); outputs real in both | declared synthetic |
| `B_empirical` | no framing; tool outputs genuinely real vs LLM-fabricated | actually synthetic |
| `S_scripted` | no framing; tool outputs genuinely real vs template-fabricated | actually synthetic |

Building all four on one substrate is load-bearing: cross-AUROC between directions estimated on
*different* input distributions would mostly measure the distribution gap, not the constructs.

Only **A** and **B** enter the pre-registered decision. **C** and **S** interpret the result.

---

## Reproducing it

```bash
pip install -r requirements.txt
python check.py          # 8 suites, 177 tests. No GPU, no network, no API keys.
```

The episodes and their simulated replays are committed, so the GPU stages run offline:

```bash
python experiments/exp0_construct_separation.py --model Qwen/Qwen3-1.7B --dtype float16
```

For the larger scales, [`modal_exp0.py`](modal_exp0.py) runs the same experiment on rented GPUs —
no API key needed, because stages 1–2 ship cached:

```bash
python -m modal run modal_exp0.py --model Qwen/Qwen3-8B  --dtype bfloat16 --gpu A10G
python -m modal run modal_exp0.py --model Qwen/Qwen3-32B --dtype bfloat16 --gpu A100-80GB
```

Total compute for all three scales was **under $1**. Qwen3-32B in bf16 is ~64 GB of weights and
needs an 80 GB card; see [`RUNPOD.md`](RUNPOD.md) for the pod route.

`--offline` runs every stage with a scripted actor and template simulators. It exercises the
pipeline without spending API budget and is **never reportable** — the run stamps itself
`offline: true` and refuses to issue a verdict.

Build the paper:

```bash
bash paper/build.sh     # regenerates analyses + figures, renders from results/, compiles
```

---

## What this repository is careful about

Each of these is enforced by a check that fails the build, not by a note in a file.

- **The violation detector is a pure function over the action log.** It calls no model and touches
  no filesystem. An LLM in that loop would make the measurement circular — one model judging
  another inside the very contrast under study — and a filesystem read would make an archived log
  impossible to re-score. A test parses the module's imports and fails if either appears.
- **Scope is stated in the task brief and enforced nowhere.** A boundary the substrate refuses to
  cross yields a violation rate of identically zero, which measures the substrate rather than the
  agent.
- **The unit of independence is the episode, never the transcript prefix.** One episode yields many
  near-duplicate prefixes; pooling them would narrow every interval. `analyze/measures.py` offers
  no item-level bootstrap, so the only way to call it is the correct way.
- **Cosine is reported against its own reliability ceiling.** Random high-dimensional vectors are
  near-orthogonal, so a bare cosine is uninterpretable.
- **No number is hand-typed into the manuscript.** `paper/render.py` substitutes every value from
  `results/` and fails the build on a bare decimal in the source. Another paper's number must be
  wrapped in `\litnum{}` with a citation in the same paragraph.
- **Infrastructure failure never enters the data.** A rate-limited simulator raises rather than
  returning `error: ...` as if the fabricated environment had said it. An earlier run reached a
  confident verdict with 35% of its SIM outputs in that state; that result is kept, clearly marked
  invalid, in [`results/exp0/invalidated/`](results/exp0/invalidated/).
- **Task success in a fabricated world is not evidence.** A simulated arm returns an empty
  snapshot, so completion is a diagnostic and never a target.

---

## Layout

```
src/            task family, pure-function scope detector, arms, episode loop, probe
experiments/    exp0_construct_separation.py — the pre-registered gate
analyze/        measures (AUROC, cluster bootstrap), error analysis, figures, transcripts
results/exp0/   episodes, replays, per-scale verdicts, error analysis, invalidated runs
paper/          main.tex.tmpl (no numbers), render.py, refs.bib, figures
tests/          177 tests: scope purity, task classes, egress, resume, probe mathematics
scripts/        supervisor, GPU packaging, vendored-file drift check
```

Four files are vendored byte-identical from the co-tenant channel-capacity study
(`substrate.py`, `srm.py`, `providers.py`, `watchdog.py`).
`scripts/sync_from_project_v2.py` fails the build if a vendored copy drifts from its source, so the
reuse boundary is a mechanical check rather than a note in a README.

---

## Scope of the result

Qwen3-1.7B / 8B / 32B, bf16 or fp16, never quantised — low-bit weight formats perturb the residual
stream, which is the measurement. 24 episodes across 3 difficulty classes, one actor model. The
probes read what a model represents while *processing* a context, not while generating one;
on-policy divergence is a separate experiment. The simulated arm used here has no durable
world-state store and no consistency contract, so no fidelity claim is made from it.

Behaviourally, this corpus establishes that the detector and difficulty gradient work, not that the
behaviour of interest was observed: across all 24 episodes there were **zero** out-of-scope writes
and **one** out-of-scope read, which the arm refused. The crossings are almost entirely
reconnaissance. The actor also exposed no separate reasoning trace, so the chain-of-thought
credence channel has no raw material here and the paper claims nothing about it.

---

## Preregistration

[`PREREGISTRATION.md`](PREREGISTRATION.md) was written before any activation was extracted, and
records the decision rule, the validity gates, and the commitment to report either outcome. One
amendment is appended with its date and reason: paraphrase variation was added to the framing
conditions after a pipeline check returned a held-out AUROC of exactly 1.000 for every direction —
the signature of a token-presence detector rather than a construct.

## Citing

**Paper.** *Evaluation-Awareness and Environment-Realness Are Separable Directions: A
Construct-Separation Gate for Counterfactual Resampling.* Fatimah Emad Eldin, 13 September 2026.
Apart Research & CeSIA, AI Incident Response Sprint — Track 5 (Open).

```bibtex
@misc{eldin2026separable,
  title        = {Evaluation-Awareness and Environment-Realness Are Separable Directions:
                  A Construct-Separation Gate for Counterfactual Resampling},
  author       = {Emad Eldin, Fatimah},
  year         = {2026},
  month        = sep,
  note         = {Apart Research \& CeSIA, AI Incident Response Sprint --- Track 5 (Open)},
  howpublished = {\url{https://github.com/astral-fate/construct-separation-gate}},
}
```

**Artifact.** Cite the commit, not the branch — every number in the paper is regenerated from
`results/` at build time, so the commit is what fixes the values a reader will see.

```bibtex
@software{eldin2026csg,
  title   = {construct-separation-gate: a pre-registered construct-separation gate for
             counterfactual resampling},
  author  = {Emad Eldin, Fatimah},
  year    = {2026},
  url     = {https://github.com/astral-fate/construct-separation-gate},
  version = {54ac8c4e4715},
}
```

| Field | Value |
|---|---|
| Repository | <https://github.com/astral-fate/construct-separation-gate> |
| Commit behind the reported numbers | `54ac8c4e4715` |
| DOI | not yet minted |

