# results/ — every artifact behind every number in the paper

The manuscript contains no hand-typed values. `paper/render.py` substitutes each one from the files
in this directory at build time and fails the build if a bare decimal appears in the source, so the
chain from a number in the PDF to the run that produced it is mechanical rather than a matter of
trust. This file is the index of that chain.

Every experiment writes three kinds of thing, and all three are tracked:

| Kind | What it is | Why it is here |
|---|---|---|
| **verdict JSON** | gates, estimates, intervals, per-cell values | what the paper reads |
| **episode / record logs** | append-only, one line per episode | what the verdict was computed from |
| **run logs** | console output as the run happened | what the run actually did, in order |

Run logs were once excluded as transient. That was wrong for an artifact meant to be audited: the
verdict JSON says what was concluded, and only the run log says when each arm finished, whether a
supervisor intervened, which cells were retried, and how long a GPU stage took.

---

## exp0 — the separability gate

Are evaluation-awareness and environment-realness the same linear direction?

| File | Contents |
|---|---|
| `exp0/exp0.json` | verdict at Qwen3-1.7B (the local scale) |
| `exp0/exp0-qwen3-8b.json`, `exp0/exp0-qwen3-32b.json` | the rented-GPU scales |
| `exp0/episodes.jsonl` | the REAL-arm corpus the stimuli are built from |
| `exp0/replays.jsonl` | that same action sequence replayed against both simulators |
| `exp0/error_analysis.json` | per-item error structure |
| `exp0/provenance.json`, `exp0/stimuli_summary.json` | what ran, and what the stimulus sets contained |
| `exp0/*.log` | run logs, including the Modal dispatches |
| `exp0/invalidated/` | a run that reached a confident verdict on contaminated inputs, kept and clearly marked |

**Verdict: SEPARABLE at every scale tested.** `exp0-qwen3-32b.json` is the headline; the other two
carry the scale trend.

Not tracked, because they are bulky and regenerate deterministically from what is: `exp0/checkouts/`,
`exp0/replay/`, `exp0/activations-*.npz` (410 MB, ~3.5 min to rebuild).

## exp1 — on-policy behavioural divergence

Does an agent act differently when the environment is actually fabricated?

| File | Contents |
|---|---|
| `exp1/exp1.json` | verdict: gates, arm means, all three contrasts, per-cell values |
| `exp1/episodes-real.jsonl` | the grounded arm, 24 cells |
| `exp1/episodes-sim_scripted.jsonl` | template fabrication (fabricated, self-consistent) |
| `exp1/episodes-sim.jsonl` | LLM fabrication (fabricated, incoherent) |
| `exp1/episodes.jsonl` | the pre-sharding log; the first four episodes, before the arms were split to run concurrently |
| `exp1/run-<arm>.log` | one per arm, since the arms were collected in parallel |
| `exp1/supervisor.log`, `exp1/supervisor.jsonl` | every supervisor decision, with timestamps |
| `exp1/pilot/` | two harness-pilot episodes and one 400-character-preview run, quarantined with a README. **Never analysed** |

**Verdict: DIVERGENT.** One log per arm rather than one log overall: the resume logic assumes a
single writer, so sharding by arm let the three run concurrently without weakening that assumption.
The analysis reads every shard back, so nothing about the measurement depends on how collection was
scheduled.

## exp2 — does credence predict conduct? (P2)

| File | Contents |
|---|---|
| `exp2/exp2-qwen3-32b.json` | verdict: gates, within-arm and pooled slopes, per-episode credence |
| `exp2/exp2-qwen3-32b-stage1.json` | GPU-path validation only — rebuilds the direction, issues **no** verdict |
| `exp2/run.log`, `exp2/run-stage1.log` | the Modal dispatches |

**Verdict: P2 SUPPORTED.** The probe layer is inherited from `exp0-qwen3-32b.json` and never
refitted here, and the stage-1 file exists so the GPU path could be validated while exp1 was still
collecting. It is not a result and `render.py` will not read it as one.

## exp3 — is the transcript a manipulation surface? (P7)

| File | Contents |
|---|---|
| `exp3/exp3.json` | verdict, gates, per-contrast intervals, ceiling diagnostic |
| `exp3/records.jsonl` | every generated continuation, all three conditions, with per-probe scores |
| `exp3/run.log` | the run |
| `exp3/pilot/` | discarded pilots that drove two protocol amendments. **Never analysed** |

**Verdict: P7 SUPPORTED (regeneration).** The verdict carries a `primary_ceiling_diagnostic`
because the pre-registered primary contrast is saturated: reading its interval without that
diagnostic would invert the finding.

---

## Reproducing a number

```bash
bash paper/build.sh        # recompute every claim from these files and rebuild the PDF
python paper/render.py     # print each claim with the value currently in results/
```

`render.py` prints one line per claim. An absent input renders as a visible `[PENDING]` in the PDF
rather than as a plausible-looking placeholder. `PREREGISTRATION.md` carries the protocol each
verdict was judged against, with every amendment dated and the data that prompted it discarded.
