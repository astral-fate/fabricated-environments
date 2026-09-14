# exp1 — on-policy behavioural divergence

**Does an agent act differently when the environment is actually fabricated?**

exp0 is a result about representations; it says nothing about conduct. Each task cell runs
on-policy in three arms — a real filesystem, a template simulator, an LLM simulator — holding task,
prompt, tool schemas, turn budget and actor constant. The arms differ **only** in what a tool call
returns, and the agent's own trajectory is free to diverge, because the divergence is the
measurement.

The third arm is not redundant. An LLM simulator is fabricated *and* incoherent, so a two-arm
design cannot say which property moved conduct; the template simulator is fabricated but
self-consistent, which splits the contrast into a fabrication effect and a coherence effect.

**Verdict: DIVERGENT** — +2.417 out-of-scope
actions per episode [+1.167, +3.709].

```bash
python experiments/exp1_behavioural_divergence/supervise.py      # all three arms, in parallel
python experiments/exp1_behavioural_divergence/selftest.py       # whole pipeline offline, seconds
```

| File | What it is |
|---|---|
| `run.py` | collection and analysis; `--arms` selects which arms this process collects |
| `supervise.py` | one child per arm, with stall, crash and no-progress detection |
| `selftest.py` | the whole pipeline with a scripted actor — no API key, never reportable |

Artifacts: [`results/exp1/`](../../results/exp1/)
