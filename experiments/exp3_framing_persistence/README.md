# exp3 — is the transcript a manipulation surface?

**Everything else manipulates credence through the substrate, which is expensive. Does the cheap
route — asserting a conclusion in the agent's own context — work?**

Three conditions per episode, differing only by one appended sentence in which the agent states a
conclusion about its environment. A conclusion rather than an instruction: an instruction tests
compliance, a stated conclusion tests whether the model carries a framing forward as its own. The
model is then asked, in a tool-free turn requiring prose, what it believes — scored by a committed
lexicon over the verdict sentence, with a task step interleaved between probes so persistence is
measured across real intervening work.

**Verdict: P7 SUPPORTED (regeneration)** — the injection takes and is then overwritten by the
model's own framing within a few turns.

```bash
python experiments/exp3_framing_persistence/run.py
```

Stimuli are exp0's frozen REAL episodes, so this shares no data with exp1 and can run alongside it.

Artifacts: [`results/exp3/`](../../results/exp3/)
