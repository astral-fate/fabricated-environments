# exp2 — does credence predict conduct?

**Does probe-measured realness credence predict scope-violation rate, where verbalized awareness
on the same trajectories does not?**

The realness direction is inherited from exp0 at the layer it selected and **never refitted here**
— choosing a layer to maximise an association with the outcome would be fitting the hypothesis. The
estimand is centred **within arm**, because credence runs higher in the fabricated arms and conduct
may also differ by arm, so a pooled association could be produced entirely by the arm label.

**Verdict: P2 SUPPORTED** — +0.3318 per standard
deviation of credence [+0.1092, +0.5210], while
verbalized suspicion spans zero.

```bash
python -m modal run experiments/exp2_credence_predicts_behaviour/modal.py
python -m modal run experiments/exp2_credence_predicts_behaviour/modal.py --stage1-only
```

`--stage1-only` validates the GPU path — weights, sharding, inherited layer, reproduction gate —
without needing exp1 to have finished. It issues no verdict, and nothing reads it as one.

| File | What it is |
|---|---|
| `run.py` | rebuild the direction, score exp1's trajectories, apply the P2 rule |
| `modal.py` | the GPU job; 32B sharded across four A10Gs |

Artifacts: [`results/exp2/`](../../results/exp2/)
