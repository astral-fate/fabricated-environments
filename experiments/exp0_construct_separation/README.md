# exp0 — the separability gate

**Are "this is an evaluation" and "this environment is not real" the same linear direction?**

If they are, a suspicion meter for resampling harnesses is redundant: existing evaluation-awareness
tooling already measures it, and the right move is to reuse that tooling rather than build
anything. Everything else in this repository sits behind this question, which is why it is a gate.

**Decision rule, fixed before any activation was extracted:** cross-AUROC between `A_eval` and
`B_empirical` ≥ 0.85 **in both directions** means one construct — report it and stop. Below that in
at least one direction, the constructs are separable and the project proceeds.

**Verdict: SEPARABLE at all three scales**, and the separation strengthens with model size.

```bash
python experiments/exp0_construct_separation/run.py --model Qwen/Qwen3-1.7B --dtype float16
python -m modal run experiments/exp0_construct_separation/modal.py \
    --model Qwen/Qwen3-32B --dtype bfloat16 --gpu A100-80GB
python experiments/exp0_construct_separation/supervise.py -- --seeds 8
```

| File | What it is |
|---|---|
| `run.py` | the experiment: episodes, replays, stimuli, extraction, verdict |
| `modal.py` | the same run on a rented GPU, with provenance guards |
| `supervise.py` | stall and quota supervision for a long collection |

Artifacts: [`results/exp0/`](../../results/exp0/)
