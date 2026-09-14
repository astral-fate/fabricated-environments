# Running exp0 at 8B and 32B on a rented GPU

The 1.7B result is in `results/exp0/exp0.json`. This is how to test whether it survives into
Heidari et al.'s 7B–49B range, which is the one scope limitation the finding currently carries.

`project-v2/colab/RUNPOD.md` covers the pod mechanics for paper 1 and still applies; this file
records only what differs for exp0.

## What makes this run cheap and safe

**No API key. No network. No secrets leave this machine.** Stages 1 and 2 — generating the 24
episodes and their simulated replays — are already done and ship inside the bundle as append-only
caches keyed by episode id. The pod re-runs only stages 3–6: stimulus construction, residual-stream
extraction, and analysis. All GPU and arithmetic.

That is not merely convenient, it is what makes the comparison mean anything. The 8B and 32B runs
read the **same 24 trajectories** the 1.7B run read, so any difference between scales is a
difference in the model rather than in freshly sampled data. `package_for_colab.py` refuses to
build a bundle containing anything key-shaped, and excludes `.env` by name.

## Which GPU

| GPU | VRAM | ~Community | Fits | Notes |
|---|---|---|---|---|
| **RTX 4090** | 24 GB | ~$0.34/hr | 8B bf16 comfortably; 32B **no** | best value for the 8B point |
| **A40 / L40S** | 48 GB | ~$0.4–0.9/hr | 32B bf16 (~64 GB weights) **no** | still short for 32B in bf16 |
| **A100 80GB** | 80 GB | ~$1.39/hr | 32B bf16 (~64 GB) **yes** | the only single-card 32B option |

Qwen3-32B in bf16 is roughly 64 GB of weights, so **32B needs an 80 GB card**. If that is more
than the result is worth, run 8B on a 4090 and report 1.7B → 8B: 8B is already inside Heidari
et al.'s stated range, which is what the limitation is about.

Do **not** reach for 4-bit to squeeze 32B onto a smaller card. NF4 perturbs the residual stream,
and the residual stream is the entire measurement — a cheap 32B number computed that way is worth
less than an honest 8B one.

## Steps

1. Deploy a pod (PyTorch template, so torch matches the driver). Attach a network volume at
   `/workspace` or a destroyed pod keeps nothing.
2. Upload `realness-code.zip` (from `python scripts/package_for_colab.py`) — about 165 KiB.
3. Run:

```bash
unzip realness-code.zip -d realness && cd realness
pip install -r requirements.txt
python scripts/check.py                      # 177 tests; no GPU, no network, no keys

python experiments/exp0_construct_separation/run.py \
    --model Qwen/Qwen3-8B  --dtype bfloat16 --batch-size 2
cp results/exp0/exp0.json results/exp0/exp0-8b.json

# only on an 80 GB card
python experiments/exp0_construct_separation/run.py \
    --model Qwen/Qwen3-32B --dtype bfloat16 --batch-size 1
cp results/exp0/exp0.json results/exp0/exp0-32b.json
```

4. Bring back the two `exp0-*.json` files. Leave the activation cache on the pod: it is
   `~0.4 GB` at 1.7B and scales with hidden size, and it regenerates in minutes.

## Check these before believing a number

The experiment enforces its own validity gates and refuses a verdict when they fail, but two
fields are worth reading with your own eyes:

- `"hardware": {"quantised": false}` — if this says `true`, the residual stream was quantised and
  the run should be repeated. A 4090 reports ~23.6 GiB, just under the 24 GiB auto-threshold, and
  would silently select 4-bit; that is why `--dtype bfloat16` is passed explicitly.
- `"n_groups": 24` on every direction — confirms the same 24 episodes, so scale is the only thing
  that changed.

The gates themselves, in `"verdict": {"checks": ...}`:
`negative_control` (shuffled labels within 0.05 of chance), `reproduction_floor`
(`A_eval` ≥ 0.70, Heidari et al.'s reported floor), `no_leakage` (no episode in both splits).

## What the comparison is for

At 1.7B the constructs are separable: cross-AUROC A→B 0.473, B→A 0.572, against a pre-registered
threshold of 0.85. The open question is whether that survives scale — a larger model might
represent "this is a test" and "this world is fake" in a shared direction that a 1.7B model has
not yet tied together, in which case the gate's answer changes and the project should rescope.

So the 8B/32B run is a real test of the finding, not a confirmation exercise. If cross-AUROC
climbs above 0.85 in both directions at 8B, **that** is the result, and it is reported as such.
