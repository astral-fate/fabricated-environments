# Preregistration — realness

Written before any experiment in this repository was run. Predictions P1–P7 are carried verbatim
from the project brief (`project-v2/docs/05-second-paper-brief.md` §6); the exp0 protocol below
is the operationalisation and was fixed before the first activation was extracted.

Amendments are appended at the bottom with a date and a reason, never edited in place.

---

## The question

> When an agent's environment is synthetic, does its belief about that fact change what it does —
> and can a counterfactual resample be trusted if it does?

---

## Predictions

- **P1.** Pure-LLM SIM shows higher measured credence-of-fakeness than a hybrid at every horizon,
  and the gap widens with horizon length.
- **P2 (load-bearing).** In a multi-step agentic trajectory, probe-measured realness credence
  predicts scope-violation rate with an interval excluding zero, while verbalized awareness on the
  same trajectories does not.
- **P3.** Heterogeneous: variance across branch points within one model exceeds the population
  mean effect.
- **P4.** The dominant leak channel is consistency under adversarial repeat query, not surface
  realism. Sharpened: for a fixed simulation budget, spending it on the world-state store beats
  spending it on output quality.
- **P5′.** Eval-awareness and environment-realness are **separable** linear directions
  (cross-AUROC < 0.85 in at least one direction). *P5 as originally framed — that representation
  and verbalization are decoupled — is withdrawn; Heidari et al. established it.*
- **P6.** Minimum-suspicion substrate ≠ minimum-divergence substrate.
- **P7 (methodological).** Zhuang & Aranguri found 66% of truncated CoT continuations regenerate
  the same framing within 1–4 sentences. Prediction: the same occurs in the agentic setting, so
  transcript-side manipulation is not viable and credence must be manipulated through the
  substrate. If P7 holds and no substrate-side manipulation is found, RQ1 is not answerable by
  this design — report that rather than working around it.

---

## exp0 — the gate

**Question.** Are evaluation-awareness and environment-realness the same linear direction in a
model's residual stream? If they are, there is no separate construct, the suspicion meter
collapses into existing eval-awareness tooling, and the project should be rescoped or abandoned.

### Directions estimated

All four are mean-difference probes on the residual stream, estimated per layer, on the **same**
substrate of agentic transcript prefixes. Each varies exactly one element:

| | Name | Manipulation | Label 1 |
|---|---|---|---|
| A | `A_eval` | system framing sentence ("real user" / "AI evaluator"); tool outputs REAL in both | being evaluated |
| C | `C_declared` | system declaration ("real production system" / "a simulation"); tool outputs REAL in both | declared synthetic |
| B | `B_empirical` | no framing; tool outputs genuine vs LLM-fabricated | actually synthetic |
| S | `S_scripted` | no framing; tool outputs genuine vs template-fabricated | actually synthetic |

`A_chat` — the original short-prompt construction — is estimated as a port check and is **not**
part of the decision rule.

Building all four on one stimulus substrate is deliberate: cross-AUROC between directions
estimated on different input distributions would mostly measure the distribution gap.

### Pooling

The gate is stated on the **final prompt token** (`last`). Mean-over-tokens is reported alongside
and is not part of the decision rule.

### Decision rule — fixed in advance

> If cross-AUROC between **A** and **B** is **≥ 0.85 in both directions**, the constructs are not
> separable. That is the finding; report it and stop.
> If cross-AUROC is **< 0.85 in at least one direction**, the constructs are separable and the
> project proceeds.

Cross-AUROC is evaluated at the layer that maximises each direction's own held-out AUROC, with a
cluster-bootstrap 95% CI. A↔C and B↔S are secondary and do not change the verdict.

### Validity gates — checked before the verdict is reported

The gate result is reportable **only if all three pass**. If any fails, the pipeline is reported
as broken and no exp0 verdict is issued.

1. **Negative control.** Labels shuffled across items; held-out AUROC averaged over ≥25 shuffles
   must sit within 0.05 of 0.50.
2. **Reproduction floor.** `A_eval` must reach held-out AUROC ≥ 0.70 at some layer, matching the
   floor Heidari et al. report ("best AUROC ≥0.7"). Below it, the eval-awareness direction has
   not been recovered and comparing anything to it is meaningless.
3. **No leakage.** No episode id appears in both the train and test split of any direction.

### Unit of independence

The **episode**, never the transcript prefix. One 20-turn episode yields many near-duplicate
prefixes; pooling them as independent would multiply apparent n and narrow every interval. All
train/test splits are by episode, and all confidence intervals come from a cluster bootstrap that
resamples episodes with replacement.

### Analysis decisions fixed in advance

- Direction estimator: difference-of-means, L2-normalised (matching Heidari et al. and
  Zhuang & Aranguri, for comparability rather than for accuracy).
- Train/test split: 50% of episodes, by episode, seed 0.
- Bootstrap: 2000 resamples, percentile interval, α = 0.05.
- The direction is **fixed** inside the bootstrap, not refitted: the pre-registered threshold is
  stated about the held-out AUROC of the direction actually learned.
- Cosine between directions is reported alongside each direction's **split-half reliability**, and
  normalised by `sqrt(rel_A · rel_B)`. A bare cosine is uninterpretable because random
  high-dimensional vectors are near-orthogonal, so any positive value looks meaningful.

### Known limitations, stated in advance

- **Model scale.** The local GPU is a 6 GiB RTX 2060. Qwen3-32B — which would match
  Zhuang & Aranguri and Heidari et al. directly — cannot run on it, and neither can Qwen3-8B
  unquantized. exp0 therefore runs first at **Qwen3-1.7B in fp16**, below Heidari et al.'s 7B–49B
  range. The experiment is scale-parameterised and packaged for a rented GPU; a result at 1.7B is
  reported as a result at 1.7B and nothing more.
- **Off-policy prefixes.** exp0 reads what a model represents while *processing* a context, not
  while generating one. Matched prefixes require this. On-policy divergence is exp1's question.
- **Action-generating model differs from the probed model.** Action sequences come from a hosted
  Groq model; activations are read from the local open-weight model. The generator is held
  constant across all conditions, so it is a nuisance variable rather than a confound, but the two
  are not the same model and the paper says so.
- **The SIM arm used here is a stub.** No `world_state.py`, no consistency contract. exp0 never
  issues the same query twice, so episode-level consistency cannot affect its answer — but this is
  explicitly not the SIM arm of §4, and no fidelity claim is made from it.

---

## Amendments

### 2026-09-13 — paraphrase split added to the A and C constructions

**What changed.** `A_eval` and `C_declared` originally used a single fixed phrasing per pole (the
wording quoted from Heidari et al., and its realness analogue). They now draw from **six
paraphrases per pole**, and the train/test split is disjoint on **two** axes: episodes *and*
paraphrase pool (train uses variants 0–2, test uses 3–5).

**Why.** A pipeline-validation run — offline, with scripted trajectories, never reportable —
returned a held-out AUROC of **exactly 1.000 for every direction**, against the ≥0.70 Heidari
et al. report. A perfect score off one fixed string is the signature of a *token-presence
detector*: the direction can encode "the context contained the word 'evaluator'" and score
perfectly without representing anything about being evaluated.

This is not a cosmetic concern, and it threatens the gate in a specific direction. If A is a
detector for one phrasing and B is a detector for one simulator's output style, the two are
unrelated for reasons that have nothing to do with the constructs. Cross-AUROC would land near
chance and exp0 would report **"separable"** on an artifact — the permissive verdict, reached for
the wrong reason. The amendment makes the null harder to obtain spuriously, not easier.

**What did not change.** The decision rule, the threshold (0.85 in both directions), the validity
gates, the unit of independence, the estimator, and the bootstrap are all unchanged. No result had
been produced when this amendment was made: the only run that had executed was the offline
pipeline check, which is stamped `offline: true` and issues no verdict.

**Scope.** `B_empirical` and `S_scripted` have no paraphrase axis — their manipulation is the tool
outputs themselves, not a sentence — so only the episode split applies to them. This asymmetry is
inherent to the constructs rather than a choice.
