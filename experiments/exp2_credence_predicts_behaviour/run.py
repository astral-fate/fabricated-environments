"""exp2 -- does probe-measured realness credence predict conduct? P2, the load-bearing prediction.

    python experiments/exp2_credence_predicts_behaviour.py --model Qwen/Qwen3-32B --dtype bfloat16

exp1 asks whether behaviour differs between substrates. P2 asks the finer question: within a
substrate, do the episodes where the model's realness credence runs higher also show more boundary
probing -- and does the agent's own verbalized suspicion predict it too, or not?

    P2. In a multi-step agentic trajectory, probe-measured realness credence predicts
        scope-violation rate with an interval excluding zero, while verbalized awareness on the
        same trajectories does not.

The protocol is fixed in `PREREGISTRATION.md` under "exp2" and is not re-derived here. Two things
about it are load-bearing and are enforced below rather than described:

**The layer is inherited, never fitted.** `B_empirical`'s best layer comes from the exp0 artifact
for this same model. Choosing a layer to maximise an association with the outcome would be fitting
the hypothesis, and the resulting number would mean nothing.

**The primary estimate is centred within arm.** Credence is expected to be higher in the fabricated
arms, and conduct may also differ by arm, so a pooled association can be produced entirely by the
arm label. Centring removes it. The pooled estimate is reported too, labelled as confounded.

Stages
------
1. direction   rebuild B_empirical from the exp0 stimuli at the inherited layer; reproduce its
               held-out AUROC or stop
2. credence    score every exp1 trajectory prefix; mean per episode
3. gates       reproduction, on-policy transfer (paired within cell), inherited layer, actor
               identity, no quantisation
4. verdict     the preregistered P2 rule, applied only if every gate passes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analyze"))

import behaviour  # noqa: E402
import credence_link as cl  # noqa: E402
from agent import dedupe_episodes, load_episodes  # noqa: E402
from credence.probe import HiddenStateExtractor, pair_activations  # noqa: E402
from credence.stimuli import _boundaries, build_pair_sets, render  # noqa: E402
from measures import (Clustered, auroc, mean_difference_direction,  # noqa: E402
                      project, split_by_group)
from task import build_task  # noqa: E402

EXP0 = ROOT / "results" / "exp0"
EXP1 = ROOT / "results" / "exp1"
OUT = ROOT / "results" / "exp2"

#: Preregistered.
ALPHA = 0.05
N_BOOT = 2000
BOOT_SEED = 0
SPLIT_SEED = 0
SPLIT_FRAC = 0.5
POOLING = "last"            #: exp0's primary pooling, inherited
DIRECTION = "B_empirical"   #: the realness direction; A_eval is not the construct under test


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- trajectories

def trajectory(ep: dict) -> tuple[list[dict], list[str], list[str]]:
    """Recover (actions, tool outputs, assistant texts) in execution order from one episode.

    `submit` is dropped: it is episode control, not an environment affordance, and the arm never
    saw it -- so including it would misalign the outputs against `ep["actions"]`, which is the
    sequence the scope detector scored.
    """
    outputs: list[str] = []
    texts: list[str] = []
    for turn in ep.get("turns", []):
        text = str(turn.get("text") or "")
        for call, res in zip(turn.get("calls", []), turn.get("results", [])):
            if call.get("name") == "submit":
                continue
            outputs.append(str(res.get("preview", "")))
            texts.append(text)
            text = ""     # the turn's prose belongs to its first call only
    return list(ep.get("actions", [])), outputs, texts


def episode_contexts(ep: dict, *, max_prefixes: int) -> list[list[dict[str, str]]]:
    """The transcript prefixes whose activations are read, in the same shape exp0 used."""
    actions, outputs, texts = trajectory(ep)
    n = min(len(actions), len(outputs))
    if n == 0:
        return []
    brief = build_task(ep["task_kind"], int(ep["seed"])).brief
    return [render(brief, actions, outputs, upto, assistant_texts=texts)
            for upto in _boundaries(n, max_prefixes)]


# --------------------------------------------------------------------------- stage 1

def actor_matches_probe(actor_alias: str, probed_model: str) -> dict:
    """Gate. The agent whose conduct was measured must be the model being probed.

    This is what lets a credence score be read as *that model's own state* rather than as a second
    model's opinion about a transcript. exp0 could not make the claim -- its actor was a different
    model -- and the whole point of exp1's actor choice was to close that gap, so it is asserted
    rather than assumed. Compared on the provider-stripped, punctuation-stripped model id, because
    the same weights are spelled `openrouter:qwen/qwen3-32b` by the API and `Qwen/Qwen3-32B` by the
    checkpoint.
    """
    def norm(s: str) -> str:
        return "".join(ch for ch in str(s).split(":")[-1].lower() if ch.isalnum())

    a, p = norm(actor_alias), norm(probed_model)
    return {"actor": actor_alias, "probed": probed_model,
            "actor_normalised": a, "probed_normalised": p, "pass": a == p}


def inherited_layer(artifact: Path) -> tuple[int, float, dict]:
    """The layer and held-out AUROC exp0 reported for this direction, on this model."""
    d = json.loads(artifact.read_text(encoding="utf-8"))
    block = d["pooling"][POOLING]["directions"][DIRECTION]
    return int(block["best_layer"]), float(block["best_auroc"]), d


def rebuild_direction(ex: HiddenStateExtractor, layer: int, batch_size: int,
                      artifact: dict) -> tuple[np.ndarray, float, dict]:
    """Re-derive B_empirical from the exp0 replay records, at the inherited layer.

    Split by EPISODE, as exp0 did: a direction trained and tested on prefixes of the same episode
    would score its own near-duplicates.

    **The stimulus construction is inherited too, not just the layer.** exp0 ran with
    `--max-prefixes 4`, while `build_pair_sets` defaults to 6; taking the default silently built a
    143-pair set where exp0 had 96, which is a different stimulus distribution wearing the same
    name. The reproduction gate would then be comparing a held-out AUROC against a number computed
    on other data. The count is read from the artifact and the result is asserted against the
    artifact's own `n_pairs` and `n_groups`, so a divergence stops the run instead of being
    absorbed into the tolerance.
    """
    records = [json.loads(line) for line in
               (EXP0 / "replays.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    max_prefixes = int(artifact["config"]["max_prefixes"])
    pair_sets = build_pair_sets(records, max_prefixes_per_episode=max_prefixes)
    pairs = pair_sets[DIRECTION].pairs
    expected = artifact["pooling"][POOLING]["directions"][DIRECTION]
    if len(pairs) != int(expected["n_pairs"]):
        raise SystemExit(
            f"rebuilt {DIRECTION} has {len(pairs)} pairs but exp0 recorded "
            f"{expected['n_pairs']} at max_prefixes={max_prefixes}. The stimulus set is not the "
            f"one the inherited layer and AUROC were computed on; refusing to reproduce against "
            f"a different distribution.")
    log(f"stage 1: {len(pairs)} {DIRECTION} pairs from {len(records)} replayed episodes "
        f"(max_prefixes={max_prefixes}, inherited)")

    # Only the inherited layer is extracted. The second axis of the result therefore indexes the
    # requested layers, so the single layer sits at position 0 -- see `HiddenStateExtractor.
    # extract`. At 32B this is the difference between a few hundred megabytes and several
    # gigabytes per batch on a card that is already holding a weight shard.
    got = pair_activations(ex, pairs, batch_size=batch_size, progress=lambda m: log(m),
                           layers=[layer])
    x = got["acts"][POOLING][:, 0, :]
    data = Clustered(x, got["y"], got["groups"])
    train, test = split_by_group(data, frac=SPLIT_FRAC, seed=SPLIT_SEED)

    direction = mean_difference_direction(train.x[train.y == 1], train.x[train.y == 0])
    held_out = auroc(project(test.x, direction).tolist(), test.y.tolist())
    if len(data.unique_groups) != int(expected["n_groups"]):
        raise SystemExit(f"rebuilt {DIRECTION} spans {len(data.unique_groups)} episodes but exp0 "
                         f"recorded {expected['n_groups']}")
    info = {"n_pairs": len(pairs), "n_groups": int(len(data.unique_groups)),
            "max_prefixes_inherited": max_prefixes,
            "n_train_groups": int(len(train.unique_groups)),
            "n_test_groups": int(len(test.unique_groups))}
    return direction, held_out, info


# --------------------------------------------------------------------------- stage 2

def score_episodes(ex: HiddenStateExtractor, cells, direction: np.ndarray, layer: int, *,
                   batch_size: int, max_prefixes: int) -> dict[str, dict]:
    """Per-episode credence: the mean projection over that episode's prefixes."""
    scores: dict[str, dict] = {}
    total = sum(len(c.eps) for c in cells)
    done = 0
    for cell in cells:
        for arm, ep in cell.eps.items():
            contexts = episode_contexts(ep, max_prefixes=max_prefixes)
            done += 1
            if not contexts:
                continue
            acts = ex.extract(contexts, batch_size=batch_size,
                              layers=[layer])[POOLING][:, 0, :]
            proj = project(acts, direction)
            scores[ep["episode_id"]] = {
                "task_id": cell.task_id, "arm": arm,
                "mean": float(np.mean(proj)), "final": float(proj[-1]),
                "n_prefixes": int(len(proj)),
            }
            if done % 10 == 0 or done == total:
                log(f"  scored {done}/{total} episodes")
    return scores


# --------------------------------------------------------------------------- stages 3-4

def analyse(cells, scores: dict[str, dict], gates: dict, *, pooling_key: str = "mean") -> dict:
    rows = []
    for cell in cells:
        for arm, ep in cell.eps.items():
            s = scores.get(ep["episode_id"])
            if s is None:
                continue
            rows.append({
                "task_id": cell.task_id, "arm": arm, "kind": cell.kind,
                "credence": s[pooling_key],
                "oos_rate": behaviour.oos_rate(ep),
                "oos_count": behaviour.oos_count(ep),
                "suspicion": behaviour.suspicion_flag(ep),
                "fabricated": 0 if arm == "real" else 1,
            })

    cred = cl.standardise([r["credence"] for r in rows])
    outcome = [r["oos_rate"] for r in rows]
    susp = [r["suspicion"] for r in rows]
    groups = [r["task_id"] for r in rows]
    arms = [r["arm"] for r in rows]

    transfer = cl.transfer_gate(cred, [r["fabricated"] for r in rows], groups,
                                n_boot=N_BOOT, seed=BOOT_SEED)
    gates["on_policy_transfer"] = {**transfer}

    results = {
        "credence_within_arm": cl.association(cred, outcome, groups, arms=arms,
                                              n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED),
        "credence_pooled": cl.association(cred, outcome, groups,
                                          n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED),
        "verbalized_within_arm": cl.association(susp, outcome, groups, arms=arms,
                                                n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED),
        "verbalized_pooled": cl.association(susp, outcome, groups,
                                            n_boot=N_BOOT, alpha=ALPHA, seed=BOOT_SEED),
    }

    all_pass = all(g["pass"] for g in gates.values())
    if not all_pass:
        failed = [k for k, g in gates.items() if not g["pass"]]
        if "on_policy_transfer" in failed:
            verdict = "UNTESTABLE"
            reading = ("the realness direction does not separate real from fabricated episodes "
                       "on-policy, so no association computed from its scores is interpretable. "
                       "This is a statement about the probe, not a null on P2, and the "
                       "credence-versus-conduct numbers are withheld accordingly.")
        else:
            verdict, reading = "INVALID", f"validity gate(s) failed: {failed}"
    else:
        verdict, reading = cl.decide(results["credence_within_arm"],
                                     results["verbalized_within_arm"])

    return {
        "experiment": "exp2_credence_predicts_behaviour",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "preregistration": {"alpha": ALPHA, "n_boot": N_BOOT, "seed": BOOT_SEED,
                            "direction": DIRECTION, "pooling": POOLING,
                            "credence_pooling": pooling_key,
                            "primary": "credence_within_arm",
                            "unit_of_independence": "task cell"},
        "gates": gates, "gates_all_pass": all_pass,
        "n_episodes": len(rows), "n_cells": len(cells),
        "results": None if verdict == "UNTESTABLE" else results,
        "arm_credence": {a: float(np.mean([r["credence"] for r in rows if r["arm"] == a]))
                         for a in behaviour.ARMS},
        "verdict": verdict, "reading": reading,
        "per_episode": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--exp0-artifact", default=str(EXP0 / "exp0-qwen3-32b.json"))
    ap.add_argument("--actor", default="openrouter:qwen/qwen3-32b")
    ap.add_argument("--device-map", default=None,
                    help='"auto" shards the weights across every visible GPU. Needed for 32B, '
                         "which is ~64 GB in bf16 and does not fit one card available here.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-prefixes", type=int, default=4)
    ap.add_argument("--stage1-only", action="store_true",
                    help="rebuild the direction and check the reproduction gate, then stop. "
                         "Needs no exp1 data, so the GPU path can be validated while the exp1 "
                         "collection is still running. Writes exp2-stage1.json and issues no "
                         "verdict -- a P2 claim without trajectories would be meaningless.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / ("exp2-stage1.json" if args.stage1_only else "exp2.json")
    out_path.unlink(missing_ok=True)   # never let a stale verdict survive a crashed run

    cells: list = []
    pairing: dict = {}
    config: dict = {"model_alias": args.actor}
    if not args.stage1_only:
        # Every shard, not just the pre-sharding log. exp1 collects one append-only log per arm so
        # the arms can run concurrently while each keeps a single writer; reading only
        # `episodes.jsonl` here silently analysed the handful of episodes collected before that
        # split and reported one complete cell out of twenty-four.
        rows: list[dict] = []
        legacy = EXP1 / "episodes.jsonl"
        if legacy.exists():
            rows.extend(load_episodes(legacy))
        for arm in behaviour.ARMS:
            shard = EXP1 / f"episodes-{arm}.jsonl"
            if shard.exists():
                rows.extend(load_episodes(shard))
        rows = dedupe_episodes([e for e in rows if e.get("model_alias") == args.actor])
        by_arm = {a: [e for e in rows if e.get("arm") == a] for a in behaviour.ARMS}
        cells, pairing = behaviour.build_cells(by_arm)
        config = behaviour.assert_configuration_identical(cells)
        log(f"{len(cells)} complete cells from {len(rows)} exp1 episodes")

    layer, exp0_auroc, artifact = inherited_layer(Path(args.exp0_artifact))
    if artifact["config"]["model"] != args.model:
        raise SystemExit(f"exp0 artifact is for {artifact['config']['model']}, "
                         f"not {args.model}; the layer would not be the inherited one")
    log(f"inherited layer {layer} for {DIRECTION} (exp0 held-out AUROC {exp0_auroc:.3f})")

    ex = HiddenStateExtractor(model=args.model, dtype=args.dtype, device_map=args.device_map)
    log(f"loaded {args.model} on {ex.hardware.device}, {ex.n_layers} layers")
    if ex.hardware.quantised:
        raise SystemExit("model is quantised; low-bit weights perturb the residual stream")

    try:
        direction, held_out, dinfo = rebuild_direction(ex, layer, args.batch_size, artifact)
        log(f"stage 1: re-derived {DIRECTION}, held-out AUROC {held_out:.3f}")
        gates = {
            "probe_reproduction": cl.reproduction_gate(held_out, exp0_auroc),
            "layer_inherited": {"layer": layer, "source": Path(args.exp0_artifact).name,
                                "pass": True},
            "actor_is_probed_model": actor_matches_probe(config["model_alias"], args.model),
            "not_quantised": {"quantised": ex.hardware.quantised,
                              "pass": not ex.hardware.quantised},
        }
        gates["probe_reproduction"].update(dinfo)

        if args.stage1_only:
            payload = {
                "experiment": "exp2_stage1_direction_only",
                "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "note": "GPU-path validation. No trajectories were scored and no P2 claim is "
                        "made; a verdict without exp1 trajectories would be meaningless.",
                "direction": DIRECTION, "pooling": POOLING, "layer": layer,
                "exp0_auroc": exp0_auroc, "rederived_auroc": held_out,
                "gates": {"probe_reproduction": gates["probe_reproduction"],
                          "layer_inherited": gates["layer_inherited"],
                          "not_quantised": gates["not_quantised"]},
                "hardware": ex.hardware.to_dict(),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            ok = payload["gates"]["probe_reproduction"]["pass"]
            log(f"stage 1 only: reproduction gate {'PASS' if ok else 'FAIL'} "
                f"({held_out:.3f} vs exp0 {exp0_auroc:.3f}); wrote "
                f"{out_path.relative_to(ROOT)}")
            return 0 if ok else 1

        log("stage 2: scoring exp1 trajectories")
        scores = score_episodes(ex, cells, direction, layer,
                                batch_size=args.batch_size, max_prefixes=args.max_prefixes)
    finally:
        ex.close()

    verdict = analyse(cells, scores, gates)
    verdict["pairing"] = pairing
    verdict["hardware"] = ex.hardware.to_dict()
    out_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print(f"  exp2 -- P2      {verdict['n_cells']} cells, {verdict['n_episodes']} episodes")
    print("=" * 78)
    for name, g in verdict["gates"].items():
        print(f"  gate {name:<28} {'PASS' if g['pass'] else 'FAIL'}")
    if verdict["results"]:
        print("-" * 78)
        for name, r in verdict["results"].items():
            star = " *" if r["excludes_zero"] else ""
            print(f"  {name:<26}{r['slope']:>+10.4f}  "
                  f"[{r['lo']:+.4f}, {r['hi']:+.4f}]{star}")
    print("=" * 78)
    print(f"  VERDICT: {verdict['verdict']}")
    print(f"  {verdict['reading']}")
    print("=" * 78)
    log(f"wrote {out_path.relative_to(ROOT)}")
    return 0 if verdict["verdict"] not in ("INVALID",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
