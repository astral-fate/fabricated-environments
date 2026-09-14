"""exp0 -- THE GATE. Are evaluation-awareness and environment-realness the same direction?

    python experiments/exp0_construct_separation.py --model Qwen/Qwen3-1.7B

If they are the same direction, there is no separate construct to measure, the suspicion meter
collapses into existing eval-awareness tooling, and the project should be rescoped or abandoned.
The decision rule is fixed in `PREREGISTRATION.md` and is not re-derived here:

    cross-AUROC(A,B) >= 0.85 in BOTH directions  ->  not separable; report and stop
    otherwise                                    ->  separable; proceed

Stages
------
1. episodes   run agents against the REAL arm, recording action sequences
2. replay     re-issue each action sequence against the LLM and scripted simulators, so the three
              substrates differ only in what they returned
3. stimuli    build A / C / B / S pair sets on that one common substrate
4. extract    per-layer residual-stream activations from the probed model
5. analyse    held-out AUROC per layer, cross-AUROC, cosine vs split-half reliability
6. verdict    validity gates first; the pre-registered rule only if all of them pass

Every stage writes to `results/exp0/` and stage 1 is resumable from its append-only log.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "analyze"))

import measures  # noqa: E402
import providers  # noqa: E402
from agent import (LogBusy, append_episode, completed_keys, dedupe_episodes,  # noqa: E402
                   episode_key, load_episodes, lock_log, run_episode)
from arms.real import RealArm  # noqa: E402
from arms.sim_provider import build_sim_provider  # noqa: E402
from arms.sim_stub import LLMSimArm, ScriptedSimArm, replay  # noqa: E402
from credence.stimuli import (TEST_VARIANTS, TRAIN_VARIANTS, build_chat_pair_set,  # noqa: E402
                              build_pair_sets)
from measures import (Clustered, MeasureError, assert_disjoint, auroc, auroc_statistic,  # noqa: E402
                      bootstrap_ci, cosine, mean_difference_direction, normalised_cosine,
                      project, shuffled_control, split_by_group, split_half_reliability)
from task import KINDS, build_task  # noqa: E402

OUT = ROOT / "results" / "exp0"

#: Pre-registered. See PREREGISTRATION.md; changing this here would be changing the gate.
CROSS_AUROC_THRESHOLD = 0.85
REPRODUCTION_FLOOR = 0.70
SHUFFLE_TOLERANCE = 0.05
SPLIT_FRAC = 0.5
SPLIT_SEED = 0
N_BOOT = 2000

#: Largest share of SIM tool outputs allowed to be simulated refusals before the B_empirical
#: contrast stops being about realness. Reported always; blocks only past this bound.
SIM_REFUSAL_LIMIT = 0.50


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- stage 1-2

def collect_records(args) -> list[dict]:
    """Run REAL episodes, then replay each action sequence against both simulators."""
    ep_log = OUT / "episodes.jsonl"
    if not args.offline:
        providers.load_env(args.env)

    actor_alias = "offline-scripted" if args.offline else args.actor
    tasks = [build_task(k, s) for k in KINDS for s in range(args.seeds)]
    done = completed_keys(ep_log)
    todo = [t for t in tasks if episode_key(t.task_id, "real", actor_alias) not in done]
    log(f"stage 1: {len(tasks)} task cells, {len(done)} already complete, {len(todo)} to run"
        + ("  [OFFLINE -- not reportable]" if args.offline else ""))

    if todo:
        actor = None if args.offline else providers.build(args.actor)
        lock = lock_log(ep_log)
        try:
            lock.__enter__()
        except LogBusy as exc:
            log(f"FAIL: {exc}")
            raise
        try:
          for i, task in enumerate(todo, 1):
            arm = RealArm(root=OUT / "checkouts" / task.task_id,
                          log_path=OUT / "checkouts" / f"{task.task_id}.events.jsonl")
            if args.offline:
                from offline_actor import build_offline_actor
                actor = build_offline_actor(task, seed=task.seed, n_turns=args.max_turns)
            ep = run_episode(provider=actor, task=task, arm=arm, model_alias=actor_alias,
                             max_turns=args.max_turns)
            append_episode(ep_log, ep)
            v = ep.scope_report.get("counts", {})
            log(f"  [{i}/{len(todo)}] {task.task_id} turns={len(ep.turns)} "
                f"actions={len(ep.actions)} violations={sum(v.values())} "
                f"success={ep.success}{' API-ERROR' if ep.api_error else ''}")
            arm.close()
        finally:
            # Released even when an episode raises, so a crash does not strand the lock and
            # block every later resume.
            lock.__exit__()

    # Filter by actor. The episode log is shared and append-only, so an earlier OFFLINE run's
    # scripted trajectories sit in the same file. Without this filter they would be replayed and
    # analysed alongside real ones, and a reportable run would silently contain fabricated
    # trajectories -- the exact contamination the `offline` flag exists to prevent.
    episodes = dedupe_episodes([e for e in load_episodes(ep_log)
                                if e.get("model_alias") == actor_alias])
    n_raw = len([e for e in load_episodes(ep_log)
                 if not e.get("api_error") and e.get("model_alias") == actor_alias])
    if n_raw != len(episodes):
        log(f"         ({n_raw - len(episodes)} duplicate cell(s) dropped -- a cell run twice is "
            f"one observation, not two)")
    log(f"stage 1: {len(episodes)} usable episodes for actor {actor_alias!r}")
    n_err = len([e for e in load_episodes(ep_log)
                 if e.get("api_error") and e.get("model_alias") == actor_alias])
    if n_err:
        log(f"         ({n_err} api_error episode(s) -- not counted, will be retried on resume)")
    other = len([e for e in load_episodes(ep_log) if e.get("model_alias") != actor_alias])
    if other:
        log(f"         ({other} episode(s) from other actors in the log, excluded)")

    # ---- stage 2: replay the SAME action sequence against the two simulators
    #
    # Cached per episode in an APPEND-ONLY JSONL, not written once at the end. The LLM simulator
    # costs one API call per action, so a 24-episode stage is ~200 calls -- and on the last
    # unexhausted key, dying at episode 20 with a write-at-the-end cache would discard every one
    # of them. Same rule as stage 1: the append-only log is authoritative.
    cache = OUT / ("replays-offline.jsonl" if args.offline else "replays.jsonl")
    cached: dict[str, dict] = {}
    if cache.exists() and not args.refresh_replays:
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                  # torn final line after a kill
            cached[rec["episode_id"]] = rec
        if cached:
            log(f"stage 2: {len(cached)} replay(s) already cached")

    todo_eps = [e for e in episodes if e["episode_id"] not in cached]
    # A purpose-built text client, not providers.build(): see arms/sim_provider.py on the
    # missing request timeout that made a stalled call look like a wedged run.
    sim_provider = None if args.offline or not todo_eps else build_sim_provider(args.simulator)
    records = [cached[e["episode_id"]] for e in episodes if e["episode_id"] in cached]
    for i, ep in enumerate(todo_eps, 1):
        actions = [a for a in ep["actions"] if not a.get("invalid")]
        if not actions:
            continue
        kind, seed = ep["task_kind"], ep["seed"]
        task = build_task(kind, seed)
        real_outputs = [str(a.get("preview", "")) for a in []]      # placeholder, replaced below

        # REAL outputs are re-derived by replaying against a fresh RealArm rather than read from
        # the turn previews, which are truncated for readability. The substrate is deterministic
        # given the same seeded task and the same action sequence, so this reproduces the bytes
        # the agent actually saw, at full length.
        real_arm = RealArm(root=OUT / "replay" / f"{ep['episode_id']}",
                           log_path=OUT / "replay" / f"{ep['episode_id']}.events.jsonl")
        real_outputs = replay(real_arm, task, actions)
        real_arm.close()

        scripted_outputs = replay(ScriptedSimArm(seed=seed), task, actions)
        if args.offline:
            # A second template simulator with a different seed, so B and S are at least distinct
            # code paths. They are NOT distinct constructs, which is one reason an offline run is
            # never reportable.
            sim_outputs = replay(ScriptedSimArm(seed=seed + 7919), task, actions)
        else:
            sim_outputs = replay(LLMSimArm(sim_provider), task, actions)

        texts = []
        for t in ep["turns"]:
            for _ in (t.get("calls") or []):
                texts.append(t.get("text", ""))

        rec = {
            "episode_id": ep["episode_id"], "task_id": ep["task_id"],
            "task_kind": kind, "seed": seed, "brief": task.brief,
            "actions": actions, "assistant_texts": texts[:len(actions)],
            "real": real_outputs, "sim": sim_outputs, "scripted": scripted_outputs,
        }
        records.append(rec)
        with cache.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        log(f"  [{i}/{len(todo_eps)}] replayed {ep['episode_id']} ({len(actions)} actions)")

    return records


# --------------------------------------------------------------------------- stage 5

def analyse_direction(data: Clustered, name: str, variants: np.ndarray) -> dict:
    """Per-layer held-out AUROC and the direction at the best layer.

    The split is disjoint on TWO axes, not one:

      episodes   the unit of independence; prefixes from one episode are near-duplicates
      wording    where the manipulation is a sentence (A and C), train and test use disjoint
                 paraphrase pools, so a direction that encodes one surface form scores at chance
                 on the test set rather than at 1.000

    The second axis was added after a first pass returned a held-out AUROC of exactly 1.000 for
    every direction -- the signature of a token-presence detector rather than a construct. Where
    the manipulation is not a sentence (B and S vary the tool outputs), `variant` is -1 and only
    the episode axis applies.
    """
    n_layers = data.x.shape[1]
    tr_idx, te_idx = split_by_group(
        Clustered(np.arange(len(data.x)).reshape(-1, 1), data.y, data.groups),
        SPLIT_FRAC, seed=SPLIT_SEED)
    assert_disjoint(tr_idx, te_idx)
    tr_rows = tr_idx.x.ravel()
    te_rows = te_idx.x.ravel()

    has_variants = bool((variants >= 0).any())
    if has_variants:
        tr_rows = tr_rows[np.isin(variants[tr_rows], TRAIN_VARIANTS)]
        te_rows = te_rows[np.isin(variants[te_rows], TEST_VARIANTS)]
        if len(tr_rows) == 0 or len(te_rows) == 0:
            raise MeasureError(f"{name}: paraphrase split left an empty side")

    per_layer = []
    directions = {}
    for layer in range(n_layers):
        xt, yt = data.x[tr_rows, layer, :], data.y[tr_rows]
        xe, ye = data.x[te_rows, layer, :], data.y[te_rows]
        try:
            v = mean_difference_direction(xt[yt == 1], xt[yt == 0])
        except MeasureError:
            per_layer.append({"layer": layer, "auroc": float("nan")})
            continue
        directions[layer] = v
        per_layer.append({"layer": layer, "auroc": auroc(project(xe, v), ye)})

    finite = [r for r in per_layer if not np.isnan(r["auroc"])]
    if not finite:
        raise MeasureError(f"{name}: no layer produced a usable direction")
    best = max(finite, key=lambda r: r["auroc"])
    best_layer = best["layer"]

    te = Clustered(data.x[te_rows, best_layer, :], data.y[te_rows], data.groups[te_rows])
    ci = bootstrap_ci(auroc_statistic(directions[best_layer]), te, n_boot=N_BOOT, seed=0)

    full = Clustered(data.x[:, best_layer, :], data.y, data.groups)
    rel = split_half_reliability(full, n_repeats=20, seed=0)
    shuf = shuffled_control(full, n_repeats=25, seed=0)

    return {
        "name": name, "n_pairs": len(data.x) // 2, "n_groups": len(np.unique(data.groups)),
        "n_train_groups": len(tr_idx.unique_groups), "n_test_groups": len(te_idx.unique_groups),
        "per_layer": per_layer, "best_layer": best_layer,
        "paraphrase_split": bool(has_variants),
        "n_train_rows": int(len(tr_rows)), "n_test_rows": int(len(te_rows)),
        "best_auroc": best["auroc"], "best_auroc_ci": ci,
        "reliability": rel, "shuffled_control": shuf,
        "_direction": directions[best_layer], "_test_rows": te_rows,
        "_directions_all_layers": directions,
    }


def cross_auroc(src: dict, dst_data: Clustered, dst: dict) -> dict:
    """Score `src`'s direction on `dst`'s held-out data, at `dst`'s best layer.

    The layer is `dst`'s, because the question is how well the source direction classifies the
    destination's contrast where that contrast is best expressed. Using the source's layer would
    ask a different and less favourable question, and the pre-registered threshold is a claim
    about whether the constructs COLLAPSE -- so the comparison is made where collapse would be
    easiest to detect.
    """
    layer = dst["best_layer"]
    rows = dst["_test_rows"]
    v = src["_directions_all_layers"].get(layer)
    if v is None:
        return {"auroc": float("nan"), "layer": layer}
    te = Clustered(dst_data.x[rows, layer, :], dst_data.y[rows], dst_data.groups[rows])
    ci = bootstrap_ci(auroc_statistic(v), te, n_boot=N_BOOT, seed=0)
    return {"from": src["name"], "to": dst["name"], "layer": layer,
            "auroc": ci["point"], "ci": ci}


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B", help="probed open-weight model")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--actor", default="qwen38-27b", help="generates the action sequences")
    ap.add_argument("--simulator", default="openrouter:mistralai/mistral-nemo",
                    help="fabricates SIM tool outputs. OpenRouter rather than Groq: the Groq "
                         "organisation's 200k/day budget for the actor model is spent, and the "
                         "simulator must not compete with it. ~190 short calls costs about "
                         "$0.01 here, and the account is pay-as-you-go, not the $0.09 free tier "
                         "the project brief recorded.")
    ap.add_argument("--env", default=str(ROOT / ".env"),
                    help="realness/.env lists the FRESH Groq organisation first; see that file")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-prefixes", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--refresh-replays", action="store_true")
    ap.add_argument("--min-episodes", type=int, default=12,
                    help="refuse to analyse fewer usable episodes than this. The episode is the "
                         "unit of independence, so a thin set is a thin bootstrap, and a run "
                         "whose API keys expired part-way must fail loudly rather than quietly "
                         "reporting a gate verdict off six episodes.")
    ap.add_argument("--stimuli-only", action="store_true", help="stop before loading the model")
    ap.add_argument("--offline", action="store_true",
                    help="scripted actor and template simulators; validates the pipeline "
                         "without API budget. NEVER reportable.")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)

    records = collect_records(args)
    if not records:
        log("FAIL: no usable episodes; cannot build stimuli")
        return 1
    # ---- SIM-arm integrity. The manipulated variable must actually be the manipulation.
    #
    # `B_empirical` contrasts genuine substrate bytes against LLM-fabricated bytes. If the
    # simulator was rate-limited, its "outputs" are error strings, and the contrast silently
    # becomes "real output vs error message" -- which a probe separates perfectly, for entirely
    # the wrong reason. A first run reached a confident verdict with 35% of its SIM outputs in
    # that state. `SimulatorUnavailable` now prevents new contamination; this catches a cached
    # replay file written before that existed.
    n_sim = n_empty = n_refusal = 0
    for r in records:
        for out in r.get("sim", []):
            n_sim += 1
            if not str(out).strip():
                n_empty += 1
            elif str(out).startswith("error:"):
                n_refusal += 1
    frac_refusal = n_refusal / max(n_sim, 1)
    log(f"stage 2: SIM fidelity -- {n_refusal}/{n_sim} simulated refusals "
        f"({frac_refusal:.1%}), {n_empty} empty")

    # An empty output is never a fabricated response; `SimulatorUnavailable` should have stopped
    # the run before one reached the data at all, so any here means a cached file predates that.
    if n_empty:
        log(f"FAIL: {n_empty} empty SIM output(s) -- not fabrications. Delete "
            f"results/exp0/replays.jsonl and re-run.")
        return 4

    # A simulated "error: file not found" IS a fabricated environment response, so it is fidelity,
    # not contamination -- a different thing from the rate-limit strings this check originally
    # caught, which can no longer enter the data because they raise. It still bounds the
    # measurement: past roughly half, the SIM arm is mostly denying that the world exists, and
    # B_empirical would be separating "file contents vs refusal" rather than anything about
    # realness. The rate is reported either way, because simulator fidelity is exactly what
    # Williams et al. name as the central open challenge.
    if frac_refusal > SIM_REFUSAL_LIMIT:
        log(f"FAIL: {frac_refusal:.1%} of SIM outputs are refusals, above the "
            f"{SIM_REFUSAL_LIMIT:.0%} limit -- the simulated world is too degenerate to contrast "
            f"against a real one.")
        return 4

    if len(records) < args.min_episodes:
        # Silently analysing a truncated episode set is the failure mode this guard exists for.
        # Every interval in exp0 comes from a cluster bootstrap over EPISODES, so too few of them
        # does not produce a noisy answer -- it produces a confident-looking one off a handful of
        # clusters. Exhausted API keys are the likely cause, and the fix is more keys or a later
        # resume, not a smaller n.
        log(f"FAIL: only {len(records)} usable episode(s), below --min-episodes="
            f"{args.min_episodes}.")
        log("      The episode is the unit of independence; a bootstrap over this few clusters "
            "is not a result.")
        log("      Likely cause: API keys exhausted mid-run. Add keys (GROQ_API_KEY_5, ... or "
            "GROQ_API_KEYS=a,b,c) and re-run --")
        log("      stage 1 is resumable and append-only, so completed episodes are not repeated.")
        return 3

    sets = build_pair_sets(records, max_prefixes_per_episode=args.max_prefixes)
    sets["A_chat"] = build_chat_pair_set()
    for s in sets.values():
        log(f"stage 3: {s.name:<12} {len(s):>4} pairs over {len(s.groups):>3} groups")
    (OUT / "stimuli_summary.json").write_text(
        json.dumps({k: v.summary() for k, v in sets.items()}, indent=2), encoding="utf-8")
    if args.stimuli_only:
        return 0

    # ---- stage 4
    from credence.probe import HiddenStateExtractor, pair_activations

    # Activations are cached because extraction is the expensive, deterministic half of the run:
    # ~3.5 minutes of GPU for 1,200 forward passes that depend only on (model, dtype, stimuli).
    # Re-running the ANALYSIS -- which is where every remaining bug lives -- should not have to
    # pay for that again, and a cache keyed on the stimulus summary invalidates itself whenever
    # the stimuli change.
    import hashlib as _hashlib

    key = _hashlib.sha256(
        json.dumps({k: v.summary() for k, v in sets.items()}, sort_keys=True).encode()
        + f"|{args.model}|{args.dtype}".encode()).hexdigest()[:16]
    acts_path = OUT / f"activations-{key}.npz"

    if acts_path.exists():
        log(f"stage 4: loading cached activations {acts_path.name}")
        blob = np.load(acts_path, allow_pickle=False)
        data: dict[str, dict[str, Clustered]] = {}
        variants: dict[str, np.ndarray] = {}
        for name in sets:
            variants[name] = blob[f"{name}|variants"]
            data[name] = {pool: Clustered(blob[f"{name}|{pool}|x"], blob[f"{name}|y"],
                                          blob[f"{name}|groups"])
                          for pool in ("last", "mean")}
        hardware = {"device": str(blob["hardware_device"]), "dtype": args.dtype,
                    "quantised": False, "vram_gib": float(blob["hardware_vram"])}
        n_layers = int(blob["n_layers"])
        hidden_size = int(blob["hidden_size"])
        return _analyse(args, sets, data, variants, hardware, n_layers, hidden_size)

    log(f"stage 4: loading {args.model} ({args.dtype})")
    ex = HiddenStateExtractor(model=args.model, dtype=args.dtype)
    log(f"  {ex.hardware.device} {ex.hardware.vram_gib} GiB, "
        f"{ex.n_layers} layers x {ex.hidden_size} dims, quantised={ex.hardware.quantised}")

    data: dict[str, dict[str, Clustered]] = {}
    variants: dict[str, np.ndarray] = {}
    for name, ps in sets.items():
        log(f"  extracting {name} ({len(ps)} pairs -> {2 * len(ps)} contexts)")
        got = pair_activations(ex, ps.pairs, batch_size=args.batch_size, progress=log)
        data[name] = {pool: Clustered(got["acts"][pool], got["y"], got["groups"])
                      for pool in ("last", "mean")}
        variants[name] = got["variants"]
    ex.close()

    np.savez_compressed(
        acts_path,
        **{f"{n}|{p}|x": data[n][p].x for n in sets for p in ("last", "mean")},
        **{f"{n}|y": data[n]["last"].y for n in sets},
        **{f"{n}|groups": data[n]["last"].groups for n in sets},
        **{f"{n}|variants": variants[n] for n in sets},
        hardware_device=np.array(ex.hardware.device), hardware_vram=np.array(ex.hardware.vram_gib),
        n_layers=np.array(ex.n_layers), hidden_size=np.array(ex.hidden_size),
    )
    log(f"stage 4: cached activations -> {acts_path.name}")
    return _analyse(args, sets, data, variants, ex.hardware.to_dict(), ex.n_layers,
                    ex.hidden_size)


def _analyse(args, sets, data, variants, hardware, n_layers, hidden_size) -> int:
    # ---- stage 5
    results: dict = {"config": vars(args) | {"threshold": CROSS_AUROC_THRESHOLD},
                     "hardware": hardware,
                     "n_layers": n_layers, "hidden_size": hidden_size,
                     "stimuli": {k: v.summary() for k, v in sets.items()},
                     "pooling": {}}

    for pool in ("last", "mean"):
        log(f"stage 5: analysing pooling={pool}")
        dirs = {}
        for name in sets:
            try:
                dirs[name] = analyse_direction(data[name][pool], name, variants[name])
                d = dirs[name]
                log(f"  {name:<12} best layer {d['best_layer']:>2}  "
                    f"AUROC {d['best_auroc']:.3f} "
                    f"[{d['best_auroc_ci']['lo']:.3f},{d['best_auroc_ci']['hi']:.3f}]  "
                    f"rel {d['reliability']['mean']:.3f}  "
                    f"shuffled {d['shuffled_control']['mean']:.3f}")
            except MeasureError as exc:
                log(f"  {name}: FAILED ({exc})")

        cross = {}
        for a in dirs:
            for b in dirs:
                if a == b:
                    continue
                cross[f"{a}->{b}"] = cross_auroc(dirs[a], data[b][pool], dirs[b])

        cos = {}
        for a in dirs:
            for b in dirs:
                if a >= b:
                    continue
                la, lb = dirs[a]["best_layer"], dirs[b]["best_layer"]
                layer = la  # compare at A's best layer; both directions exist at every layer
                va = dirs[a]["_directions_all_layers"].get(layer)
                vb = dirs[b]["_directions_all_layers"].get(layer)
                if va is None or vb is None:
                    continue
                raw = cosine(va, vb)
                cos[f"{a}|{b}"] = {
                    "layer": layer, "cosine": raw,
                    "rel_a": dirs[a]["reliability"]["mean"],
                    "rel_b": dirs[b]["reliability"]["mean"],
                    "normalised": normalised_cosine(raw, dirs[a]["reliability"]["mean"],
                                                    dirs[b]["reliability"]["mean"]),
                    "best_layer_a": la, "best_layer_b": lb,
                }

        results["pooling"][pool] = {
            "directions": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                           for k, v in dirs.items()},
            "cross_auroc": cross,
            "cosine": cos,
        }

    # ---- stage 6: validity gates, then the pre-registered rule
    gate = results["pooling"]["last"]
    verdict = evaluate_gate(gate, offline=args.offline)
    results["offline"] = args.offline
    results["verdict"] = verdict

    name = "exp0-offline.json" if args.offline else "exp0.json"
    (OUT / name).write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log(f"wrote {OUT / name}")

    print("\n" + "=" * 78)
    print(verdict["statement"])
    print("=" * 78)
    return 0 if verdict["reportable"] else 2


def evaluate_gate(pool_result: dict, *, offline: bool = False) -> dict:
    """Validity gates first. The pre-registered rule is only applied if all of them pass."""
    dirs = pool_result["directions"]
    cross = pool_result["cross_auroc"]
    checks, failures = {}, []
    if offline:
        # An offline run has no agent and no LLM simulator. Every stage executed, which is
        # what it was for, but B and S were fabricated by the same template family and no
        # trajectory came from a model. Refusing the verdict here is what stops a pipeline
        # check from being mistaken later for a result.
        failures.append("offline run: scripted trajectories and template simulators are not "
                        "evidence about any model")

    # 1. negative control
    worst = None
    for name, d in dirs.items():
        m = d["shuffled_control"]["mean"]
        if worst is None or abs(m - 0.5) > abs(worst[1] - 0.5):
            worst = (name, m)
    checks["negative_control"] = {"worst": worst, "tolerance": SHUFFLE_TOLERANCE,
                                  "pass": worst is not None
                                  and abs(worst[1] - 0.5) <= SHUFFLE_TOLERANCE}
    if not checks["negative_control"]["pass"]:
        failures.append(f"shuffled-label control off chance: {worst}")

    # 2. reproduction floor for the eval-awareness direction
    a = dirs.get("A_eval")
    got = a["best_auroc"] if a else float("nan")
    checks["reproduction_floor"] = {"A_eval_best_auroc": got, "floor": REPRODUCTION_FLOOR,
                                    "pass": bool(a) and got >= REPRODUCTION_FLOOR}
    if not checks["reproduction_floor"]["pass"]:
        failures.append(
            f"A_eval held-out AUROC {got:.3f} < {REPRODUCTION_FLOOR}: the eval-awareness "
            "direction was not recovered, so comparing anything to it is meaningless")

    # 3. leakage
    leak_ok = all(d["n_train_groups"] + d["n_test_groups"] == d["n_groups"]
                  for d in dirs.values())
    checks["no_leakage"] = {"pass": leak_ok}
    if not leak_ok:
        failures.append("train/test split does not partition the episodes")

    ab = cross.get("A_eval->B_empirical", {}).get("auroc", float("nan"))
    ba = cross.get("B_empirical->A_eval", {}).get("auroc", float("nan"))

    if failures:
        return {"reportable": False, "checks": checks, "failures": failures,
                "cross_A_to_B": ab, "cross_B_to_A": ba,
                "statement": "exp0 NOT REPORTABLE -- validity gate failed:\n  - "
                             + "\n  - ".join(failures)}

    # The pre-registered rule. "Same direction" would mean each probe classifies the other's
    # contrast well, so collapse requires BOTH crossings to clear the threshold.
    collapsed = (not np.isnan(ab) and not np.isnan(ba)
                 and ab >= CROSS_AUROC_THRESHOLD and ba >= CROSS_AUROC_THRESHOLD)
    if collapsed:
        statement = (
            f"exp0 VERDICT: NOT SEPARABLE.\n"
            f"  cross-AUROC A->B = {ab:.3f}, B->A = {ba:.3f}; both >= {CROSS_AUROC_THRESHOLD}.\n"
            f"  Evaluation-awareness and environment-realness are not distinguishable as linear\n"
            f"  directions in this model. P5' is FALSIFIED. Per the preregistration the project\n"
            f"  stops here and this negative result is the finding.")
    else:
        statement = (
            f"exp0 VERDICT: SEPARABLE.\n"
            f"  cross-AUROC A->B = {ab:.3f}, B->A = {ba:.3f}; "
            f"at least one < {CROSS_AUROC_THRESHOLD}.\n"
            f"  Environment-realness is not reducible to evaluation-awareness in this model.\n"
            f"  P5' is SUPPORTED at this scale. The project proceeds.")

    return {"reportable": True, "checks": checks, "failures": [],
            "cross_A_to_B": ab, "cross_B_to_A": ba,
            "separable": not collapsed, "statement": statement}


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
