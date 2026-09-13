"""Where the probes fail, and whether the failures are structured.

    python analyze/error_analysis.py

A held-out AUROC says how often the direction ranks a positive above a negative. It says nothing
about *which* items it gets wrong, and that distinction carries the interesting content. Two
pipelines with the same AUROC are different objects if one errs uniformly at random and the other
errs on every prefix from three particular episodes: the first is noise, the second is a subgroup
the construct does not cover.

This module answers four questions that the summary numbers cannot:

  concentration   are errors spread across episodes, or concentrated in a few? Reported as the
                  share of all errors contributed by the worst-performing quartile of episodes,
                  against the 25% a uniform error process would give.
  task class      does accuracy depend on whether the task was solvable, blocked or tempting? The
                  three classes differ in agent behaviour, so a direction that only works on one
                  of them would be a much weaker result than its pooled AUROC suggests.
  prefix depth    does accuracy depend on how much transcript the model has seen? A realness
                  direction that only separates once many tool outputs have accumulated behaves
                  differently from one that fires immediately, and the distinction matters for any
                  instrument meant to run early in a trajectory.
  paraphrase      for the sentence-swapped directions, does accuracy depend on which held-out
                  wording was used? Large variation would mean the direction is still partly
                  lexical despite the disjoint-wording split.

Scope: per-item scores are recomputed from the locally cached activations, which exist only for
the development-scale run. The larger scales report their layer profile (available in every result
file) but not their item-level errors, and the paper says so rather than implying otherwise.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exp0"
sys.path.insert(0, str(ROOT / "analyze"))
sys.path.insert(0, str(ROOT / "src"))

from measures import (Clustered, MeasureError, auroc, mean_difference_direction,  # noqa: E402
                      project, split_by_group)

SPLIT_FRAC, SPLIT_SEED = 0.5, 0


def _cache_path() -> Path | None:
    hits = sorted(RESULTS.glob("activations-*.npz"))
    return hits[0] if hits else None


def _episode_kinds() -> dict[str, str]:
    """episode_id -> task kind, from the append-only episode log."""
    out: dict[str, str] = {}
    path = RESULTS / "episodes.jsonl"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not ep.get("api_error"):
            out[ep["episode_id"]] = ep["task_kind"]
    return out


def _rebuild_items() -> dict[str, list[Any]]:
    """Rebuild the stimulus sets so row order can be mapped back to item identity.

    `build_pair_sets` is deterministic given the replay cache, and `pair_activations` emits the
    pos/neg rows of each pair in order, so the i-th pair corresponds to rows 2i and 2i+1. That is
    the only link back from an activation matrix to `episode@boundary` identity, because the
    extracted arrays carry groups and variants but not item ids.
    """
    from credence.stimuli import build_chat_pair_set, build_pair_sets

    cache = RESULTS / "replays.jsonl"
    records = []
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    sets = build_pair_sets(records, max_prefixes_per_episode=4)
    sets["A_chat"] = build_chat_pair_set()
    return {name: ps.pairs for name, ps in sets.items()}


def _boundary(item: str) -> int | None:
    """Parse the prefix depth out of an item id shaped `<episode>@<k>` or `<episode>@<k>v<n>`."""
    if "@" not in item:
        return None
    tail = item.split("@", 1)[1]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


def analyse(name: str, blob: Any, pairs: list[Any], kinds: dict[str, str]) -> dict[str, Any]:
    x = blob[f"{name}|last|x"]
    y = blob[f"{name}|y"]
    groups = blob[f"{name}|groups"]
    variants = blob[f"{name}|variants"]

    # Same split, same seed as the experiment, so the errors analysed are the errors reported.
    idx = Clustered(np.arange(len(x)).reshape(-1, 1), y, groups)
    tr_idx, te_idx = split_by_group(idx, SPLIT_FRAC, seed=SPLIT_SEED)
    tr, te = tr_idx.x.ravel(), te_idx.x.ravel()
    if (variants >= 0).any():
        tr = tr[np.isin(variants[tr], (0, 1, 2))]
        te = te[np.isin(variants[te], (3, 4, 5))]

    # Layer 0 is the embedding output, where the final prompt token is the SAME token in both
    # members of a pair -- the manipulation sits earlier in the context and has not yet been
    # mixed in by attention. The class means are therefore identical and the direction is
    # undefined, which is a fact about the architecture rather than a failure. `exp0` skips such
    # layers when building its curve; this does the same so the two agree.
    best, best_layer = -1.0, None
    for layer in range(x.shape[1]):
        try:
            v = mean_difference_direction(x[tr, layer, :][y[tr] == 1], x[tr, layer, :][y[tr] == 0])
        except MeasureError:
            continue
        a = auroc(project(x[te, layer, :], v), y[te])
        if not np.isnan(a) and a > best:
            best, best_layer = a, layer
    if best_layer is None:
        raise MeasureError(f"{name}: no layer produced a usable direction")

    v = mean_difference_direction(x[tr, best_layer, :][y[tr] == 1],
                                  x[tr, best_layer, :][y[tr] == 0])
    scores = project(x[te, best_layer, :], v)

    # Threshold at the midpoint of class means: a decision rule the direction itself implies,
    # rather than one tuned on the test set.
    thr = (scores[y[te] == 1].mean() + scores[y[te] == 0].mean()) / 2.0
    pred = (scores > thr).astype(int)
    correct = pred == y[te]

    # ---- concentration
    per_ep: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(te, correct):
        per_ep[str(groups[row])].append(bool(ok))
    ep_err = {e: 1.0 - (sum(v_) / len(v_)) for e, v_ in per_ep.items() if v_}
    n_err_total = sum((1 - sum(v_) / len(v_)) * len(v_) for v_ in per_ep.values())
    ranked = sorted(ep_err.items(), key=lambda kv: -kv[1])
    q = max(1, len(ranked) // 4)
    worst_err = sum((ep_err[e]) * len(per_ep[e]) for e, _ in ranked[:q])
    concentration = (worst_err / n_err_total) if n_err_total > 0 else float("nan")

    # ---- by task class
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for row, ok in zip(te, correct):
        by_kind[kinds.get(str(groups[row]), "unknown")].append(bool(ok))

    # ---- by prefix depth and paraphrase, via rebuilt item identity
    by_depth: dict[int, list[bool]] = defaultdict(list)
    by_variant: dict[int, list[bool]] = defaultdict(list)
    for row, ok in zip(te, correct):
        pair = pairs[row // 2] if row // 2 < len(pairs) else None
        if pair is None:
            continue
        d = _boundary(pair.item)
        if d is not None:
            by_depth[d].append(bool(ok))
        if pair.variant >= 0:
            by_variant[pair.variant].append(bool(ok))

    def acc(d: dict[Any, list[bool]]) -> dict[str, Any]:
        return {str(k): {"n": len(v_), "accuracy": round(sum(v_) / len(v_), 3)}
                for k, v_ in sorted(d.items())}

    return {
        "direction": name,
        "best_layer": best_layer,
        "held_out_auroc": round(best, 3),
        "n_test_items": int(len(te)),
        "n_errors": int((~correct).sum()),
        "overall_accuracy": round(float(correct.mean()), 3),
        "error_concentration_worst_quartile": (None if np.isnan(concentration)
                                               else round(float(concentration), 3)),
        "uniform_expectation": 0.25,
        "by_task_class": acc(by_kind),
        "by_prefix_depth": acc(by_depth),
        "by_paraphrase": acc(by_variant),
    }


def main() -> int:
    cache = _cache_path()
    if cache is None:
        print("no local activation cache; run exp0 first")
        return 1
    print(f"loading {cache.name}")
    blob = np.load(cache, allow_pickle=False)
    kinds = _episode_kinds()
    items = _rebuild_items()

    out: dict[str, Any] = {"source": cache.name, "directions": {}}
    for name in ("A_eval", "C_declared", "B_empirical", "S_scripted"):
        if f"{name}|last|x" not in blob:
            continue
        rep = analyse(name, blob, items.get(name, []), kinds)
        out["directions"][name] = rep
        print(f"\n{name}  layer {rep['best_layer']}  AUROC {rep['held_out_auroc']}  "
              f"acc {rep['overall_accuracy']}  errors {rep['n_errors']}/{rep['n_test_items']}")
        print(f"  error concentration (worst quartile of episodes): "
              f"{rep['error_concentration_worst_quartile']} vs {rep['uniform_expectation']} uniform")
        print(f"  by task class : {rep['by_task_class']}")
        print(f"  by depth      : {rep['by_prefix_depth']}")
        if rep["by_paraphrase"]:
            print(f"  by paraphrase : {rep['by_paraphrase']}")

    dest = RESULTS / "error_analysis.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
