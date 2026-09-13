"""Measures for exp0: AUROC, mean-difference directions, cosine, and cluster bootstrap CIs.

Two rules are enforced here mechanically rather than by discipline, because both were learned
expensively.

1. The unit of independence is the EPISODE, never the prefix
------------------------------------------------------------
One 20-turn episode yields ~20 transcript prefixes that are near-duplicates of each other. Pooling
them as independent observations multiplies apparent n by 20 and makes every interval far too
narrow. `project-v2/analyze/measures.py:473` records paper 1 doing exactly this once:

    "Generation is the declared unit of independence, and until now it was declared but not
    enforced ... Pooling them is textbook pseudo-replication, and it makes an exact test MORE
    anti-conservative than the normal approximation."

So every function that takes items also takes `groups`, and every resample here resamples GROUPS
with replacement and takes all of a chosen group's items. `bootstrap_ci` has no signature that
lets a caller forget.

2. A bare cosine is uninterpretable, so it is reported against its own reliability ceiling
------------------------------------------------------------------------------------------
In high dimensions two random unit vectors have cosine ~0, so *any* positive cosine looks
meaningful. The question "are these the same direction?" is only answerable relative to how well
each direction is measured at all. `split_half_reliability` trains a direction twice on disjoint
halves of the episodes and takes the cosine between the halves; `normalised_cosine` then divides
cos(A,B) by sqrt(rel(A) * rel(B)). A normalised value near 1 means "as aligned as this data can
show anything to be", which is the comparison the gate actually needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


class MeasureError(ValueError):
    """Raised when a measure is asked for something its assumptions do not support."""


# --------------------------------------------------------------------------- AUROC

def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties handled by average rank.

    `labels` are 0/1. Returns NaN when either class is absent, which happens in bootstrap
    resamples and must propagate rather than silently becoming 0.5.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.shape != y.shape:
        raise MeasureError(f"scores {s.shape} and labels {y.shape} differ in length")
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ranks within tied groups, or ties inflate or deflate the statistic
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------- directions

def mean_difference_direction(x_pos: np.ndarray, x_neg: np.ndarray) -> np.ndarray:
    """The mean-difference (difference-of-means) probe direction, L2-normalised.

    This is the estimator Heidari et al. and Zhuang & Aranguri both use on Qwen-family residual
    streams, chosen here for comparability rather than for accuracy: a logistic probe would
    usually separate better, but it would not be the same object whose cosine those papers report.
    """
    p = np.asarray(x_pos, dtype=np.float64)
    n = np.asarray(x_neg, dtype=np.float64)
    if p.ndim != 2 or n.ndim != 2 or p.shape[1] != n.shape[1]:
        raise MeasureError(f"activation shapes incompatible: {p.shape} vs {n.shape}")
    if len(p) == 0 or len(n) == 0:
        raise MeasureError("a direction needs at least one example of each class")
    d = p.mean(axis=0) - n.mean(axis=0)
    norm = float(np.linalg.norm(d))
    if norm == 0.0:
        raise MeasureError("degenerate direction: the two class means are identical")
    return d / norm


def project(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Score every row by its projection onto `direction`."""
    return np.asarray(x, dtype=np.float64) @ np.asarray(direction, dtype=np.float64)


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    a = np.asarray(u, dtype=np.float64)
    b = np.asarray(v, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def normalised_cosine(cos_ab: float, rel_a: float, rel_b: float) -> float:
    """cos(A,B) as a fraction of the ceiling that A and B's own reliabilities permit.

    Returns NaN when either reliability is non-positive: a direction that does not agree with
    itself across halves cannot meaningfully be compared to anything else, and reporting a large
    normalised value off a near-zero denominator would be the worst kind of artifact.
    """
    if not (rel_a > 0 and rel_b > 0):
        return float("nan")
    return float(cos_ab / math.sqrt(rel_a * rel_b))


# --------------------------------------------------------------------------- clustering

@dataclass(frozen=True)
class Clustered:
    """Items with the group each belongs to. `groups` is the unit of independence."""

    x: np.ndarray                 #: (n_items, n_features)
    y: np.ndarray                 #: (n_items,) 0/1 labels
    groups: np.ndarray            #: (n_items,) group id per item -- an episode id, usually

    def __post_init__(self) -> None:
        n = len(self.x)
        if not (len(self.y) == len(self.groups) == n):
            raise MeasureError(
                f"x/y/groups lengths disagree: {n}, {len(self.y)}, {len(self.groups)}")

    @property
    def unique_groups(self) -> np.ndarray:
        return np.unique(self.groups)

    def select(self, groups: Sequence) -> "Clustered":
        mask = np.isin(self.groups, np.asarray(list(groups)))
        return Clustered(self.x[mask], self.y[mask], self.groups[mask])


def split_by_group(data: Clustered, frac: float = 0.5, seed: int = 0
                   ) -> tuple[Clustered, Clustered]:
    """Split into two parts by GROUP, so no group's items straddle the boundary.

    A train/test split that let one episode appear on both sides would let the probe see a
    near-duplicate of every test item, and the held-out AUROC would be meaningless.
    """
    rng = np.random.default_rng(seed)
    gs = data.unique_groups.copy()
    rng.shuffle(gs)
    cut = max(1, int(round(len(gs) * frac)))
    if cut >= len(gs):
        raise MeasureError(f"cannot split {len(gs)} group(s) at frac={frac}: one side is empty")
    return data.select(gs[:cut]), data.select(gs[cut:])


def assert_disjoint(a: Clustered, b: Clustered) -> None:
    """Guard against leakage. Called at every train/test boundary in exp0."""
    overlap = set(a.unique_groups.tolist()) & set(b.unique_groups.tolist())
    if overlap:
        raise MeasureError(f"{len(overlap)} group(s) appear in both splits: {sorted(overlap)[:5]}")


# --------------------------------------------------------------------------- bootstrap

def bootstrap_ci(statistic: Callable[[Clustered], float], data: Clustered, *,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
                 ) -> dict[str, float]:
    """Percentile CI from a CLUSTER bootstrap: groups are resampled, never individual items.

    There is deliberately no item-level variant in this module. The only way to call this is the
    correct way.

    Resamples yielding NaN (a class absent from the draw) are dropped and counted, because
    silently treating them as 0.5 would pull every interval toward chance.
    """
    if n_boot < 1:
        raise MeasureError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    gs = data.unique_groups
    if len(gs) < 2:
        raise MeasureError(f"a cluster bootstrap needs >= 2 groups, got {len(gs)}")

    point = statistic(data)
    draws: list[float] = []
    n_degenerate = 0
    for _ in range(n_boot):
        chosen = rng.choice(gs, size=len(gs), replace=True)
        # np.isin would deduplicate a group drawn twice; concatenating preserves its weight
        parts = [data.select([g]) for g in chosen]
        resample = Clustered(
            np.concatenate([p.x for p in parts]),
            np.concatenate([p.y for p in parts]),
            np.concatenate([p.groups for p in parts]),
        )
        try:
            v = statistic(resample)
        except MeasureError:
            n_degenerate += 1
            continue
        if math.isnan(v):
            n_degenerate += 1
            continue
        draws.append(v)

    if not draws:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "n_degenerate": n_degenerate, "n_groups": len(gs)}
    arr = np.asarray(draws)
    return {
        "point": float(point),
        "lo": float(np.percentile(arr, 100 * alpha / 2)),
        "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "n_boot": len(draws),
        "n_degenerate": n_degenerate,
        "n_groups": int(len(gs)),
    }


def auroc_statistic(direction: np.ndarray) -> Callable[[Clustered], float]:
    """A statistic closure for `bootstrap_ci`: AUROC of a FIXED direction on the given data.

    The direction is fixed rather than refitted inside the bootstrap. Refitting would estimate
    the variability of the whole train-and-test procedure; what exp0 reports is the sampling
    variability of a held-out AUROC for the direction actually learned, which is the quantity the
    pre-registered threshold is stated about.
    """

    def stat(d: Clustered) -> float:
        return auroc(project(d.x, direction), d.y)

    return stat


# --------------------------------------------------------------------------- reliability

def split_half_reliability(data: Clustered, *, n_repeats: int = 20, seed: int = 0
                           ) -> dict[str, float]:
    """How well this construct agrees with itself: cosine between directions from disjoint halves.

    This is the ceiling any cross-construct cosine should be read against. Repeated over random
    halves and averaged, because a single split is noisy.
    """
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for i in range(n_repeats):
        try:
            a, b = split_by_group(data, 0.5, seed=int(rng.integers(0, 2 ** 31 - 1)))
            assert_disjoint(a, b)
            da = mean_difference_direction(a.x[a.y == 1], a.x[a.y == 0])
            db = mean_difference_direction(b.x[b.y == 1], b.x[b.y == 0])
        except MeasureError:
            continue
        vals.append(cosine(da, db))
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    arr = np.asarray(vals)
    return {"mean": float(arr.mean()), "lo": float(np.percentile(arr, 2.5)),
            "hi": float(np.percentile(arr, 97.5)), "n": len(vals)}


def shuffle_labels(data: Clustered, seed: int = 0) -> Clustered:
    """Permute labels across all items, preserving the overall label balance.

    Shuffling *within* a group would be a weaker control, because a group whose items all share
    one label would come back unchanged.
    """
    rng = np.random.default_rng(seed)
    y = data.y.copy()
    rng.shuffle(y)
    return Clustered(data.x, y, data.groups)


def shuffled_control(data: Clustered, *, n_repeats: int = 25, frac: float = 0.5,
                     seed: int = 0) -> dict[str, float]:
    """The negative control, run as a DISTRIBUTION rather than a single draw.

    A single shuffled run is itself noisy: with a fixed direction and a few dozen test groups, a
    held-out AUROC of 0.58 arises by chance often enough that a 95% interval excludes 0.5 about
    one time in twenty. Asserting the gate on one such draw would make the pipeline's own sanity
    check flaky in exactly the direction that matters -- it would intermittently claim the
    pipeline is broken when it is fine, and a reader could not tell which run they were seeing.

    So the control refits and re-evaluates over `n_repeats` independent shuffles and reports the
    distribution of held-out AUROCs. The mean is what should sit at 0.5; the spread is the
    chance variation the real measurement must be read against, and is worth reporting for that
    reason alone.
    """
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_repeats):
        s = int(rng.integers(0, 2 ** 31 - 1))
        shuffled = shuffle_labels(data, seed=s)
        try:
            tr, te = split_by_group(shuffled, frac, seed=s)
            assert_disjoint(tr, te)
            v = mean_difference_direction(tr.x[tr.y == 1], tr.x[tr.y == 0])
        except MeasureError:
            continue
        a = auroc(project(te.x, v), te.y)
        if not math.isnan(a):
            vals.append(a)
    if not vals:
        return {"mean": float("nan"), "sd": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "n": 0}
    arr = np.asarray(vals)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "lo": float(np.percentile(arr, 2.5)), "hi": float(np.percentile(arr, 97.5)),
            "n": len(vals)}
