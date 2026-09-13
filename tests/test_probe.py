"""Probe mathematics, validated on synthetic data where the answer is known in advance.

None of these tests touch a model. They check that the estimator recovers a planted separation,
lands at chance when there is none, and -- most importantly -- that the cluster bootstrap and the
train/test split respect the episode as the unit of independence. A probe pipeline that is wrong
in that last way still produces confident-looking numbers, so it is checked directly: the same
data analysed with items-as-independent must yield a visibly narrower interval than the cluster
bootstrap, and the leakage guard must actually fire.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analyze"))

from measures import (Clustered, MeasureError, assert_disjoint, auroc, auroc_statistic,  # noqa: E402
                      bootstrap_ci, cosine, mean_difference_direction, normalised_cosine,
                      project, shuffle_labels, shuffled_control, split_by_group,
                      split_half_reliability)


def make_clustered(n_groups=20, per_group=10, dim=32, sep=1.5, seed=0) -> Clustered:
    """Half the groups positive, half negative, separated along axis 0 by `sep`.

    Items within a group are near-duplicates -- exactly the structure a transcript's prefixes
    have -- so a pipeline that treats items as independent will overstate its confidence here in
    the same way it would there.
    """
    rng = np.random.default_rng(seed)
    xs, ys, gs = [], [], []
    for g in range(n_groups):
        label = g % 2
        centre = rng.normal(0, 1, dim)
        centre[0] += sep if label else -sep
        xs.append(centre + rng.normal(0, 0.05, (per_group, dim)))   # tight cluster
        ys.extend([label] * per_group)
        gs.extend([f"ep{g}"] * per_group)
    return Clustered(np.vstack(xs), np.asarray(ys), np.asarray(gs))


# --------------------------------------------------------------------------- AUROC

def test_auroc_perfect_separation():
    assert auroc([0.0, 0.1, 0.9, 1.0], [0, 0, 1, 1]) == 1.0


def test_auroc_inverted_separation():
    assert auroc([0.0, 0.1, 0.9, 1.0], [1, 1, 0, 0]) == 0.0


def test_auroc_all_ties_is_chance():
    assert auroc([1.0] * 6, [0, 0, 0, 1, 1, 1]) == 0.5


def test_auroc_single_class_is_nan_not_half():
    """A degenerate bootstrap draw must propagate, not silently become chance."""
    assert np.isnan(auroc([1.0, 2.0], [1, 1]))


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    s = rng.normal(size=200)
    y = (rng.random(200) < 0.4).astype(int)
    assert auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_auroc_length_mismatch_raises():
    with pytest.raises(MeasureError):
        auroc([1.0, 2.0], [1])


# --------------------------------------------------------------------------- directions

def test_direction_recovers_a_planted_axis():
    d = make_clustered(sep=3.0)
    v = mean_difference_direction(d.x[d.y == 1], d.x[d.y == 0])
    assert abs(v[0]) > 0.9, "the planted axis should dominate the direction"
    assert np.linalg.norm(v) == pytest.approx(1.0)


def test_direction_separates_what_it_was_trained_on():
    d = make_clustered(sep=2.0)
    v = mean_difference_direction(d.x[d.y == 1], d.x[d.y == 0])
    assert auroc(project(d.x, v), d.y) > 0.95


def test_direction_on_unseparated_data_is_at_chance():
    d = make_clustered(sep=0.0, seed=7)
    tr, te = split_by_group(d, 0.5, seed=1)
    v = mean_difference_direction(tr.x[tr.y == 1], tr.x[tr.y == 0])
    assert 0.2 < auroc(project(te.x, v), te.y) < 0.8


def test_degenerate_direction_raises():
    x = np.ones((4, 3))
    with pytest.raises(MeasureError):
        mean_difference_direction(x[:2], x[2:])


def test_empty_class_raises():
    with pytest.raises(MeasureError):
        mean_difference_direction(np.zeros((0, 3)), np.ones((2, 3)))


# --------------------------------------------------------------------------- cosine

def test_cosine_basics():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)
    assert cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert np.isnan(cosine([0, 0, 0], [1, 0, 0]))


def test_random_high_dimensional_vectors_are_near_orthogonal():
    """Why a bare cosine is uninterpretable: chance alignment is ~0 in high dimensions, so any
    positive value looks impressive until it is compared against a reliability ceiling."""
    rng = np.random.default_rng(0)
    vals = [abs(cosine(rng.normal(size=2048), rng.normal(size=2048))) for _ in range(50)]
    assert max(vals) < 0.12


def test_normalised_cosine_divides_by_the_reliability_ceiling():
    assert normalised_cosine(0.4, 0.8, 0.8) == pytest.approx(0.5)
    assert normalised_cosine(0.5, 1.0, 1.0) == pytest.approx(0.5)


def test_normalised_cosine_refuses_a_non_positive_denominator():
    """A direction that does not agree with itself cannot be compared to anything."""
    assert np.isnan(normalised_cosine(0.9, 0.0, 0.8))
    assert np.isnan(normalised_cosine(0.9, -0.1, 0.8))


# --------------------------------------------------------------------------- splitting

def test_split_is_by_group_and_disjoint():
    d = make_clustered(n_groups=10)
    a, b = split_by_group(d, 0.5, seed=0)
    assert_disjoint(a, b)
    assert len(a.unique_groups) + len(b.unique_groups) == 10
    assert len(a.x) + len(b.x) == len(d.x)


def test_split_keeps_every_item_of_a_group_together():
    d = make_clustered(n_groups=8, per_group=5)
    a, b = split_by_group(d, 0.5, seed=2)
    for part in (a, b):
        for g in part.unique_groups:
            assert int((d.groups == g).sum()) == int((part.groups == g).sum())


def test_leakage_guard_actually_fires():
    d = make_clustered(n_groups=6)
    with pytest.raises(MeasureError):
        assert_disjoint(d, d)


def test_split_refuses_when_one_side_would_be_empty():
    d = make_clustered(n_groups=3)
    with pytest.raises(MeasureError):
        split_by_group(d, 1.0)


def test_held_out_auroc_is_honest_about_a_null():
    """Train and test on disjoint episodes of unseparated data: no illusion of signal."""
    d = make_clustered(sep=0.0, n_groups=40, seed=11)
    tr, te = split_by_group(d, 0.5, seed=5)
    assert_disjoint(tr, te)
    v = mean_difference_direction(tr.x[tr.y == 1], tr.x[tr.y == 0])
    ci = bootstrap_ci(auroc_statistic(v), te, n_boot=300, seed=0)
    assert ci["lo"] < 0.5 < ci["hi"], "a null must produce an interval covering chance"


# --------------------------------------------------------------------------- bootstrap

def test_bootstrap_resamples_groups_not_items():
    """The check that matters: clustering must widen the interval relative to treating
    near-duplicate items as independent observations.

    A FIXED direction and genuinely overlapping groups are used, so the statistic sits at an
    intermediate value and the two intervals have width to compare. A fitted direction on easy
    data pins both to 1.0 and the comparison becomes vacuous.
    """
    d = make_clustered(n_groups=12, per_group=20, sep=0.3, seed=4)
    v = np.zeros(d.x.shape[1])
    v[0] = 1.0                              # the true planted axis, not fitted to this sample

    clustered = bootstrap_ci(auroc_statistic(v), d, n_boot=500, seed=0)
    fake = Clustered(d.x, d.y, np.arange(len(d.x)).astype(str))   # every item its own "episode"
    item_level = bootstrap_ci(auroc_statistic(v), fake, n_boot=500, seed=0)

    assert (clustered["hi"] - clustered["lo"]) > (item_level["hi"] - item_level["lo"]), (
        "the cluster bootstrap must be wider than the pseudo-replicated one")
    assert clustered["n_groups"] == 12


def test_bootstrap_is_deterministic_given_a_seed():
    d = make_clustered(n_groups=10)
    v = mean_difference_direction(d.x[d.y == 1], d.x[d.y == 0])
    a = bootstrap_ci(auroc_statistic(v), d, n_boot=100, seed=42)
    b = bootstrap_ci(auroc_statistic(v), d, n_boot=100, seed=42)
    assert a == b


def test_bootstrap_counts_degenerate_draws_rather_than_hiding_them():
    d = make_clustered(n_groups=2, per_group=4)
    v = mean_difference_direction(d.x[d.y == 1], d.x[d.y == 0])
    out = bootstrap_ci(auroc_statistic(v), d, n_boot=200, seed=0)
    assert out["n_degenerate"] > 0          # some draws pick one group twice -> one class only
    assert out["n_boot"] + out["n_degenerate"] == 200


def test_bootstrap_needs_at_least_two_groups():
    d = make_clustered(n_groups=2)
    single = d.select([d.unique_groups[0]])
    v = np.ones(d.x.shape[1]) / np.sqrt(d.x.shape[1])
    with pytest.raises(MeasureError):
        bootstrap_ci(auroc_statistic(v), single, n_boot=10)


def test_bootstrap_preserves_the_weight_of_a_group_drawn_twice():
    """np.isin would deduplicate; the resample must keep the duplicate's weight."""
    d = make_clustered(n_groups=4, per_group=3)
    sizes = []

    def stat(sample: Clustered) -> float:
        sizes.append(len(sample.x))
        return 0.5

    bootstrap_ci(stat, d, n_boot=30, seed=1)
    assert set(sizes) == {12}, "every resample must have as many items as the original"


# --------------------------------------------------------------------------- controls

def test_shuffled_control_lands_at_chance_on_average():
    """The negative control the gate depends on. If this fails, no exp0 number is reportable.

    Averaged over repeated shuffles, because a single shuffled draw excludes 0.5 roughly one
    time in twenty by construction -- which would make the pipeline's own sanity check flaky.
    """
    d = make_clustered(sep=3.0, n_groups=40, seed=9)
    out = shuffled_control(d, n_repeats=25, seed=0)
    assert out["n"] == 25
    assert abs(out["mean"] - 0.5) < 0.05, f"shuffled labels must average to chance, got {out}"


def test_shuffled_control_is_noisy_enough_to_be_worth_reporting():
    """The spread is the chance variation a real AUROC must be read against."""
    d = make_clustered(sep=3.0, n_groups=40, seed=9)
    out = shuffled_control(d, n_repeats=25, seed=0)
    assert out["sd"] > 0.01


def test_a_single_shuffled_draw_is_not_a_reliable_control():
    """Documents why `shuffled_control` averages rather than taking one draw.

    The claim is about spread, not about any particular draw: single shuffled runs scatter around
    chance widely enough that a 95% interval built from one of them excludes 0.5 a noticeable
    fraction of the time. Asserting on the standard deviation states that directly, instead of
    on a max that would be a constant tuned to whichever seed happened to be used.
    """
    d = make_clustered(sep=3.0, n_groups=40, seed=9)
    singles = []
    for s in range(30):
        tr, te = split_by_group(shuffle_labels(d, seed=s), 0.5, seed=s)
        v = mean_difference_direction(tr.x[tr.y == 1], tr.x[tr.y == 0])
        singles.append(auroc(project(te.x, v), te.y))
    arr = np.asarray(singles)
    assert arr.mean() == pytest.approx(0.5, abs=0.05), "single draws must still centre on chance"
    assert arr.std(ddof=1) > 0.02, (
        "if single draws had negligible spread, averaging would be unnecessary -- "
        f"got sd={arr.std(ddof=1):.4f}")


def test_shuffling_preserves_the_label_balance():
    d = make_clustered(n_groups=10)
    s = shuffle_labels(d, seed=0)
    assert s.y.sum() == d.y.sum()


def test_split_half_reliability_is_high_for_a_strong_direction():
    d = make_clustered(sep=3.0, n_groups=40, seed=2)
    rel = split_half_reliability(d, n_repeats=10, seed=0)
    assert rel["mean"] > 0.8 and rel["n"] == 10


def test_split_half_reliability_is_low_for_noise():
    d = make_clustered(sep=0.0, n_groups=40, seed=13)
    rel = split_half_reliability(d, n_repeats=10, seed=0)
    assert rel["mean"] < 0.5


def test_clustered_rejects_mismatched_lengths():
    with pytest.raises(MeasureError):
        Clustered(np.zeros((4, 2)), np.zeros(3), np.zeros(4))
