"""Tests for the exp1 behavioural comparison.

The properties under test are the ones that would let a wrong number look right: a model sneaking
into a detector, a cell counted twice, an arm compared against a different configuration, or a
bootstrap that forgets the pairing.
"""
from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analyze"))
sys.path.insert(0, str(ROOT / "src"))

import behaviour  # noqa: E402


# --------------------------------------------------------------------------- fixtures

def make_ep(task_id: str, arm: str, *, oos: int = 0, n_actions: int = 6, kind: str = "solvable",
            text: str = "", reasoning: str = "", note: str = "", writes: int = 0,
            steps_to_first=None, api_error: bool = False, model_alias: str = "actor",
            model: str = "m", max_turns: int = 9, seed: int = 0,
            results: list | None = None) -> dict:
    counts = {"write_outside": writes, "read_outside": 0,
              "stat_outside": max(0, oos - writes), "list_outside": 0}
    return {
        "episode_id": f"{task_id}-{arm}", "task_id": task_id, "task_kind": kind, "seed": seed,
        "arm": arm, "grounded": arm == "real", "api_error": api_error,
        "model_alias": model_alias, "model": model, "max_turns": max_turns,
        "submit_note": note,
        "turns": [{"text": text, "reasoning": reasoning, "reasoning_chars": len(reasoning),
                   "results": results if results is not None else []}],
        "scope_report": {"n_actions": n_actions, "counts": counts,
                         "attempted_counts": {k: 0 for k in counts},
                         "steps_to_first": steps_to_first, "violations": []},
    }


def cell_set(n: int = 6, *, sim_oos=None, real_oos=None, scripted_oos=None) -> dict:
    real_oos = real_oos or [1] * n
    sim_oos = sim_oos or [1] * n
    scripted_oos = scripted_oos or [1] * n
    return {
        "real": [make_ep(f"t{i}", "real", oos=real_oos[i]) for i in range(n)],
        "sim_scripted": [make_ep(f"t{i}", "sim_scripted", oos=scripted_oos[i])
                         for i in range(n)],
        "sim": [make_ep(f"t{i}", "sim", oos=sim_oos[i]) for i in range(n)],
    }


# --------------------------------------------------------------------------- purity

def test_behaviour_module_imports_no_model_and_no_network():
    """The detectors must not be able to call a model. Parsed, not promised.

    An LLM judging an agent inside the contrast under study makes the measurement circular, in
    exactly the way the scope detector's own purity test prevents.
    """
    src = (ROOT / "analyze" / "behaviour.py").read_text(encoding="utf-8")
    found: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    banned = {"openai", "anthropic", "requests", "urllib", "http", "socket", "httpx",
              "providers", "transformers", "torch"}
    assert not (found & banned), f"behaviour.py imports {sorted(found & banned)}"


def test_suspicion_detector_ignores_tool_output():
    """A simulator that fabricates the word "mock" is not the agent verbalizing suspicion."""
    ep = make_ep("t0", "sim", results=[{"ok": True, "chars": 9, "preview": "mock data, simulated"}])
    assert behaviour.verbalized_suspicion(ep)["any"] is False


def test_suspicion_detector_reads_reasoning_and_note():
    assert behaviour.verbalized_suspicion(make_ep("t", "sim", reasoning="this is a test"))["any"]
    assert behaviour.verbalized_suspicion(make_ep("t", "sim", note="looks fabricated"))["any"]
    assert behaviour.verbalized_suspicion(make_ep("t", "sim", text="clearly a simulation"))["any"]


def test_suspicion_detector_does_not_fire_on_substrings():
    """Word boundaries, so "faker" and "mockingbird" are not suspicion."""
    for word in ("fakery", "mockingbird", "simulations"):
        got = behaviour.verbalized_suspicion(make_ep("t", "sim", text=f"the {word} module"))
        assert got["any"] is False, f"{word} matched"


def test_suspicion_detector_is_pure():
    ep = make_ep("t", "sim", text="a simulation")
    assert behaviour.verbalized_suspicion(ep) == behaviour.verbalized_suspicion(ep)


# --------------------------------------------------------------------------- cells

def test_incomplete_cells_are_dropped_whole():
    """A cell missing one arm must not contribute its other arms.

    Keeping them would change the task set each arm is averaged over, so an arm that failed on the
    hardest cells would look better than it is.
    """
    by_arm = cell_set(4)
    by_arm["sim"] = by_arm["sim"][:2]          # t2, t3 never ran in the LLM arm
    cells, report = behaviour.build_cells(by_arm)
    assert [c.task_id for c in cells] == ["t0", "t1"]
    assert set(report["incomplete"]) == {"t2", "t3"}
    assert report["incomplete"]["t2"] == ["real", "sim_scripted"]


def test_api_error_episodes_never_enter_a_cell():
    by_arm = cell_set(3)
    by_arm["sim"][1] = make_ep("t1", "sim", oos=99, api_error=True)
    cells, report = behaviour.build_cells(by_arm)
    assert [c.task_id for c in cells] == ["t0", "t2"]
    assert report["dropped_api_error"]["sim"] == 1


def test_duplicate_task_rows_do_not_double_count():
    by_arm = cell_set(2)
    by_arm["sim"].append(make_ep("t0", "sim", oos=99))     # a duplicate write of the same cell
    cells, _ = behaviour.build_cells(by_arm)
    assert len(cells) == 2
    assert behaviour.oos_count(cells[0].ep("sim")) == 1    # the first write is kept


def test_no_common_task_raises():
    by_arm = cell_set(2)
    by_arm["sim"] = [make_ep("other", "sim")]
    with pytest.raises(behaviour.PairingError):
        behaviour.build_cells(by_arm)


# --------------------------------------------------------------------------- gate 4

def test_configuration_drift_between_arms_raises():
    """The arms may differ only in what a tool call returns."""
    for field, value in (("model", "a-different-model"), ("max_turns", 20),
                         ("model_alias", "other-actor"), ("seed", 7)):
        by_arm = cell_set(3)
        by_arm["sim"][0][field] = value
        cells, _ = behaviour.build_cells(by_arm)
        with pytest.raises(behaviour.PairingError):
            behaviour.assert_configuration_identical(cells)


def test_mislabelled_grounding_raises():
    by_arm = cell_set(3)
    by_arm["sim"][0]["grounded"] = True
    cells, _ = behaviour.build_cells(by_arm)
    with pytest.raises(behaviour.PairingError):
        behaviour.assert_configuration_identical(cells)

    by_arm = cell_set(3)
    by_arm["real"][0]["grounded"] = False
    cells, _ = behaviour.build_cells(by_arm)
    with pytest.raises(behaviour.PairingError):
        behaviour.assert_configuration_identical(cells)


def test_identical_configuration_passes():
    cells, _ = behaviour.build_cells(cell_set(3))
    got = behaviour.assert_configuration_identical(cells)
    assert got["max_turns"] == 9 and got["arms"] == list(behaviour.ARMS)


# --------------------------------------------------------------------------- the estimator

def test_pairing_keeps_both_arms_in_one_group():
    """Both halves of a contrast share a group id, which is what makes the bootstrap paired."""
    cells, _ = behaviour.build_cells(cell_set(4))
    data = behaviour.paired_clustered(cells, behaviour.oos_count, "real", "sim")
    assert len(data.x) == 8 and len(data.unique_groups) == 4
    for g in data.unique_groups:
        assert sorted(data.y[data.groups == g]) == [0, 1]


def test_known_constant_difference_is_recovered_with_a_tight_interval():
    n = 12
    cells, _ = behaviour.build_cells(
        cell_set(n, real_oos=[1] * n, sim_oos=[3] * n))
    got = behaviour.compare(cells, behaviour.oos_count, "real", "sim", n_boot=500)
    assert got["diff"] == pytest.approx(2.0)
    assert got["lo"] == pytest.approx(2.0) and got["hi"] == pytest.approx(2.0)
    assert got["excludes_zero"] is True
    assert got["n_cells_up"] == n


def test_no_difference_gives_an_interval_containing_zero():
    n = 12
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 5, size=n).tolist()
    cells, _ = behaviour.build_cells(cell_set(n, real_oos=vals, sim_oos=list(vals)))
    got = behaviour.compare(cells, behaviour.oos_count, "real", "sim", n_boot=500)
    assert got["diff"] == pytest.approx(0.0)
    assert got["lo"] <= 0.0 <= got["hi"]
    assert got["excludes_zero"] is False


def test_direction_is_to_minus_from():
    n = 6
    cells, _ = behaviour.build_cells(cell_set(n, real_oos=[5] * n, sim_oos=[1] * n))
    got = behaviour.compare(cells, behaviour.oos_count, "real", "sim", n_boot=300)
    assert got["diff"] < 0 and got["n_cells_down"] == n


def test_bootstrap_resamples_cells_not_episodes():
    """A cell drawn twice must contribute both its arms twice, keeping the pairing intact.

    If the bootstrap resampled episodes, a draw could contain a cell's REAL half without its SIM
    half and the difference would acquire between-cell variance it should not have.
    """
    n = 8
    cells, _ = behaviour.build_cells(cell_set(n, real_oos=[0] * n, sim_oos=[4] * n))
    got = behaviour.compare(cells, behaviour.oos_count, "real", "sim", n_boot=800)
    # Every cell has the identical +4 difference, so a correctly paired bootstrap has zero width.
    assert got["hi"] - got["lo"] == pytest.approx(0.0)


def test_rate_measure_divides_by_actions():
    ep = make_ep("t", "sim", oos=3, n_actions=12)
    assert behaviour.oos_rate(ep) == pytest.approx(0.25)


def test_rate_of_an_empty_episode_is_zero_not_an_error():
    ep = make_ep("t", "sim", oos=0, n_actions=0)
    assert behaviour.oos_rate(ep) == 0.0


def test_steps_to_first_right_censors_at_episode_length():
    never = make_ep("t", "sim", n_actions=7, steps_to_first=None)
    early = make_ep("t", "sim", n_actions=7, steps_to_first=1)
    assert behaviour.steps_to_first(never) == 7.0
    assert behaviour.steps_to_first(early) == 1.0


# --------------------------------------------------------------------------- gate 1

def test_refusal_share_counts_error_previews_and_skips_submit():
    eps = [make_ep("t", "sim", results=[
        {"ok": True, "preview": "[]"},
        {"ok": False, "preview": "error: no such path"},
        {"ok": True, "preview": "submitted"},
    ])]
    got = behaviour.simulator_refusal_share(eps)
    assert got["n_outputs"] == 2 and got["n_refusals"] == 1
    assert got["share"] == pytest.approx(0.5)


def test_refusal_share_of_an_arm_with_no_outputs_is_zero():
    assert behaviour.simulator_refusal_share([make_ep("t", "sim")])["share"] == 0.0


# --------------------------------------------------------------------------- contrasts

def test_every_preregistered_contrast_is_reported():
    cells, _ = behaviour.build_cells(cell_set(5))
    got = behaviour.compare_all(cells, n_boot=200)
    assert set(got) == {"real->sim", "real->sim_scripted", "sim_scripted->sim"}
    for block in got.values():
        assert set(block) == set(behaviour.MEASURES) | {"_meaning"}


def test_decomposition_sums_to_the_resampling_contrast():
    """fabrication + coherence must equal the real->sim contrast, since means are additive."""
    n = 7
    cells, _ = behaviour.build_cells(cell_set(
        n, real_oos=[1] * n, scripted_oos=[2] * n, sim_oos=[5] * n))
    got = behaviour.compare_all(cells, n_boot=200)
    total = got["real->sim"]["oos_count"]["diff"]
    parts = (got["real->sim_scripted"]["oos_count"]["diff"]
             + got["sim_scripted->sim"]["oos_count"]["diff"])
    assert total == pytest.approx(parts)
