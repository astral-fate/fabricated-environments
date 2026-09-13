"""Resume invariants, ported from `project-v2/runner/test_resume.py`.

Each case corresponds to something that actually went wrong in paper 1, and the invariant that
matters for every rate this paper reports is the last one: one episode per (task, arm, model)
cell, ever.

  torn final line      a kill mid-write must not be read as a completed episode and must not be
                       fatal.
  corrupt state file   `state.json` was trusted once and replayed six finished episodes, writing
                       six duplicate pairs. Progress is rebuilt from the append-only log; the
                       state file is only ever additive to it.
  deleted state file   losing it must cost nothing.
  api_error episodes   must NOT count as done -- they are retried. Counting a zero-turn 429 as a
                       failure once produced a false null.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from agent import (append_episode, completed_keys, episode_key, load_episodes,  # noqa: E402
                   run_episode, save_state)
from arms.real import RealArm  # noqa: E402
from mock_provider import FlakyProvider, ScriptedProvider, StallingProvider  # noqa: E402
from task import build_task  # noqa: E402

SOLVE_SCRIPT = [
    ("listing", [("list_dir", {"path": "workspace"})]),
    ("reading", [("read_file", {"path": "workspace/components/component_0.conf"})]),
    ("writing", [("write_file", {"path": "workspace/manifest.txt", "content": "x"})]),
    ("done", [("submit", {"note": "finished"})]),
]


def _run(tmp_path, task=None, provider=None, name="e"):
    task = task or build_task("solvable", 0)
    arm = RealArm(root=tmp_path / f"root-{name}", log_path=tmp_path / f"ev-{name}.jsonl")
    return run_episode(provider=provider or ScriptedProvider(SOLVE_SCRIPT), task=task, arm=arm,
                       model_alias="mock")


# --------------------------------------------------------------------------- the loop

def test_a_clean_episode_records_turns_actions_and_a_scope_report(tmp_path):
    ep = _run(tmp_path)
    assert ep.api_error == ""
    assert ep.submitted is True
    assert len(ep.turns) == 4
    assert len(ep.actions) == 3            # submit is not an arm action
    assert ep.scope_report["any_violation"] is False
    assert ep.arm == "real" and ep.grounded is True


def test_scope_report_is_recomputable_from_the_recorded_actions(tmp_path):
    """An archived episode must be re-scorable without the substrate that produced it."""
    from scope import score_episode
    ep = _run(tmp_path)
    assert score_episode(ep.actions).to_dict() == ep.scope_report


def test_violating_episode_is_scored(tmp_path):
    script = [("peek", [("read_file", {"path": "secrets/.env"})]),
              ("done", [("submit", {"note": "read it"})])]
    ep = _run(tmp_path, task=build_task("tempting", 0), provider=ScriptedProvider(script))
    assert ep.scope_report["counts"]["read_outside"] == 1
    assert ep.scope_report["steps_to_first"] == 0


def test_reasoning_is_recorded_with_its_true_length(tmp_path):
    """`r_cot` reads this channel; discarding it would remove the paper's evidence."""
    thoughts = ["this environment smells synthetic", "but I will proceed"]
    ep = _run(tmp_path, provider=ScriptedProvider(SOLVE_SCRIPT, reasoning=thoughts))
    assert ep.turns[0].reasoning == thoughts[0]
    assert ep.turns[0].reasoning_chars == len(thoughts[0])


def test_a_stalling_provider_ends_the_episode_rather_than_burning_the_budget(tmp_path):
    arm = RealArm(root=tmp_path / "root", log_path=tmp_path / "ev.jsonl")
    ep = run_episode(provider=StallingProvider([]), task=build_task("solvable", 0), arm=arm,
                     model_alias="mock", max_turns=20)
    assert ep.stalled is True
    assert len(ep.turns) < 20


def test_success_reflects_the_task_check(tmp_path):
    task = build_task("solvable", 0)
    body = "\n".join(task.required_values)
    script = [("write", [("write_file", {"path": task.target, "content": body})]),
              ("done", [("submit", {"note": "ok"})])]
    ep = _run(tmp_path, task=task, provider=ScriptedProvider(script))
    assert ep.success is True


def test_blocked_task_cannot_succeed_however_the_agent_behaves(tmp_path):
    task = build_task("blocked", 0)
    everything = "\n".join(task.files.values())
    script = [("write", [("write_file", {"path": task.target, "content": everything[:8000]})]),
              ("done", [("submit", {"note": "tried"})])]
    ep = _run(tmp_path, task=task, provider=ScriptedProvider(script))
    assert ep.success is False


# --------------------------------------------------------------------------- api_error

def test_api_error_is_recorded_not_raised(tmp_path):
    ep = _run(tmp_path, provider=FlakyProvider(SOLVE_SCRIPT, fail_at=0))
    assert ep.api_error.startswith("RuntimeError")
    assert ep.submitted is False


def test_api_error_episodes_do_not_count_as_done(tmp_path):
    """Counting a zero-turn 429 as a failure produced a false null in paper 1."""
    log = tmp_path / "episodes.jsonl"
    ep = _run(tmp_path, provider=FlakyProvider(SOLVE_SCRIPT, fail_at=0))
    append_episode(log, ep)
    assert completed_keys(log) == set()

    good = _run(tmp_path, name="g")
    append_episode(log, good)
    assert completed_keys(log) == {episode_key(good.task_id, good.arm, good.model_alias)}


def test_partial_progress_before_an_api_error_is_still_recorded(tmp_path):
    ep = _run(tmp_path, provider=FlakyProvider(SOLVE_SCRIPT, fail_at=2))
    assert ep.api_error
    assert len(ep.actions) == 2            # the two calls that landed before the failure


# --------------------------------------------------------------------------- the log

def test_torn_final_line_is_skipped_not_fatal(tmp_path):
    log = tmp_path / "episodes.jsonl"
    append_episode(log, _run(tmp_path, name="a"))
    with log.open("a", encoding="utf-8") as fh:
        fh.write('{"episode_id": "torn", "task_id": "sol')     # killed mid-write
    assert len(load_episodes(log)) == 1
    assert len(completed_keys(log)) == 1


def test_blank_lines_are_tolerated(tmp_path):
    log = tmp_path / "episodes.jsonl"
    append_episode(log, _run(tmp_path, name="a"))
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    assert len(load_episodes(log)) == 1


def test_missing_log_is_empty_not_an_error(tmp_path):
    assert load_episodes(tmp_path / "nope.jsonl") == []
    assert completed_keys(tmp_path / "nope.jsonl") == set()


def test_log_is_append_only_across_runs(tmp_path):
    log = tmp_path / "episodes.jsonl"
    append_episode(log, _run(tmp_path, name="a"))
    first = log.read_text(encoding="utf-8")
    append_episode(log, _run(tmp_path, task=build_task("blocked", 1), name="b"))
    assert log.read_text(encoding="utf-8").startswith(first), "earlier lines were rewritten"
    assert len(load_episodes(log)) == 2


# --------------------------------------------------------------------------- state file

def test_a_corrupt_state_file_costs_nothing(tmp_path):
    """The log is the source of truth; state.json is advisory and never read back for progress."""
    log, state = tmp_path / "episodes.jsonl", tmp_path / "state.json"
    ep = _run(tmp_path)
    append_episode(log, ep)
    save_state(state, completed_keys(log))
    state.write_text('{"completed": ["gar', encoding="utf-8")      # truncated by a hard kill
    assert completed_keys(log) == {episode_key(ep.task_id, ep.arm, ep.model_alias)}


def test_a_deleted_state_file_costs_nothing(tmp_path):
    log, state = tmp_path / "episodes.jsonl", tmp_path / "state.json"
    ep = _run(tmp_path)
    append_episode(log, ep)
    save_state(state, completed_keys(log))
    state.unlink()
    assert completed_keys(log) == {episode_key(ep.task_id, ep.arm, ep.model_alias)}


def test_a_lying_state_file_cannot_mark_work_done(tmp_path):
    """This is the failure that replayed six finished episodes in paper 1."""
    log, state = tmp_path / "episodes.jsonl", tmp_path / "state.json"
    save_state(state, {episode_key("solvable-s9", "real", "mock")})
    assert completed_keys(log) == set(), "progress must come from the log, not the state file"


def test_state_write_is_atomic(tmp_path):
    state = tmp_path / "state.json"
    save_state(state, {"a"})
    save_state(state, {"a", "b"})
    assert not list(tmp_path.glob("*.tmp")), "a temp file was left behind"
    assert json.loads(state.read_text(encoding="utf-8"))["completed"] == ["a", "b"]


# --------------------------------------------------------------------------- no duplicates

def test_resuming_never_reruns_a_completed_cell(tmp_path):
    """The invariant every rate depends on: one episode per (task, arm, model) cell."""
    log = tmp_path / "episodes.jsonl"
    tasks = [build_task("solvable", 0), build_task("blocked", 1), build_task("tempting", 2)]

    def sweep(stop_after: int | None = None) -> int:
        done = completed_keys(log)
        ran = 0
        for i, t in enumerate(tasks):
            if episode_key(t.task_id, "real", "mock") in done:
                continue
            if stop_after is not None and ran >= stop_after:
                break                       # simulate a kill part-way through
            append_episode(log, _run(tmp_path, task=t, name=f"{i}-{ran}"))
            ran += 1
        return ran

    assert sweep(stop_after=2) == 2
    assert sweep() == 1                     # only the unfinished cell is redone
    assert sweep() == 0                     # and nothing at all on a third pass

    eps = load_episodes(log)
    keys = [episode_key(e["task_id"], e["arm"], e["model_alias"]) for e in eps]
    assert len(keys) == len(set(keys)) == 3


def test_arms_produce_distinct_cells(tmp_path):
    """A REAL and a SIM episode of the same task are different cells and must both run."""
    log = tmp_path / "episodes.jsonl"
    ep = _run(tmp_path)
    append_episode(log, ep)
    assert episode_key(ep.task_id, "sim", "mock") not in completed_keys(log)


# --------------------------------------------------------------------------- dedupe and locking

def test_duplicate_cells_are_collapsed_to_one_observation(tmp_path):
    """Two writers on one log can both run the same cell. That happened here: an interrupted run
    was relaunched while the old process was still finishing, and two cells were written twice.

    A duplicate is not extra evidence -- it is the same cell twice, and the episode is the unit of
    independence for every interval in this project, so keeping both would inflate n with a
    correlated observation.
    """
    from agent import dedupe_episodes

    log = tmp_path / "episodes.jsonl"
    task = build_task("blocked", 4)
    first = _run(tmp_path, task=task, name="a")
    second = _run(tmp_path, task=task, name="b")
    append_episode(log, first)
    append_episode(log, second)
    assert first.episode_id != second.episode_id

    eps = load_episodes(log)
    assert len(eps) == 2
    deduped = dedupe_episodes(eps)
    assert len(deduped) == 1
    assert deduped[0]["episode_id"] == first.episode_id, "the FIRST written must be kept"


def test_dedupe_keeps_distinct_cells_and_drops_api_errors(tmp_path):
    from agent import dedupe_episodes

    log = tmp_path / "episodes.jsonl"
    append_episode(log, _run(tmp_path, task=build_task("solvable", 0), name="a"))
    append_episode(log, _run(tmp_path, task=build_task("blocked", 1), name="b"))
    append_episode(log, _run(tmp_path, task=build_task("tempting", 2),
                             provider=FlakyProvider(SOLVE_SCRIPT, fail_at=0), name="c"))
    deduped = dedupe_episodes(load_episodes(log))
    assert len(deduped) == 2
    assert all(not e["api_error"] for e in deduped)


def test_dedupe_separates_arms_and_actors(tmp_path):
    """Same task under a different arm or actor is a DIFFERENT cell and must survive."""
    from agent import dedupe_episodes

    ep = _run(tmp_path)
    a = json.loads(ep.to_json())
    b = dict(a, arm="sim", episode_id="x")
    c = dict(a, model_alias="other", episode_id="y")
    assert len(dedupe_episodes([a, b, c])) == 3


def test_log_lock_excludes_a_second_writer(tmp_path):
    """The race that produced the duplicates, prevented rather than repaired."""
    from agent import LogBusy, lock_log

    log = tmp_path / "episodes.jsonl"
    with lock_log(log):
        with pytest.raises(LogBusy):
            with lock_log(log):
                pass


def test_log_lock_is_released_on_exit_and_on_error(tmp_path):
    from agent import lock_log

    log = tmp_path / "episodes.jsonl"
    with lock_log(log):
        pass
    with lock_log(log):            # re-acquirable
        pass

    try:
        with lock_log(log):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    with lock_log(log):            # released even when the body raised
        pass
    assert not (tmp_path / "episodes.jsonl.lock").exists()


def test_log_lock_names_the_holder_so_a_stale_lock_can_be_cleared(tmp_path):
    from agent import LogBusy, lock_log

    log = tmp_path / "episodes.jsonl"
    with lock_log(log):
        with pytest.raises(LogBusy) as exc:
            with lock_log(log):
                pass
        assert "pid" in str(exc.value)


def test_a_lock_held_by_a_dead_process_is_reclaimed(tmp_path):
    """The supervisor kills children by design, so they never release the lock. Refusing to
    reclaim a dead holder's lock blocked every subsequent resume and cost two run windows."""
    from agent import lock_log

    log = tmp_path / "episodes.jsonl"
    stale = tmp_path / "episodes.jsonl.lock"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("999999", encoding="utf-8")     # a pid that is not running
    with lock_log(log):
        assert stale.exists()
    assert not stale.exists()


def test_a_lock_with_an_unparseable_holder_is_reclaimed(tmp_path):
    from agent import lock_log

    log = tmp_path / "episodes.jsonl"
    (tmp_path / "episodes.jsonl.lock").write_text("", encoding="utf-8")
    with lock_log(log):
        pass


def test_pid_liveness_probe_does_not_kill_the_process_it_asks_about(tmp_path):
    """On Windows `os.kill(pid, 0)` terminates the target. Our own pid must survive the probe."""
    import os
    from agent import _pid_alive

    assert _pid_alive(os.getpid()) is True
    assert _pid_alive(os.getpid()) is True          # still alive after being asked twice
    assert _pid_alive(999999) is False
