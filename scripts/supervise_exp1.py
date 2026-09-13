"""Supervise the exp1 collection so a wedged run costs minutes, not a night.

    python scripts/supervise_exp1.py                     # all three arms, in parallel
    python scripts/supervise_exp1.py --arms real         # one arm only

Everything after `--` is passed straight to `experiments/exp1_behavioural_divergence.py`.

Why this is not `scripts/supervise.py`
--------------------------------------
That module supervises exp0, and its failure signatures are Groq's: a daily token budget scoped to
an organisation, key rotation, cooldowns measured against a refill rate. exp1 runs on OpenRouter
with a prepaid balance and three arms, so none of those signals exist here and the ones that matter
are different. The repository's own rule -- stated in `sync_from_project_v2.py` and followed by
`supervise.py` itself -- is that a module which must differ gets its own file and a docstring
saying what diverged, rather than a flag that makes one file mean two things.

One child per arm, and why that is safe
---------------------------------------
Collection is bound by actor inference latency, not by anything local: an episode is several
minutes of a reasoning model generating tokens on someone else's GPU, and the three arms have no
interaction. Running them sequentially wastes wall-clock for no benefit, so this launches one child
per arm.

The resume logic assumes a SINGLE WRITER per episode log -- progress is rebuilt from the log, so two
processes appending to one file can both decide a cell is outstanding and both run it, which is how
this project produced duplicate episodes once before. Sharding by arm preserves that assumption
instead of weakening it: each child owns `episodes-<arm>.jsonl` and its own lock, and the analysis
reads every shard back afterwards. Nothing about the measurement depends on how collection was
scheduled.

What it watches, per child
--------------------------
  stall        no new episode for `--stall-seconds`. A wedged provider call is the failure this run
               is actually exposed to: the OpenRouter client retries with a 180 s timeout, so a bad
               request can sit for minutes producing nothing. Kill and relaunch -- resume is
               append-only and covered by `tests/test_resume.py`, so a restart costs at most the
               episode in flight. The lock the killed child held is reclaimed automatically,
               because `lock_log` reclaims a lock whose recorded pid is dead.

  crash        child exits non-zero. Relaunch, up to `--max-restarts`.

  no progress  a relaunch that produces no new cell is a loop, not a recovery. Counted separately
               and given up on sooner, because a run that starts cleanly and then produces nothing
               is a different failure from one that dies loudly.

The point of all three is that **nothing fails silently**. A supervised run either finishes with
every cell filled, or it exits non-zero having written the reason to `supervisor.jsonl` and printed
a banner naming which arms were short. A supervisor that quietly stops supervising is worse than
none, because the run still looks alive.

Progress is measured from the episode logs, not by parsing child stdout: the logs are what the
experiment itself resumes from, so the supervisor and the experiment cannot disagree about what is
done, and a child whose output is buffered still shows progress.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "exp1_behavioural_divergence.py"

ARMS = ("real", "sim_scripted", "sim")


def now() -> str:
    return time.strftime("%H:%M:%S")


@dataclass
class Child:
    arm: str
    proc: subprocess.Popen | None = None
    handle: object = None
    last_progress: int = 0
    progress_at_launch: int = 0
    last_change: float = field(default_factory=time.time)
    restarts: int = 0
    idle_restarts: int = 0
    finished: bool = False
    failure: str = ""


class Supervisor:
    def __init__(self, out: Path, actor: str, per_arm: int) -> None:
        self.out = out
        self.actor = actor
        self.per_arm = per_arm
        self.state = out / "supervisor.jsonl"

    def record(self, event: str, **detail) -> None:
        self.state.parent.mkdir(parents=True, exist_ok=True)
        with self.state.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "event": event, **detail}) + "\n")
            fh.flush()
        print(f"[{now()}] [supervise] {event}: {detail}", flush=True)

    def _rows(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                      # torn final line after a kill; expected
        return out

    def progress(self, arm: str) -> tuple[int, int]:
        """(distinct usable cells, api_error episodes) for one arm.

        Distinct cells, not lines: that is what the experiment analyses after `dedupe_episodes`,
        so the supervisor never reports progress the experiment will not see. The pre-sharding log
        is included because episodes collected before the split still count as done.
        """
        rows = self._rows(self.out / f"episodes-{arm}.jsonl") + self._rows(
            self.out / "episodes.jsonl")
        cells, errors = set(), 0
        for ep in rows:
            if ep.get("model_alias") != self.actor or ep.get("arm") != arm:
                continue
            if ep.get("api_error"):
                errors += 1
            else:
                cells.add(ep.get("task_id", ""))
        return (len(cells), errors)

    def launch(self, child: Child, passthrough: list[str]) -> None:
        cmd = [sys.executable, "-u", str(EXP), "--arms", child.arm, *passthrough]
        log_path = self.out / f"run-{child.arm}.log"
        child.handle = log_path.open("a", encoding="utf-8")
        child.handle.write(
            f"\n===== supervisor launch {time.strftime('%Y-%m-%dT%H:%M:%S')} =====\n")
        child.handle.flush()
        child.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=child.handle,
                                      stderr=subprocess.STDOUT)
        child.progress_at_launch = self.progress(child.arm)[0]
        child.last_progress = child.progress_at_launch
        child.last_change = time.time()
        self.record("launch", arm=child.arm, done=child.progress_at_launch, log=log_path.name)

    @staticmethod
    def stop(child: Child) -> None:
        if child.proc is None or child.proc.poll() is not None:
            return
        child.proc.terminate()
        try:
            child.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.proc.kill()
            child.proc.wait(timeout=20)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--stall-seconds", type=float, default=900.0,
                    help="an episode at 20 turns with a reasoning actor takes several minutes, so "
                         "this must sit well above a slow episode or healthy work gets killed")
    ap.add_argument("--max-restarts", type=int, default=12)
    ap.add_argument("--max-idle-restarts", type=int, default=3)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--actor", default="openrouter:qwen/qwen3-32b")
    ap.add_argument("--out", default=str(ROOT / "results" / "exp1"))
    ap.add_argument("--cells-per-arm", type=int, default=24)
    ap.add_argument("--no-final-analysis", action="store_true")
    args, rest = ap.parse_known_args(argv)
    passthrough = [a for a in rest if a != "--"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    sup = Supervisor(out, args.actor, args.cells_per_arm)

    children = {a: Child(arm=a) for a in arms}
    sup.record("start", arms=arms, cells_per_arm=args.cells_per_arm,
               done={a: sup.progress(a)[0] for a in arms},
               stall_seconds=args.stall_seconds, passthrough=" ".join(passthrough))

    for child in children.values():
        if sup.progress(child.arm)[0] >= args.cells_per_arm:
            child.finished = True
            sup.record("arm_already_complete", arm=child.arm)
        else:
            sup.launch(child, passthrough)

    exit_code = 1
    try:
        while not all(c.finished for c in children.values()):
            time.sleep(args.poll)
            for child in children.values():
                if child.finished:
                    continue
                done, errors = sup.progress(child.arm)
                code = child.proc.poll() if child.proc else 0

                if done != child.last_progress:
                    sup.record("progress", arm=child.arm, done=done,
                               of=args.cells_per_arm, api_errors=errors)
                    child.last_progress, child.last_change = done, time.time()

                if code is None and time.time() - child.last_change > args.stall_seconds:
                    sup.record("stall", arm=child.arm, done=done,
                               idle_seconds=round(time.time() - child.last_change, 1))
                    sup.stop(child)
                    code = child.proc.returncode if child.proc else 1

                if code is None:
                    continue                                  # still working

                done, errors = sup.progress(child.arm)
                if done >= args.cells_per_arm:
                    child.finished = True
                    sup.record("arm_complete", arm=child.arm, done=done,
                               restarts=child.restarts, api_errors=errors)
                    continue

                if done == child.progress_at_launch:
                    child.idle_restarts += 1
                    sup.record("relaunch_made_no_progress", arm=child.arm,
                               consecutive=child.idle_restarts, limit=args.max_idle_restarts)
                else:
                    child.idle_restarts = 0

                if child.idle_restarts >= args.max_idle_restarts:
                    child.finished, child.failure = True, "no progress across relaunches"
                    sup.record("FATAL_no_progress", arm=child.arm, done=done)
                    continue
                if child.restarts >= args.max_restarts:
                    child.finished, child.failure = True, "restart budget exhausted"
                    sup.record("FATAL_restart_budget_exhausted", arm=child.arm, done=done)
                    continue

                child.restarts += 1
                sup.record("relaunching", arm=child.arm, restarts=child.restarts, done=done)
                time.sleep(5)
                sup.launch(child, passthrough)

        short = {a: sup.progress(a)[0] for a in arms
                 if sup.progress(a)[0] < args.cells_per_arm}
        exit_code = 0 if not short else 2
        if short:
            sup.record("incomplete", short=short, expected=args.cells_per_arm)

    except KeyboardInterrupt:
        sup.record("interrupted_by_user")
        exit_code = 130
    finally:
        for child in children.values():
            sup.stop(child)
            if child.handle:
                child.handle.close()

        # A supervisor that stops supervising without saying so is worse than none at all: the run
        # looks alive. Print the outcome on every path out, including an exception.
        totals = {a: sup.progress(a) for a in arms}
        filled = sum(c for c, _ in totals.values())
        expected = args.cells_per_arm * len(arms)
        print("\n" + "=" * 78, flush=True)
        print(f"  exp1 supervisor: {'COMPLETE' if exit_code == 0 else f'INCOMPLETE (exit {exit_code})'}",
              flush=True)
        for arm, (cells, errors) in totals.items():
            note = f"   !! {children[arm].failure}" if children[arm].failure else ""
            print(f"    {arm:<14} {cells:>3}/{args.cells_per_arm}  api_errors {errors}  "
                  f"restarts {children[arm].restarts}{note}", flush=True)
        print(f"  total {filled}/{expected}   history: {sup.state}", flush=True)
        print("=" * 78, flush=True)
        sup.record("supervisor_exit", code=exit_code, filled=filled, expected=expected)

    if exit_code == 0 and not args.no_final_analysis:
        sup.record("final_analysis")
        proc = subprocess.run(
            [sys.executable, "-u", str(EXP), "--analyse-only", "--out", str(out),
             "--actor", args.actor],
            cwd=str(ROOT), text=True)
        sup.record("final_analysis_done", code=proc.returncode)
        return proc.returncode

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
