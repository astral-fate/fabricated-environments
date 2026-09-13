"""Supervise a long exp0 run so an exhausted key costs minutes, not hours.

    python scripts/supervise.py -- --seeds 8 --max-turns 9

Everything after `--` is passed straight to `experiments/exp0_construct_separation.py`.

Why this is not `scripts/watchdog.py`
-------------------------------------
`watchdog.py` is vendored byte-identical from `project-v2` and is hardcoded to that project's
`run_local.py`. Editing it would break `sync_from_project_v2.py`, which is the check that keeps the
reuse boundary honest. The rule written into that script is: if a vendored file needs to differ
here, move it out of VENDORED and give it its own module with a docstring saying what diverged.
This is that module. The supervision *strategy* is the same one paper 1 arrived at; only the
failure signatures are specific to exp0.

What it watches, and why each one
----------------------------------
`providers._retry` already handles a single call well -- it honours Retry-After, backs off, and
caps the wait -- and `GroqProvider` already rotates to the next key when waiting cannot help.
Neither can see the shape of the whole run, which is where the time goes.

  stall        no new episode line for `--stall-seconds`. The run is wedged: a retry ladder can
               burn eight attempts over roughly ten minutes per call, and a wedged episode can
               eat an hour producing nothing. Kill and relaunch; resume is append-only and
               tested (`tests/test_resume.py`), so a restart costs at most the episode in flight.

  budget spent THE failure this run is actually exposed to, and it is NOT what it first looks
               like. Groq's free tier enforces two ceilings: ~8,000 tokens per MINUTE, which
               refills in milliseconds, and 200,000 tokens per DAY -- the second scoped to the
               ORGANISATION, not the key. Measured on `qwen/qwen3.8-27b`, 13 Sep 2026:

                   "on tokens per day (TPD): Limit 200000, Used 198880 ...
                    in organization org_01jz7..."

               Two consequences. Key rotation is useless here: four keys on one organisation
               share one daily pool, so rotating burns all four in a second and then gives up,
               converting "wait" into `api_error`. And a cooldown cannot fix it either -- the
               daily window refills at roughly 4,000 tokens per 21 minutes, so at ~12,400 tokens
               per episode a single further episode costs about an hour of waiting.

               The supervisor therefore reports the distinction rather than grinding: a
               per-minute stall is worth cooling down for; a spent daily budget needs a key on a
               DIFFERENT organisation, a different model (each has its own daily pool), or
               tomorrow.

  thin set     exp0 itself refuses to analyse fewer than `--min-episodes` usable episodes, but
               that check fires at the end. The supervisor reports the running count so a spent
               budget is visible while there is still time to add a key.

Every decision is appended to `results/exp0/supervisor.jsonl` as it happens, so the supervisor's
own history survives its own death.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "exp0_construct_separation.py"
OUT = ROOT / "results" / "exp0"
LOG = OUT / "real-run.log"
STATE = OUT / "supervisor.jsonl"

EPISODE_RE = re.compile(r"\[(\d+)/(\d+)\]\s+(\S+)")
ROTATE_RE = re.compile(r"rotating to key (\d+)/(\d+)")
DONE_RE = re.compile(r"^exit=(\d+)", re.MULTILINE)


def record(event: str, **detail) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with STATE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "event": event, **detail}) + "\n")
        fh.flush()
    print(f"[supervise] {event}: {detail}", flush=True)


def usable_and_errored(actor: str) -> tuple[int, int]:
    """(distinct usable cells, api_error episodes) for `actor`, from the append-only log.

    Counts DISTINCT cells, not clean lines, because that is what `exp0` will actually analyse
    after `dedupe_episodes`. Reporting raw lines here would let the supervisor say "18 usable"
    while exp0 sees 16 -- and a `--min-episodes` threshold between the two would fail at the end
    of a run the supervisor had reported as healthy throughout.
    """
    path = OUT / "episodes.jsonl"
    if not path.exists():
        return (0, 0)
    cells: set[tuple[str, str]] = set()
    err = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue                       # torn final line after a kill; expected
        if ep.get("model_alias") != actor:
            continue
        if ep.get("api_error"):
            err += 1
        else:
            cells.add((ep.get("task_id", ""), ep.get("arm", "")))
    return (len(cells), err)


def log_lines() -> int:
    """Current episode-log length, used as a watermark so history is not re-judged."""
    path = OUT / "episodes.jsonl"
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def trailing_api_errors(actor: str, since: int = 0) -> int:
    """Consecutive api_error episodes at the end of the log, counting only lines after `since`.

    The watermark is load-bearing, not defensive. Without it this counts the trailing run over the
    WHOLE history -- and since a rate-limited run ends in api_errors, the counter is already above
    the threshold before the next launch does anything. Observed: the supervisor declared the
    budget spent 20 seconds after relaunching, killed a child that had not yet attempted an
    episode, cooled down, and repeated. The signal must be about what THIS attempt produced.
    """
    path = OUT / "episodes.jsonl"
    if not path.exists():
        return 0
    run = 0
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if i < since:
            continue
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ep.get("model_alias") != actor:
            continue
        run = run + 1 if ep.get("api_error") else 0
    return run


def key_position(text: str) -> tuple[int, int] | None:
    hits = ROTATE_RE.findall(text)
    return (int(hits[-1][0]), int(hits[-1][1])) if hits else None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stall-seconds", type=float, default=420.0)
    ap.add_argument("--max-restarts", type=int, default=6)
    ap.add_argument("--max-api-errors", type=int, default=3,
                    help="consecutive api_error episodes before treating the budget as spent")
    ap.add_argument("--cooldown-seconds", type=float, default=900.0,
                    help="wait this long after exhaustion, then resume. Useful only for the "
                         "PER-MINUTE ceiling. See the module docstring: the limit that actually "
                         "stopped this project is per-DAY and per-ORGANISATION, and no cooldown "
                         "of a practical length clears it.")
    ap.add_argument("--max-cooldowns", type=int, default=4)
    ap.add_argument("--actor", default="qwen38-27b")
    ap.add_argument("--poll", type=float, default=20.0)
    args, passthrough = ap.parse_known_args(argv)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    restarts = 0
    cooldowns = 0
    while True:
        cmd = [sys.executable, "-u", str(EXP), *passthrough]
        # Watermark BEFORE launching: everything already in the log is history, and judging this
        # attempt by it is what made the supervisor kill a child 20 seconds after starting it.
        watermark = log_lines()
        record("launch", cmd=" ".join(cmd[2:]), restart=restarts, log_watermark=watermark)
        killed = False
        cooled = False
        with LOG.open("a", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)

            last_size = LOG.stat().st_size if LOG.exists() else 0
            last_change = time.time()
            last_key: tuple[int, int] | None = None

            while proc.poll() is None:
                time.sleep(args.poll)
                size = LOG.stat().st_size if LOG.exists() else 0
                if size != last_size:
                    last_size, last_change = size, time.time()

                text = LOG.read_text(encoding="utf-8", errors="replace")[-20000:]
                pos = key_position(text)
                if pos and pos != last_key:
                    last_key = pos
                    record("key_rotation", key=pos[0], of=pos[1],
                           note="last key in use" if pos[0] >= pos[1] else "")

                trailing = trailing_api_errors(args.actor, since=watermark)
                if trailing >= args.max_api_errors:
                    killed = True
                    cooled = True
                    clean, err = usable_and_errored(args.actor)
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    cooldowns += 1
                    record("rate_limited", trailing_api_errors=trailing, clean_episodes=clean,
                           api_error_episodes=err, cooldown=cooldowns)
                    if cooldowns > args.max_cooldowns:
                        print("\n" + "=" * 78)
                        print(f"STOPPED after {cooldowns - 1} cooldown(s): the limit is not "
                              "recovering.")
                        print(f"  usable episodes: {clean}")
                        print("\nAdd capacity, then re-run the same command:")
                        print("  GROQ_API_KEY_5=... (numbered form), or GROQ_API_KEYS=k1,k2,k3")
                        print("Stage 1 is append-only; completed episodes are kept.")
                        print("=" * 78)
                        return 4
                    print(f"\n[supervise] every key rate-limited after {clean} clean episode(s). "
                          f"Cooling down {args.cooldown_seconds:.0f}s "
                          f"({cooldowns}/{args.max_cooldowns}), then resuming.", flush=True)
                    time.sleep(args.cooldown_seconds)
                    break

                if time.time() - last_change > args.stall_seconds:
                    killed = True
                    record("stall", seconds=round(time.time() - last_change, 1))
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break

        # Only a NATURAL exit ends supervision. `killed` distinguishes "the child finished" from
        # "we terminated it and are about to relaunch" -- without it, a cooldown fell through to
        # this branch, returned the terminated child's exit code, and the supervisor stopped after
        # a single cooldown instead of resuming.
        if not killed and proc.poll() is not None:
            code = proc.returncode or 0
            clean, err = usable_and_errored(args.actor)
            record("exit", code=code, clean_episodes=clean, api_error_episodes=err)
            return int(code)

        # A cooldown relaunch is governed by --max-cooldowns, not --max-restarts. Counting it
        # against both would let the restart limit (6) trip before the cooldown budget (12) and
        # abandon a run that was waiting out a rate limit exactly as designed.
        if not cooled:
            restarts += 1
        if restarts > args.max_restarts:
            record("giving_up", restarts=restarts)
            print(f"\nSTOPPED: {restarts} restarts without completing. Something is wrong that "
                  "restarting does not fix; read results/exp0/real-run.log.")
            return 5
        record("restarting", restart=restarts)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
