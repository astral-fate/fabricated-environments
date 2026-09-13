"""The REAL arm: a sealed, instrumented directory tree where tool calls do what they say.

Built on `src/substrate.py`, vendored byte-identical from `project-v2/src/substrate.py` (paper 1,
*Closing Content Does Not Close the Channel*). Reused rather than rewritten because it already
does the two things this arm needs: it mediates every primitive through one audited path, and it
refuses traversal outside its root. `scripts/sync_from_project_v2.py` fails if the vendored copy
drifts from its source.

The closure ladder is pinned to the `open` rung
------------------------------------------------
Paper 1's substrate exists to *neutralise* attributes at varying strengths -- that is what its
`ClosurePolicy` is for. This paper wants the opposite: an environment where tool calls behave
ordinarily, because "ordinary" is the ground truth the SIM arm is trying to imitate. So the
policy is `from_rung("open")`, which closes nothing. Any other rung would make the REAL arm
itself unusual and would confound the contrast.

Why the root contains out-of-scope material
--------------------------------------------
`Substrate._resolve` refuses any path escaping its root, so a scope boundary drawn at the root
would be enforced by the substrate and the violation rate would be identically zero -- measuring
the substrate, not the agent. The root therefore holds `workspace/` alongside `secrets/`, `etc/`,
`home/` and `var/`, all genuinely readable and writable. The boundary exists only in the task
brief. See `scope.py` for the full argument.

Egress
------
This arm has no network affordance and cannot acquire one: `substrate.py` imports only
`hashlib`, `json`, `os`, `shutil`, `time`, `dataclasses`, `pathlib` and `typing`.
`verify_no_egress()` checks that structurally at setup; `tests/test_real_arm.py` additionally
denies `socket.socket` for the duration of a full episode and requires the episode to complete,
which exhibits the property rather than asserting it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

from srm import ClosurePolicy
from substrate import Substrate, SubstrateError

from .base import Arm

#: Directories always present in the root, so the arm's shape never varies with the task.
SKELETON = ("workspace", "workspace/components", "secrets", "etc", "home/user", "var/log")


class RealArm(Arm):
    """Tool calls execute against a real filesystem tree. Nothing is fabricated."""

    name = "real"
    grounded = True

    def __init__(self, root: str | Path, log_path: str | Path | None = None):
        super().__init__()
        self.root = Path(root)
        self.log_path = Path(log_path) if log_path else self.root.parent / f"{self.root.name}.events.jsonl"
        self.substrate: Substrate | None = None

    # ------------------------------------------------------------------ lifecycle
    def reset(self, task: Any) -> None:
        """Install the task's seeded filesystem into a freshly created root."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

        # `open` closes nothing: every attribute behaves ordinarily. See module docstring.
        self.substrate = Substrate(self.root, ClosurePolicy.from_rung("open"), self.log_path)
        self.substrate.bind(episode=getattr(task, "task_id", "unbound"), agent="agent", generation=0)
        self.substrate.seed(entries=dict(task.files), skeleton=SKELETON)

        self.actions = []
        self._index = 0

    def snapshot(self) -> dict[str, str]:
        """Every file's contents, read directly rather than through the substrate.

        Deliberately bypasses `self.substrate` so that scoring an episode does not append seven
        more events to the audit log and does not appear in the action log as agent behaviour.
        The snapshot is the experimenter's read, not the agent's.
        """
        out: dict[str, str] = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                out[rel] = p.read_text(encoding="utf-8", errors="replace")
        return out

    def close(self) -> None:
        self.substrate = None

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, name: str, args: Mapping[str, Any]) -> tuple[str, bool]:
        if self.substrate is None:
            raise RuntimeError("RealArm.reset() must be called before any tool call")
        sub = self.substrate
        try:
            if name == "list_dir":
                return json.dumps(sub.list_dir(str(args.get("path", ".")))), True
            if name == "read_file":
                return sub.read_file(str(args["path"])), True
            if name == "stat":
                return json.dumps(sub.stat(str(args["path"]))), True
            if name == "write_file":
                stored = sub.write_file(str(args["path"]), str(args.get("content", "")))
                return json.dumps({"written": stored}), True
            if name == "mkdir":
                return json.dumps({"created": sub.mkdir(str(args["path"]))}), True
            if name == "move":
                return json.dumps({"moved": sub.move(str(args["src"]), str(args["dst"]))}), True
            if name == "delete":
                sub.delete(str(args["path"]))
                return json.dumps({"deleted": str(args["path"])}), True
        except KeyError as exc:
            return f"error: missing required argument {exc}", False
        except SubstrateError as exc:
            # The substrate's refusal messages describe the cache's policy, never the experiment,
            # so they are safe to surface to the agent verbatim.
            return f"error: {exc}", False
        raise AssertionError(f"unreachable: {name}")   # base.call filters unknown tools
