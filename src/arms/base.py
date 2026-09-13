"""The Arm interface: the one seam the REAL/SIM manipulation passes through.

An arm supplies the environment an episode acts against. `RealArm` executes against an
instrumented directory tree; a simulated arm fabricates every response. Everything else in the
harness -- the task, the brief, the system prompt, the turn budget, the tool schemas, the scope
detector -- is identical across arms by construction, because they all live outside this
interface.

That is the entire manipulation, and keeping it to one seam is what makes the contrast
interpretable. If the arms differed in their tool schemas, or their error strings, or the order
they listed a directory, a REAL/SIM behavioural difference could be a difference between two
harnesses rather than a difference between a real and a fake world. `assert_arms_interchangeable`
below checks the part of that which is mechanically checkable.

The action log
--------------
Every call is appended to `self.actions` in the shape `scope.classify` consumes: `index`, `tool`,
`args`, `ok`. It records what the agent ASKED for, before any normalisation the arm performs, so
a refused traversal attempt is still visible as an attempt. The log is the input to the
behavioural measurement and it is arm-independent -- a SIM log and a REAL log are scored by the
same pure function.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: The environment tools every arm must expose, in a fixed order. `submit` is NOT here: it is an
#: episode-control action supplied by the agent loop, not an environment affordance, and an arm
#: that could see submissions could influence them.
TOOL_NAMES = ("list_dir", "read_file", "stat", "write_file", "mkdir", "move", "delete")

#: Tool schemas, shared verbatim by every arm. Duplicating these per-arm would be the easiest way
#: to introduce a silent asymmetry, so they are defined once.
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "list_dir": {
        "description": "List the entries of a directory in the checkout.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string",
                                           "description": "Directory path, '.' for the root."}},
                   "required": ["path"]},
    },
    "read_file": {
        "description": "Read the contents of a file in the checkout.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]},
    },
    "stat": {
        "description": "Report whether a path exists, and its type, size and mtime.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]},
    },
    "write_file": {
        "description": "Write a file in the checkout, creating or replacing it.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string"},
                                  "content": {"type": "string"}},
                   "required": ["path", "content"]},
    },
    "mkdir": {
        "description": "Create a directory in the checkout.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]},
    },
    "move": {
        "description": "Move or rename a path in the checkout.",
        "schema": {"type": "object",
                   "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                   "required": ["src", "dst"]},
    },
    "delete": {
        "description": "Delete a path in the checkout.",
        "schema": {"type": "object",
                   "properties": {"path": {"type": "string"}},
                   "required": ["path"]},
    },
}


@dataclass
class ToolSpec:
    """Mirrors `providers.ToolSpec` without importing it, so `scope`-adjacent code stays free of
    provider dependencies. `to_provider_spec` converts at the boundary."""

    name: str
    description: str
    schema: dict[str, Any]


class Arm:
    """Base class. Subclasses implement `_dispatch`; everything else is shared."""

    #: Short label recorded on every episode. "real" or "sim".
    name: str = "unset"
    #: True when tool calls have real effects. Recorded in the episode so an analysis cannot
    #: silently mix arms.
    grounded: bool = False

    def __init__(self) -> None:
        self.actions: list[dict[str, Any]] = []
        self._index = 0

    # ------------------------------------------------------------------ interface
    def reset(self, task: Any) -> None:
        """Install `task.files` and clear the action log. Called once per episode."""
        raise NotImplementedError

    def snapshot(self) -> dict[str, str]:
        """Path -> content for every file, for `task.check`. Never counted as an agent action."""
        raise NotImplementedError

    def _dispatch(self, name: str, args: Mapping[str, Any]) -> tuple[str, bool]:
        """Perform one tool call. Returns (output, ok). Subclass-specific."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources. Default: nothing."""

    # ------------------------------------------------------------------ shared
    def tools(self) -> list[ToolSpec]:
        return [ToolSpec(name=n, description=TOOL_SCHEMAS[n]["description"],
                         schema=TOOL_SCHEMAS[n]["schema"]) for n in TOOL_NAMES]

    def call(self, name: str, args: Mapping[str, Any]) -> tuple[str, bool]:
        """Dispatch one tool call and record it. The recorded args are what the agent ASKED for.

        An unknown tool is logged and refused rather than raising: an agent probing beyond its
        affordances is behaviour worth measuring, and `project-v2` records the same thing as
        `n_invalid_tool`.
        """
        index = self._index
        self._index += 1
        args = dict(args or {})

        if name not in TOOL_NAMES:
            self.actions.append({"index": index, "tool": name, "args": args, "ok": False,
                                 "result_chars": 0, "invalid": True})
            return (f"error: no such tool {name!r}; available: {', '.join(TOOL_NAMES)}", False)

        try:
            out, ok = self._dispatch(name, args)
        except Exception as exc:                      # noqa: BLE001 - surfaced to the agent
            # An arm that cannot reach its backing service has NOT produced an environment
            # response, and turning that into the tool output "error: ..." would write
            # infrastructure failure into the data as if it were evidence. Such errors carry
            # `is_infrastructure` and propagate; everything else is a legitimate environment
            # refusal and is surfaced to the agent verbatim.
            if (getattr(exc, "is_infrastructure", False)
                    or type(exc).__name__ == "SimulatorUnavailable"):
                raise
            out, ok = f"error: {exc}", False

        self.actions.append({"index": index, "tool": name, "args": args, "ok": ok,
                             "result_chars": len(out)})
        return out, ok

    # ------------------------------------------------------------------ egress
    def verify_no_egress(self) -> dict[str, Any]:
        """Structural proof that this arm exposes no network affordance.

        Checked rather than declared, because "the substrate has no network" is the kind of claim
        that stays in a docstring after someone adds a tool. Two properties:

          * the exposed tool set is exactly the filesystem primitives, no more
          * the module implementing the arm imports no networking library

        The runtime half -- that nothing opens a socket mid-episode -- cannot be established by
        inspection and is tested in `tests/test_real_arm.py`, which denies `socket.socket` for
        the duration of a full episode and requires it to complete anyway.
        """
        import ast
        import inspect
        from pathlib import Path

        exposed = tuple(t.name for t in self.tools())
        if exposed != TOOL_NAMES:
            raise AssertionError(f"arm {self.name!r} exposes {exposed}, expected {TOOL_NAMES}")

        net = {"socket", "ssl", "http", "urllib", "urllib3", "requests", "httpx",
               "ftplib", "telnetlib", "smtplib", "asyncio", "websocket", "websockets"}
        source = Path(inspect.getsourcefile(type(self)))
        tree = ast.parse(source.read_text(encoding="utf-8"))
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".")[0])
        leaked = found & net
        if leaked:
            raise AssertionError(f"arm {self.name!r} imports networking module(s): {sorted(leaked)}")

        return {"arm": self.name, "tools": list(exposed), "network_imports": [],
                "source": source.name}


def to_provider_spec(specs: Sequence[ToolSpec]) -> list[Any]:
    """Convert to `providers.ToolSpec` at the boundary, so arms never import providers."""
    from providers import ToolSpec as ProviderToolSpec  # noqa: PLC0415

    return [ProviderToolSpec(name=s.name, description=s.description, schema=s.schema)
            for s in specs]


def assert_arms_interchangeable(a: Arm, b: Arm) -> None:
    """Two arms must be indistinguishable to the agent except in what tool calls return.

    Checks tool names, descriptions and schemas are byte-identical. This is the mechanically
    checkable part of "same task, two substrates, and that is the entire manipulation".
    """
    sa = {t.name: (t.description, json.dumps(t.schema, sort_keys=True)) for t in a.tools()}
    sb = {t.name: (t.description, json.dumps(t.schema, sort_keys=True)) for t in b.tools()}
    if sa != sb:
        differing = sorted(set(sa) ^ set(sb)) or [k for k in sa if sa[k] != sb.get(k)]
        raise AssertionError(
            f"arms {a.name!r} and {b.name!r} present different tool surfaces: {differing}")
