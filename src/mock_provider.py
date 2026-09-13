"""A scripted provider: deterministic episodes with no network, no keys and no GPU.

Exists so the resume invariants and the episode loop can be tested at speed and in CI. Paper 1's
`runner/mock.py` serves the same purpose; this is a smaller version shaped to this harness's tool
set.

`ScriptedProvider` replays a fixed list of turns. `FlakyProvider` raises on a chosen turn, which
is how the api_error path is exercised without waiting for a real rate limit.
"""
from __future__ import annotations

from typing import Any, Sequence

from providers import Provider, Step, ToolCall


class ScriptedProvider(Provider):
    """Replays `script`: a list of (text, [(tool_name, args), ...]) pairs, one per turn.

    Running off the end of the script yields a bare submit, so a script shorter than the turn
    budget terminates cleanly rather than stalling.
    """

    family = "mock"

    def __init__(self, script: Sequence[tuple[str, Sequence[tuple[str, dict[str, Any]]]]],
                 model: str = "scripted", reasoning: Sequence[str] | None = None):
        self.script = list(script)
        self.model = model
        self.reasoning = list(reasoning or [])
        self.turn = 0

    def step(self, system, messages, tools) -> Step:
        i = self.turn
        self.turn += 1
        think = self.reasoning[i] if i < len(self.reasoning) else ""
        if i >= len(self.script):
            return Step(text="", reasoning=think,
                        tool_calls=[ToolCall(id=f"c{i}", name="submit",
                                             arguments={"note": "script exhausted"})],
                        usage={"in": 10, "out": 5})
        text, calls = self.script[i]
        return Step(text=text, reasoning=think,
                    tool_calls=[ToolCall(id=f"c{i}_{j}", name=n, arguments=dict(a))
                                for j, (n, a) in enumerate(calls)],
                    usage={"in": 10, "out": 5})

    def append_assistant(self, messages, step) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": step.text or ""}
        if step.tool_calls:
            msg["tool_calls"] = [{"function": {"name": c.name, "arguments": c.arguments}}
                                 for c in step.tool_calls]
        messages.append(msg)

    def append_tool_results(self, messages, results) -> None:
        for c, out, _ok in results:
            messages.append({"role": "tool", "name": c.name, "content": out})


class FlakyProvider(ScriptedProvider):
    """Raises on turn `fail_at`, to exercise the api_error path deterministically."""

    def __init__(self, *args, fail_at: int = 0, **kw):
        super().__init__(*args, **kw)
        self.fail_at = fail_at

    def step(self, system, messages, tools) -> Step:
        if self.turn == self.fail_at:
            self.turn += 1
            raise RuntimeError("429 rate limit exceeded")
        return super().step(system, messages, tools)


class StallingProvider(ScriptedProvider):
    """Never calls a tool. Exercises the stall path."""

    def step(self, system, messages, tools) -> Step:
        self.turn += 1
        return Step(text="Let me think about this some more.", tool_calls=[],
                    usage={"in": 10, "out": 5})
