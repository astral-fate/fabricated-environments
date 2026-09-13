"""A minimal text-completion client for the SIM arm.

Why not `providers.OpenRouterProvider`
--------------------------------------
The simulator's job is to emit the bytes a filesystem call returned. It never needs a tool call,
never needs multi-turn history, and never needs the provider registry's failover machinery. Using
the full agent provider for it imports three problems that cost an hour of silent hangs:

1. **No request timeout.** `OpenRouterProvider` builds its client without one, so the OpenAI SDK
   default applies (600 s, with its own retries on top). A single stalled request therefore hangs
   for ten minutes producing no output, which looks exactly like a wedged run. Measured: a replay
   stage sat for five minutes without completing one episode, while the same request issued
   directly returned in 4 s.
2. **`max_completion_tokens`.** Inherited from `OpenAIProvider`; endpoints vary in whether they
   accept it, and the failure mode is a bare 400 that the retry ladder then backs off over.
3. **An empty `tools` list.** Passing `tools=[]` makes some endpoints set `tool_choice: "none"`,
   and a model that emits a tool call anyway is rejected with `tool_use_failed`. Sending no tools
   field at all removes the whole category.

`providers.py` is vendored byte-identical from paper 1 and `sync_from_project_v2.py` fails the
build if it drifts, so the fix belongs here rather than there -- the rule that script states is to
give a diverging need its own module. This is that module, and it is deliberately small enough to
read in one sitting.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

#: Per-request ceiling. Short by design: a simulated tool output is a few hundred tokens, so a
#: request that has not returned in this long is stalled rather than slow, and waiting longer
#: only delays the retry that will actually succeed.
TIMEOUT_SECONDS = 45.0

#: Attempts inside this client for transport-level faults. Distinct from `LLMSimArm.ATTEMPTS`,
#: which resamples a bad *generation*; this retries a failed *request*.
MAX_RETRIES = 3


@dataclass
class SimStep:
    """Mirrors the one field of `providers.Step` that the SIM arm reads."""

    text: str = ""


class TextSimProvider:
    """One-shot text completion over an OpenAI-compatible endpoint.

    Exposes just enough of the `Provider` surface for `LLMSimArm`: a `step(system, messages,
    tools)` that ignores `tools` and returns an object with `.text`.
    """

    family = "text-sim"

    def __init__(self, model: str = "openai/gpt-oss-20b",
                 base_url: str = "https://openrouter.ai/api/v1",
                 api_key_env: str = "OPENROUTER_API_KEY",
                 max_tokens: int = 512,
                 timeout: float = TIMEOUT_SECONDS):
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set")
        # An explicit timeout, and the SDK's own retries disabled: retries are handled here so a
        # failure is visible and bounded rather than multiplying with the SDK's.
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    def step(self, system: str, messages: list, tools=None) -> SimStep:
        """Ignores `tools` entirely -- see the module docstring on `tool_use_failed`."""
        payload = [{"role": "system", "content": system}] + [
            {"role": str(m["role"]), "content": str(m["content"])} for m in messages
        ]
        last: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=payload, max_tokens=self.max_tokens,
                )
            except Exception as exc:                  # noqa: BLE001 - surfaced after the ladder
                last = exc
                time.sleep(min(2.0 * (2 ** attempt), 15.0))
                continue
            if not resp.choices:
                last = RuntimeError("no choices returned")
                continue
            return SimStep(text=resp.choices[0].message.content or "")
        raise RuntimeError(f"{MAX_RETRIES} request(s) failed; last: {type(last).__name__}: {last}")


def build_sim_provider(spec: str) -> TextSimProvider:
    """`openrouter:<model>` or `groq:<model>`; bare text is treated as an OpenRouter model id."""
    if spec.startswith("groq:"):
        return TextSimProvider(model=spec.split(":", 1)[1],
                               base_url="https://api.groq.com/openai/v1",
                               api_key_env="GROQ_API_KEY")
    if spec.startswith("openrouter:"):
        spec = spec.split(":", 1)[1]
    return TextSimProvider(model=spec)
