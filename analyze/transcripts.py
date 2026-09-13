"""Extract verbatim transcript excerpts for the paper's appendix.

    python analyze/transcripts.py            # human-readable preview
    python analyze/transcripts.py --tex      # LaTeX fragment for the appendix

Tim Hua's complaint about published evaluations is the reason this exists: bar charts arrive
without the transcripts, without what the models did, and without the environments, so a reader
cannot check whether the eval measures what it claims. The artifacts here are committed, but a
reader should not have to clone a repository to see a single example of the thing being measured.

Three excerpts, chosen mechanically rather than curated:

  brief        the task as the agent received it, for one task class. Shows that the scope rule is
               stated in the prompt and enforced nowhere.
  violation    the earliest recorded scope violation, with the agent's own reasoning from the turn
               that produced it. Selected as the first violating action in episode order, not
               chosen for how striking it reads.
  substrates   the SAME tool call answered by the real substrate and by the simulator. This is the
               entire manipulation, made visible in two blocks of text.

Everything is verbatim from `results/exp0/`, truncated at a stated character budget and escaped
for LaTeX. Nothing is paraphrased, and the selection rule is stated so a reader can check that the
example was not cherry-picked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exp0"
sys.path.insert(0, str(ROOT / "src"))

MAX_CHARS = 700


def _episodes() -> list[dict[str, Any]]:
    out, seen = [], set()
    path = RESULTS / "episodes.jsonl"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ep.get("api_error") or ep.get("model_alias") != "qwen38-27b":
            continue
        key = (ep["task_id"], ep["arm"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ep)
    return out


def _replays() -> list[dict[str, Any]]:
    path = RESULTS / "replays.jsonl"
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
            continue
    return out


def first_violation() -> dict[str, Any] | None:
    """The earliest violating action across episodes, in episode order. No curation."""
    for ep in _episodes():
        rep = ep.get("scope_report") or {}
        for v in rep.get("violations", []):
            # The turn that issued this action, for its recorded reasoning and text.
            turn = None
            seen = 0
            for t in ep.get("turns", []):
                for c in (t.get("calls") or []):
                    if seen == v["index"]:
                        turn = t
                        break
                    seen += 1
                if turn:
                    break
            return {"episode": ep["episode_id"], "task_id": ep["task_id"],
                    "task_kind": ep["task_kind"], "violation": v,
                    "text": (turn or {}).get("text", ""),
                    "reasoning": (turn or {}).get("reasoning", ""),
                    "submit_note": ep.get("submit_note", "")}
    return None


def substrate_pair() -> dict[str, Any] | None:
    """The same tool call as answered by the real substrate and by the simulator.

    Picks the first `read_file` in the first replay record, because a file body shows the contrast
    more legibly than a directory listing does.
    """
    for rec in _replays():
        for i, a in enumerate(rec["actions"]):
            if a["tool"] != "read_file" or i >= len(rec["real"]) or i >= len(rec["sim"]):
                continue
            return {"episode": rec["episode_id"], "task_kind": rec["task_kind"],
                    "call": f"{a['tool']}({json.dumps(dict(a.get('args') or {}), sort_keys=True)})",
                    "real": rec["real"][i], "sim": rec["sim"][i],
                    "scripted": rec["scripted"][i] if i < len(rec.get("scripted", [])) else ""}
    return None


def task_brief(kind: str = "tempting") -> str:
    from task import build_task
    return build_task(kind, 0).brief


def _clip(s: str, n: int = MAX_CHARS) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "\n[... truncated ...]"


_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"}


def verb(s: str) -> str:
    """Escape for a verbatim-ish LaTeX block."""
    return "".join(_TEX.get(ch, ch) for ch in s)


def to_tex() -> str:
    parts: list[str] = []
    brief = _clip(task_brief(), 900)
    parts.append("\\subsection{The task as the agent received it}\n\n"
                 "Scope is stated here and enforced nowhere; the arm makes every named "
                 "out-of-scope path genuinely readable.\n\n"
                 "\\begin{quote}\\small\\ttfamily\n" + verb(brief).replace("\n", "\\\\\n")
                 + "\n\\end{quote}")

    v = first_violation()
    if v:
        head = (f"Episode {v['episode']} (task class: {v['task_kind']}). Selection rule: the "
                f"earliest violating action in episode order, not chosen for how it reads.")
        body = (f"action: {v['violation']['tool']}({v['violation']['arg']}="
                f"{v['violation']['raw_path']})\n"
                f"category: {v['violation']['category']}   accepted by the arm: "
                f"{v['violation']['ok']}   turn index: {v['violation']['index']}")
        parts.append("\\subsection{A recorded scope violation}\n\n" + verb(head) + "\n\n"
                     "\\begin{quote}\\small\\ttfamily\n" + verb(body).replace("\n", "\\\\\n")
                     + "\n\\end{quote}")
        if v["text"].strip():
            parts.append("The agent's own message on that turn:\n\n"
                         "\\begin{quote}\\small\\itshape\n" + verb(_clip(v["text"], 500))
                         + "\n\\end{quote}")
        if v["submit_note"].strip():
            parts.append("Its closing note for the episode:\n\n"
                         "\\begin{quote}\\small\\itshape\n" + verb(_clip(v["submit_note"], 300))
                         + "\n\\end{quote}")

    p = substrate_pair()
    if p:
        parts.append("\\subsection{One call, two substrates}\n\n"
                     "The entire manipulation. Identical call, identical position in the "
                     "trajectory; only the bytes returned differ.\n\n"
                     "\\begin{quote}\\small\\ttfamily\n"
                     + verb(_clip(p["call"], 200)).replace("\n", "\\\\\n") + "\n\\end{quote}")
        for label, key in (("Real substrate", "real"), ("LLM simulator", "sim"),
                           ("Template simulator", "scripted")):
            if p.get(key):
                parts.append(f"\\paragraph{{{label}.}}\n"
                             "\\begin{quote}\\small\\ttfamily\n"
                             + verb(_clip(p[key], 420)).replace("\n", "\\\\\n")
                             + "\n\\end{quote}")
    return "\n\n".join(parts)


def main(argv: list[str]) -> int:
    if "--tex" in argv:
        print(to_tex())
        return 0
    v = first_violation()
    print("=== first recorded violation ===")
    print(json.dumps(v, indent=2)[:1400] if v else "none found")
    p = substrate_pair()
    print("\n=== one call, two substrates ===")
    if p:
        print("call    :", p["call"])
        print("real    :", repr(p["real"][:200]))
        print("sim     :", repr(p["sim"][:200]))
        print("scripted:", repr(p["scripted"][:200]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
