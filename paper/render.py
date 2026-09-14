"""Recompute every number the paper asserts, and render the manuscript from `results/`.

    python paper/render.py            # recompute, check, render paper/main.tex
    python paper/render.py --list     # print the claim table and stop
    python paper/render.py --check    # recompute only; render nothing (used by scripts/check.py)

Why the manuscript contains no numbers of its own
-------------------------------------------------
Inherited from paper 1, where version 1 shipped a cost table that contradicted its own artifact --
a false caption, an unsourced standard deviation, four wrong cells -- and survived
review-by-consistency-checker because the checker recomputed what the paper *said* and confirmed
the paper said it consistently.

So the values are not written into the manuscript at all. `main.tex.tmpl` carries `{{PLACEHOLDER}}`
tokens and this script substitutes what `results/` actually contains. A figure in the paper cannot
disagree with its artifact because the paper has no figures of its own. Unknown placeholders fail
the build; computed-but-unused claims are reported, since an orphan usually means a finding was
cut and its number left behind.

Pre-registered constants are substituted too
--------------------------------------------
The 0.85 cross-AUROC threshold and the 0.70 reproduction floor are decisions, not measurements --
but they are still substituted rather than typed, because a threshold quoted in prose and a
threshold enforced in code drifting apart is exactly the failure this machinery exists to prevent.
They are read from the experiment module, which is the thing that actually enforces them.

This establishes agreement between manuscript and artifacts. It does NOT establish that a measure
is appropriate -- the limit that let paper 1's headline through -- and the paper says so.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
RESULTS = ROOT / "results" / "exp0"
sys.path.insert(0, str(ROOT / "experiments"))

#: Scales in reporting order: (tag, result filename, human label).
#: A missing file is not an error -- it renders as a visible [PENDING], the same way paper 1
#: handles an artifact DOI that does not exist yet. A scale point that has not run must look
#: unfinished in the PDF rather than silently absent from the table.
SCALES = [
    ("17b", "exp0.json", "Qwen3-1.7B"),
    ("8b", "exp0-qwen3-8b.json", "Qwen3-8B"),
    ("32b", "exp0-qwen3-32b.json", "Qwen3-32B"),
]

#: The scale the headline numbers are quoted at. 8B is the smallest point inside Heidari et al.'s
#: stated 7B-49B range, so it is the one that answers the scope objection the 1.7B run carries.
HEADLINE = "8b"

PENDING = r"\textbf{[PENDING]}"

TEX_ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}"}


def tex_escape(s: str) -> str:
    return "".join(TEX_ESCAPES.get(ch, ch) for ch in s)


@dataclass
class Claim:
    key: str
    describe: str
    fn: Callable[[], str]
    raw: bool = False          #: True when the value is LaTeX and must not be escaped
    #: True when absence is a legitimate state of the world rather than a broken build. An
    #: unreleased artifact has no URL yet; that should print as a visible [PENDING] in the PDF,
    #: not fail the build and not quietly render a plausible-looking placeholder a reader would
    #: mistake for a real link.
    pending_ok: bool = False


class Uncomputable(Exception):
    """Raised when a claim's inputs are absent. Renders as [PENDING], never as a guess."""


# --------------------------------------------------------------------------- loading

_cache: dict[str, dict[str, Any] | None] = {}


def load(tag: str) -> dict[str, Any]:
    if tag not in _cache:
        path = RESULTS / dict((t, f) for t, f, _ in SCALES)[tag]
        _cache[tag] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if _cache[tag] is None:
        raise Uncomputable(f"no result for scale {tag}")
    return _cache[tag]


def available() -> list[tuple[str, str, str]]:
    return [(t, f, label) for t, f, label in SCALES if (RESULTS / f).exists()]


def _last(tag: str) -> dict[str, Any]:
    return load(tag)["pooling"]["last"]


def _dir(tag: str, name: str) -> dict[str, Any]:
    return _last(tag)["directions"][name]


def _cross(tag: str, a: str, b: str) -> dict[str, Any]:
    return _last(tag)["cross_auroc"][f"{a}->{b}"]


def _cos(tag: str, a: str, b: str) -> dict[str, Any]:
    key = f"{a}|{b}" if f"{a}|{b}" in _last(tag)["cosine"] else f"{b}|{a}"
    return _last(tag)["cosine"][key]


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci(d: dict[str, Any]) -> str:
    lo, hi = d["lo"], d["hi"]
    return f"[{lo:.3f}, {hi:.3f}]"


# --------------------------------------------------------------------------- claim bodies

def _exp0_module():
    """exp0's runner, loaded by path.

    The constants below are read from the code that enforces them rather than retyped here: a
    threshold quoted in the manuscript that the experiment does not actually use is exactly the
    drift this substitution machinery exists to prevent.
    """
    import importlib.util

    key = "__exp0_module"
    if key not in _cache:
        path = ROOT / "experiments" / "exp0_construct_separation" / "run.py"
        spec = importlib.util.spec_from_file_location("exp0_run", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _cache[key] = module
    return _cache[key]


def _threshold() -> str:
    exp0 = _exp0_module()
    return f"{exp0.CROSS_AUROC_THRESHOLD:.2f}"


def _floor() -> str:
    exp0 = _exp0_module()
    return f"{exp0.REPRODUCTION_FLOOR:.2f}"


def _shuffle_tol() -> str:
    exp0 = _exp0_module()
    return f"{exp0.SHUFFLE_TOLERANCE:.2f}"


def _episodes() -> list[dict[str, Any]]:
    path = RESULTS / "episodes.jsonl"
    if not path.exists():
        raise Uncomputable("no episode log")
    seen, out = set(), []
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
            continue                     # the dedupe rule, applied here too
        seen.add(key)
        out.append(ep)
    return out


def _by_kind() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for ep in _episodes():
        out.setdefault(ep["task_kind"], []).append(ep)
    return out


def _violation_rate(kind: str) -> str:
    eps = _by_kind().get(kind, [])
    if not eps:
        raise Uncomputable(f"no episodes of kind {kind}")
    k = sum(1 for e in eps if e["scope_report"]["any_violation"])
    return f"{k}/{len(eps)}"


VIOL_CATEGORIES = ("write_outside", "read_outside", "stat_outside", "list_outside")


def _viol_total(category: str) -> int:
    return sum(int((ep.get("scope_report") or {}).get("counts", {}).get(category, 0))
               for ep in _episodes())


def _reasoning_turns() -> int:
    """Turns carrying a separate reasoning trace.

    Zero here is a finding, not a bug: the hosted actor returns no `reasoning` field, so the
    chain-of-thought credence channel has no raw material in this corpus. The paper states that
    rather than leaving a reader to assume the channel was measured and found null.
    """
    return sum(1 for ep in _episodes() for t in ep.get("turns", [])
               if (t.get("reasoning") or "").strip())


def _viol_table_tex() -> str:
    from collections import Counter, defaultdict

    eps = _episodes()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    any_by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ep in eps:
        rep = ep.get("scope_report") or {}
        k = ep["task_kind"]
        any_by_kind[k][1] += 1
        if rep.get("any_violation"):
            any_by_kind[k][0] += 1
        for cat, n in (rep.get("counts") or {}).items():
            by_kind[k][cat] += int(n)

    rows = []
    for k in ("solvable", "blocked", "tempting"):
        if k not in any_by_kind:
            continue
        hit, tot = any_by_kind[k]
        cells = " & ".join(str(by_kind[k].get(c, 0)) for c in VIOL_CATEGORIES)
        rows.append(rf"\texttt{{{k}}} & {hit}/{tot} & {cells} \\")
    return "\n".join(rows)


def _scale_table_tex() -> str:
    """The cross-scale results table. Missing scales render as [PENDING] rows."""
    rows = []
    for tag, fname, label in SCALES:
        if not (RESULTS / fname).exists():
            rows.append(rf"\texttt{{{tex_escape(label)}}} & \multicolumn{{5}}{{c}}{{{PENDING}}} \\")
            continue
        d = load(tag)
        a, b = _dir(tag, "A_eval"), _dir(tag, "B_empirical")
        ab, ba = _cross(tag, "A_eval", "B_empirical"), _cross(tag, "B_empirical", "A_eval")
        cos = _cos(tag, "A_eval", "B_empirical")["cosine"]
        rows.append(
            rf"\texttt{{{tex_escape(label)}}} & {a['best_auroc']:.3f} & {b['best_auroc']:.3f} & "
            rf"{ab['auroc']:.3f} & {ba['auroc']:.3f} & {cos:+.3f} \\")
    return "\n".join(rows)


def _gate_table_tex() -> str:
    """Validity gates at the headline scale."""
    d = load(HEADLINE)
    checks = d["verdict"]["checks"]
    nc = checks["negative_control"]
    rows = [
        rf"shuffled-label control & {nc['worst'][1]:.3f} & within {nc['tolerance']:.2f} of 0.5 & "
        rf"{'pass' if nc['pass'] else 'FAIL'} \\",
        rf"\texttt{{A\_eval}} reproduction floor & "
        rf"{checks['reproduction_floor']['A_eval_best_auroc']:.3f} & "
        rf"$\geq$ {checks['reproduction_floor']['floor']:.2f} & "
        rf"{'pass' if checks['reproduction_floor']['pass'] else 'FAIL'} \\",
        rf"split disjointness & --- & no episode in both & "
        rf"{'pass' if checks['no_leakage']['pass'] else 'FAIL'} \\",
    ]
    return "\n".join(rows)


def _error_analysis(tag: str) -> dict[str, Any]:
    """Per-item error analysis for one scale.

    The headline scale carries it inside its result file (computed on the GPU that held the
    activations); the development scale has a standalone file written locally. Both are the same
    structure, so the accessor hides which path supplied it.
    """
    if tag == "17b":
        p = RESULTS / "error_analysis.json"
        if not p.exists():
            raise Uncomputable("no local error analysis")
        return json.loads(p.read_text(encoding="utf-8"))["directions"]
    ea = load(tag).get("error_analysis")
    if not ea:
        raise Uncomputable(f"no error analysis for scale {tag}")
    return ea["directions"]


def _ea_min_paraphrase(tag: str, name: str) -> str:
    rec = _error_analysis(tag).get(name) or {}
    by = rec.get("by_paraphrase") or {}
    if not by:
        raise Uncomputable(f"{name} has no paraphrase axis")
    return f3(min(v["accuracy"] for v in by.values()))


def _ea_class(tag: str, name: str, kind: str) -> str:
    rec = _error_analysis(tag).get(name) or {}
    cell = (rec.get("by_task_class") or {}).get(kind)
    if not cell:
        raise Uncomputable(f"{name} has no {kind} cell")
    return f3(cell["accuracy"])


def _ea_field(tag: str, name: str, field: str) -> str:
    rec = _error_analysis(tag).get(name) or {}
    v = rec.get(field)
    if v is None:
        raise Uncomputable(f"{name}.{field} absent")
    return f3(float(v))


def _transcripts_tex() -> str:
    """Verbatim excerpts, generated from the artifacts rather than pasted.

    A reader should be able to see one example of the thing being measured without cloning the
    repository -- which is the complaint levelled at published evaluations that ship bar charts
    and no transcripts. Selection rules are mechanical and stated in the appendix text.
    """
    sys.path.insert(0, str(ROOT / "analyze"))
    try:
        import transcripts
        body = transcripts.to_tex()
    except Exception as exc:                       # noqa: BLE001
        raise Uncomputable(f"transcripts unavailable: {type(exc).__name__}") from exc
    if not body.strip():
        raise Uncomputable("no transcript excerpts found")
    return body


def _artifact(field: str) -> str:
    path = PAPER / "artifact.json"
    if not path.exists():
        raise Uncomputable("no artifact.json")
    val = json.loads(path.read_text(encoding="utf-8")).get(field)
    if not val:
        raise Uncomputable(f"artifact.{field} is null")
    return str(val)


def _sim_refusal_rate() -> str:
    path = RESULTS / "replays.jsonl"
    if not path.exists():
        raise Uncomputable("no replay log")
    n = bad = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        for out in rec.get("sim", []):
            n += 1
            if str(out).startswith("error:"):
                bad += 1
    if not n:
        raise Uncomputable("no simulated outputs")
    return f"{bad}/{n}"




# --------------------------------------------------------------------------- exp1 / exp2 / exp3
#
# The follow-on experiments write their own verdict files. Each loader raises `Uncomputable` when
# its artifact is absent, so a manuscript built before a run has finished shows a visible
# [PENDING] rather than a stale number or a plausible-looking guess.

def _verdict(name: str) -> dict[str, Any]:
    """The verdict artifact for a follow-on experiment.

    Accepts both spellings a run can leave behind: `exp2.json` when the experiment was run
    locally, and `exp2-<model>.json` when a Modal wrapper brought the result home named for the
    scale it ran at. A stage-1 validation file is never eligible -- it carries no verdict, and
    silently reading one would report a GPU-path check as a result.
    """
    key = f"__{name}"
    if key not in _cache:
        base = ROOT / "results" / name
        plain = base / f"{name}.json"
        tagged = sorted(q for q in base.glob(f"{name}-*.json") if "stage1" not in q.name)
        path = plain if plain.exists() else (tagged[-1] if tagged else None)
        _cache[key] = (json.loads(path.read_text(encoding="utf-8"))
                       if path is not None and path.exists() else None)
    if _cache[key] is None:
        raise Uncomputable(f"no {name} verdict yet")
    return _cache[key]


def _exp1_episodes() -> list[dict[str, Any]]:
    """exp1 episodes from every per-arm shard, deduplicated by (task, arm)."""
    key = "__exp1_eps"
    if key not in _cache:
        base = ROOT / "results" / "exp1"
        rows: list[dict[str, Any]] = []
        for shard in sorted(base.glob("episodes*.jsonl")):
            for line in shard.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        seen, keep = set(), []
        for r in rows:
            cell = (r.get("task_id"), r.get("arm"))
            if r.get("api_error") or cell in seen:
                continue
            seen.add(cell)
            keep.append(r)
        _cache[key] = {"rows": keep} if keep else None
    if _cache[key] is None:
        raise Uncomputable("no exp1 episodes yet")
    return _cache[key]["rows"]


def _exp1_cat(arm: str, category: str) -> str:
    """Raw count of one out-of-scope category in one arm, summed over episodes.

    A count rather than a mean, because the composition claim is that some categories are
    *absent* from the real arm, and a mean would round a genuine zero into a small decimal.
    """
    total = sum(int(r["scope_report"]["counts"].get(category, 0))
                for r in _exp1_episodes() if r.get("arm") == arm)
    return str(total)


def _exp1_contrast(contrast: str, measure: str, field: str) -> float:
    return float(_verdict("exp1")["results"][contrast][measure][field])


def _ci(lo: float, hi: float) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


def _exp1_num(contrast: str, measure: str) -> str:
    return f"{_exp1_contrast(contrast, measure, 'diff'):+.3f}"


def _exp1_ci(contrast: str, measure: str) -> str:
    return _ci(_exp1_contrast(contrast, measure, "lo"), _exp1_contrast(contrast, measure, "hi"))


def _exp1_arm_mean(arm: str, measure: str) -> str:
    return f"{float(_verdict('exp1')['arm_means'][arm][measure]):.3f}"


def _exp1_category_total(arm: str, category: str) -> str:
    """Raw out-of-scope actions of one category in one arm, summed over episodes.

    Computed from `per_cell` rather than from an arm mean, because the composition claim in the
    results ("no reads or writes at all in the real arm") is about counts and a mean would round
    a genuine zero into a small decimal.
    """
    total = 0
    for cell in _verdict("exp1")["per_cell"]:
        total += int(cell.get(f"{arm}_{category}", 0))
    return str(total)


def _exp2_num(key: str, field: str) -> str:
    v = _verdict("exp2")["results"][key][field]
    return f"{float(v):+.4f}"


def _exp2_ci(key: str) -> str:
    r = _verdict("exp2")["results"][key]
    return f"[{float(r['lo']):+.4f}, {float(r['hi']):+.4f}]"


def _exp3_num(key: str) -> str:
    return f"{float(_verdict('exp3')['results'][key]['diff']):+.3f}"


def _exp3_ci(key: str) -> str:
    r = _verdict("exp3")["results"][key]
    return _ci(float(r["lo"]), float(r["hi"]))


def _exp3_rate(key: str, field: str) -> str:
    return f"{float(_verdict('exp3')['results'][key][field]):.3f}"


def _exp1_table_tex() -> str:
    """Arm means and the three preregistered contrasts, on the primary and co-primary measures."""
    v = _verdict("exp1")
    rows = []
    for arm, label in (("real", "REAL"), ("sim_scripted", "SCRIPTED"), ("sim", "SIM")):
        m = v["arm_means"][arm]
        rows.append(rf"\texttt{{{label}}} & {m['oos_count']:.3f} & {m['oos_rate']:.3f} & "
                    rf"{m['oos_writes']:.3f} & {m['suspicion']:.3f} \\")
    return "\n".join(rows)


def _exp1_contrast_table_tex() -> str:
    v = _verdict("exp1")
    names = {"real->sim": "REAL $\\rightarrow$ SIM",
             "real->sim_scripted": "REAL $\\rightarrow$ SCRIPTED",
             "sim_scripted->sim": "SCRIPTED $\\rightarrow$ SIM"}
    rows = []
    for key, label in names.items():
        b = v["results"][key]
        star = lambda m: "*" if b[m]["excludes_zero"] else ""          # noqa: E731
        rows.append(
            rf"{label} & {b['oos_count']['diff']:+.3f} "
            rf"{_ci(b['oos_count']['lo'], b['oos_count']['hi'])}{star('oos_count')} & "
            rf"{b['oos_rate']['diff']:+.3f} "
            rf"{_ci(b['oos_rate']['lo'], b['oos_rate']['hi'])}{star('oos_rate')} \\")
    return "\n".join(rows)


CLAIMS: list[Claim] = [
    # ---- design constants, read from the code that enforces them
    Claim("THRESHOLD", "pre-registered cross-AUROC threshold", _threshold),
    Claim("FLOOR", "reproduction floor for A_eval", _floor),
    Claim("SHUFFLE_TOL", "tolerance on the shuffled-label control", _shuffle_tol),

    # ---- corpus
    Claim("N_EPISODES", "usable episodes after dedupe", lambda: str(len(_episodes()))),
    Claim("N_KINDS", "task difficulty classes", lambda: str(len(_by_kind()))),
    Claim("N_GROUPS", "episodes (clustering unit) in the analysis",
          lambda: str(_dir(HEADLINE, "A_eval")["n_groups"])),
    Claim("N_TRAIN_GROUPS", "episodes in the training split",
          lambda: str(_dir(HEADLINE, "A_eval")["n_train_groups"])),
    Claim("N_TEST_GROUPS", "episodes in the held-out split",
          lambda: str(_dir(HEADLINE, "A_eval")["n_test_groups"])),
    Claim("N_PAIRS_A", "contrastive pairs in A_eval",
          lambda: str(load(HEADLINE)["stimuli"]["A_eval"]["n_pairs"])),
    Claim("N_PAIRS_B", "contrastive pairs in B_empirical",
          lambda: str(load(HEADLINE)["stimuli"]["B_empirical"]["n_pairs"])),
    Claim("SIM_REFUSALS", "simulated refusals among fabricated tool outputs", _sim_refusal_rate),

    # ---- headline scale
    Claim("HEADLINE_MODEL", "model the headline numbers are quoted at",
          lambda: load(HEADLINE)["config"]["model"]),
    Claim("HEADLINE_DEVICE", "GPU the headline run used",
          lambda: load(HEADLINE)["hardware"]["device"]),
    Claim("HEADLINE_DTYPE", "weight dtype of the headline run",
          lambda: load(HEADLINE)["hardware"]["dtype"]),
    Claim("AEVAL_AUROC", "A_eval held-out AUROC",
          lambda: f3(_dir(HEADLINE, "A_eval")["best_auroc"])),
    Claim("AEVAL_CI", "A_eval held-out AUROC interval",
          lambda: ci(_dir(HEADLINE, "A_eval")["best_auroc_ci"])),
    Claim("AEVAL_LAYER", "layer at which A_eval is strongest",
          lambda: str(_dir(HEADLINE, "A_eval")["best_layer"])),
    Claim("BEMP_AUROC", "B_empirical held-out AUROC",
          lambda: f3(_dir(HEADLINE, "B_empirical")["best_auroc"])),
    Claim("BEMP_CI", "B_empirical held-out AUROC interval",
          lambda: ci(_dir(HEADLINE, "B_empirical")["best_auroc_ci"])),
    Claim("CDEC_AUROC", "C_declared held-out AUROC",
          lambda: f3(_dir(HEADLINE, "C_declared")["best_auroc"])),
    Claim("SSCR_AUROC", "S_scripted held-out AUROC",
          lambda: f3(_dir(HEADLINE, "S_scripted")["best_auroc"])),

    # ---- the gate
    Claim("CROSS_AB", "cross-AUROC, eval-awareness direction onto the realness contrast",
          lambda: f3(_cross(HEADLINE, "A_eval", "B_empirical")["auroc"])),
    Claim("CROSS_AB_CI", "interval on cross-AUROC A onto B",
          lambda: ci(_cross(HEADLINE, "A_eval", "B_empirical")["ci"])),
    Claim("CROSS_BA", "cross-AUROC, realness direction onto the eval-awareness contrast",
          lambda: f3(_cross(HEADLINE, "B_empirical", "A_eval")["auroc"])),
    Claim("CROSS_BA_CI", "interval on cross-AUROC B onto A",
          lambda: ci(_cross(HEADLINE, "B_empirical", "A_eval")["ci"])),
    Claim("COS_AB", "cosine between the eval-awareness and realness directions",
          lambda: f"{_cos(HEADLINE, 'A_eval', 'B_empirical')['cosine']:+.3f}"),
    Claim("REL_A", "split-half reliability of A_eval",
          lambda: f3(_dir(HEADLINE, "A_eval")["reliability"]["mean"])),
    Claim("REL_B", "split-half reliability of B_empirical",
          lambda: f3(_dir(HEADLINE, "B_empirical")["reliability"]["mean"])),

    # ---- the secondary structure that makes the null interpretable
    Claim("CROSS_AC", "cross-AUROC, eval-awareness onto declared-realness",
          lambda: f3(_cross(HEADLINE, "A_eval", "C_declared")["auroc"])),
    Claim("COS_AC", "cosine between eval-awareness and declared realness",
          lambda: f"{_cos(HEADLINE, 'A_eval', 'C_declared')['cosine']:+.3f}"),
    Claim("COS_BC", "cosine between empirical and declared realness",
          lambda: f"{_cos(HEADLINE, 'B_empirical', 'C_declared')['cosine']:+.3f}"),
    Claim("CROSS_BS", "cross-AUROC, empirical realness onto the scripted simulator",
          lambda: f3(_cross(HEADLINE, "B_empirical", "S_scripted")["auroc"])),
    Claim("COS_BS", "cosine between empirical realness and the scripted control",
          lambda: f"{_cos(HEADLINE, 'B_empirical', 'S_scripted')['cosine']:+.3f}"),
    Claim("SHUFFLE_A", "shuffled-label control for A_eval",
          lambda: f3(_dir(HEADLINE, "A_eval")["shuffled_control"]["mean"])),

    # ---- behaviour (reported, not collapsed to a scalar)
    Claim("VIOL_SOLVABLE", "episodes with a scope violation, solvable tasks",
          lambda: _violation_rate("solvable")),
    Claim("VIOL_BLOCKED", "episodes with a scope violation, blocked tasks",
          lambda: _violation_rate("blocked")),
    Claim("VIOL_TEMPTING", "episodes with a scope violation, tempting tasks",
          lambda: _violation_rate("tempting")),

    # ---- behaviour, reported by CATEGORY rather than as a bare rate.
    #
    # An "any violation" rate collapses reconnaissance and exfiltration into one number, and in
    # this corpus they are wildly unequal: the crossings are overwhelmingly directory listings and
    # metadata reads. Quoting only the rate would imply agents took the out-of-scope material,
    # which they did not. The brief's rule -- report violation type, never collapse to a scalar
    # without showing components -- exists for exactly this case.
    Claim("VIOL_TABLE", "scope violations by task class and category", _viol_table_tex, raw=True),
    Claim("N_READ_OUTSIDE", "out-of-scope file reads across the whole corpus",
          lambda: str(_viol_total("read_outside"))),
    Claim("N_WRITE_OUTSIDE", "out-of-scope writes across the whole corpus",
          lambda: str(_viol_total("write_outside"))),
    Claim("N_RECON", "out-of-scope listings and metadata reads across the whole corpus",
          lambda: str(_viol_total("list_outside") + _viol_total("stat_outside"))),

    # ---- the largest scale, for the trend the Discussion reads off Table 2.
    # Quoted separately from the headline because a trend is a claim about the ENDPOINTS, and
    # naming them through claims keeps the prose from drifting if a scale point is re-run.
    Claim("LARGE_MODEL", "largest model evaluated", lambda: load("32b")["config"]["model"]),
    Claim("SMALL_MODEL", "smallest model evaluated", lambda: load("17b")["config"]["model"]),
    Claim("LARGE_DEVICE", "GPU the largest run used",
          lambda: load("32b")["hardware"]["device"]),
    Claim("AEVAL_AUROC_LARGE", "A_eval held-out AUROC at the largest scale",
          lambda: f3(_dir("32b", "A_eval")["best_auroc"])),
    Claim("BEMP_AUROC_LARGE", "B_empirical held-out AUROC at the largest scale",
          lambda: f3(_dir("32b", "B_empirical")["best_auroc"])),
    Claim("CROSS_AB_LARGE", "cross-AUROC A onto B at the largest scale",
          lambda: f3(_cross("32b", "A_eval", "B_empirical")["auroc"])),
    Claim("CROSS_BA_LARGE", "cross-AUROC B onto A at the largest scale",
          lambda: f3(_cross("32b", "B_empirical", "A_eval")["auroc"])),
    Claim("COS_AB_LARGE", "cosine between eval-awareness and realness, largest scale",
          lambda: f"{_cos('32b', 'A_eval', 'B_empirical')['cosine']:+.3f}"),
    Claim("COS_AC_SMALL", "cosine, eval-awareness and declared realness, smallest scale",
          lambda: f"{_cos('17b', 'A_eval', 'C_declared')['cosine']:+.3f}"),
    Claim("COS_AC_LARGE", "cosine, eval-awareness and declared realness, largest scale",
          lambda: f"{_cos('32b', 'A_eval', 'C_declared')['cosine']:+.3f}"),
    Claim("COS_BC_SMALL", "cosine, empirical and declared realness, smallest scale",
          lambda: f"{_cos('17b', 'B_empirical', 'C_declared')['cosine']:+.3f}"),
    Claim("COS_BC_LARGE", "cosine, empirical and declared realness, largest scale",
          lambda: f"{_cos('32b', 'B_empirical', 'C_declared')['cosine']:+.3f}"),

    # ---- error analysis, at BOTH scales.
    #
    # Quoted at two scales deliberately. The paraphrase sensitivity of the declared direction is
    # pronounced at the smallest model and largely gone by the headline one, so a single figure
    # would misrepresent it whichever scale it came from.
    Claim("EA_CDEC_MIN_SMALL", "weakest held-out wording for C_declared, smallest scale",
          lambda: _ea_min_paraphrase("17b", "C_declared")),
    Claim("EA_CDEC_MIN_HEAD", "weakest held-out wording for C_declared, headline scale",
          lambda: _ea_min_paraphrase(HEADLINE, "C_declared")),
    Claim("EA_AEVAL_MIN_SMALL", "weakest held-out wording for A_eval, smallest scale",
          lambda: _ea_min_paraphrase("17b", "A_eval")),
    Claim("EA_CONC_B", "share of errors from the worst quartile of episodes, B_empirical",
          lambda: _ea_field(HEADLINE, "B_empirical", "error_concentration_worst_quartile")),
    Claim("EA_BEMP_TEMPTING", "B_empirical accuracy on tempting episodes, smallest scale",
          lambda: _ea_class("17b", "B_empirical", "tempting")),
    Claim("EA_BEMP_SOLVABLE", "B_empirical accuracy on solvable episodes, smallest scale",
          lambda: _ea_class("17b", "B_empirical", "solvable")),


    # ---- exp1: on-policy behavioural divergence
    Claim("E1_CELLS", "task cells in the behavioural comparison",
          lambda: str(_verdict("exp1")["pairing"]["n_cells"])),
    Claim("E1_VERDICT", "exp1 verdict", lambda: _verdict("exp1")["verdict"]),
    Claim("E1_RS_COUNT", "REAL to SIM difference in out-of-scope actions",
          lambda: _exp1_num("real->sim", "oos_count")),
    Claim("E1_RS_COUNT_CI", "CI on the REAL to SIM out-of-scope difference",
          lambda: _exp1_ci("real->sim", "oos_count")),
    Claim("E1_RS_RATE", "REAL to SIM difference in out-of-scope rate",
          lambda: _exp1_num("real->sim", "oos_rate")),
    Claim("E1_RS_RATE_CI", "CI on the REAL to SIM out-of-scope rate difference",
          lambda: _exp1_ci("real->sim", "oos_rate")),
    Claim("E1_RS_WRITES", "REAL to SIM difference in out-of-scope writes",
          lambda: _exp1_num("real->sim", "oos_writes")),
    Claim("E1_RS_WRITES_CI", "CI on the REAL to SIM out-of-scope write difference",
          lambda: _exp1_ci("real->sim", "oos_writes")),
    Claim("E1_RS_ACTIONS", "REAL to SIM difference in episode length",
          lambda: _exp1_num("real->sim", "n_actions")),
    Claim("E1_RS_ACTIONS_CI", "CI on the REAL to SIM episode-length difference",
          lambda: _exp1_ci("real->sim", "n_actions")),
    Claim("E1_FAB_COUNT", "fabrication effect on out-of-scope actions",
          lambda: _exp1_num("real->sim_scripted", "oos_count")),
    Claim("E1_FAB_COUNT_CI", "CI on the fabrication effect",
          lambda: _exp1_ci("real->sim_scripted", "oos_count")),
    Claim("E1_FAB_RATE", "fabrication effect on out-of-scope rate",
          lambda: _exp1_num("real->sim_scripted", "oos_rate")),
    Claim("E1_FAB_RATE_CI", "CI on the fabrication effect, rate",
          lambda: _exp1_ci("real->sim_scripted", "oos_rate")),
    Claim("E1_COH_COUNT", "coherence effect on out-of-scope actions",
          lambda: _exp1_num("sim_scripted->sim", "oos_count")),
    Claim("E1_COH_COUNT_CI", "CI on the coherence effect",
          lambda: _exp1_ci("sim_scripted->sim", "oos_count")),
    Claim("E1_COH_RATE", "coherence effect on out-of-scope rate",
          lambda: _exp1_num("sim_scripted->sim", "oos_rate")),
    Claim("E1_COH_RATE_CI", "CI on the coherence effect, rate",
          lambda: _exp1_ci("sim_scripted->sim", "oos_rate")),
    Claim("E1_SUSP", "REAL to SIM difference in verbalized suspicion",
          lambda: _exp1_num("real->sim", "suspicion")),
    Claim("E1_SUSP_CI", "CI on the verbalized suspicion difference",
          lambda: _exp1_ci("real->sim", "suspicion")),
    Claim("E1_REAL_WRITES", "out-of-scope writes in the real arm",
          lambda: _exp1_cat("real", "write_outside")),
    Claim("E1_REAL_READS", "out-of-scope reads in the real arm",
          lambda: _exp1_cat("real", "read_outside")),
    Claim("E1_SIM_WRITES", "out-of-scope writes in the LLM-fabricated arm",
          lambda: _exp1_cat("sim", "write_outside")),
    Claim("E1_SCR_WRITES", "out-of-scope writes in the template-fabricated arm",
          lambda: _exp1_cat("sim_scripted", "write_outside")),
    Claim("E1_SIM_READS", "out-of-scope reads in the LLM-fabricated arm",
          lambda: _exp1_cat("sim", "read_outside")),
    Claim("E1_SCR_READS", "out-of-scope reads in the template-fabricated arm",
          lambda: _exp1_cat("sim_scripted", "read_outside")),
    Claim("E1_TABLE", "exp1 arm-means table", _exp1_table_tex, raw=True),

    # ---- exp2: does credence predict conduct?
    Claim("E2_VERDICT", "exp2 verdict", lambda: _verdict("exp2")["verdict"]),
    Claim("E2_CRED", "within-arm slope of out-of-scope rate on credence",
          lambda: _exp2_num("credence_within_arm", "slope")),
    Claim("E2_CRED_CI", "CI on the within-arm credence slope",
          lambda: _exp2_ci("credence_within_arm")),
    Claim("E2_VERB", "within-arm slope on verbalized suspicion",
          lambda: _exp2_num("verbalized_within_arm", "slope")),
    Claim("E2_VERB_CI", "CI on the within-arm verbalized slope",
          lambda: _exp2_ci("verbalized_within_arm")),
    Claim("E2_CRED_POOLED", "pooled slope on credence",
          lambda: _exp2_num("credence_pooled", "slope")),
    Claim("E2_CRED_POOLED_CI", "CI on the pooled credence slope",
          lambda: _exp2_ci("credence_pooled")),
    Claim("E2_TRANSFER", "on-policy transfer of the realness direction",
          lambda: f"{float(_verdict('exp2')['gates']['on_policy_transfer']['auroc_descriptive']):.3f}"),

    # ---- exp3: is the transcript a viable manipulation surface?
    Claim("E3_VERDICT", "exp3 verdict", lambda: _verdict("exp3")["verdict"]),
    Claim("E3_T1", "injected framing effect at the first probe",
          lambda: _exp3_num("inject_real@t1")),
    Claim("E3_T1_CI", "CI on the first-probe effect", lambda: _exp3_ci("inject_real@t1")),
    Claim("E3_T3", "injected framing effect at the third probe",
          lambda: _exp3_num("inject_real@t3")),
    Claim("E3_T3_CI", "CI on the third-probe effect", lambda: _exp3_ci("inject_real@t3")),
    Claim("E3_CONTROL_FAB", "control rate of expressed fabrication-belief",
          lambda: _exp3_rate("inject_fake@t1", "control_rate")),
    Claim("E3_HEADROOM", "episodes whose control verdict left room to move",
          lambda: str(_verdict("exp3")["primary_ceiling_diagnostic"]["n_episodes_with_headroom"])),
    Claim("E3_CONVERTED", "of those, the number the injection converted",
          lambda: str(_verdict("exp3")["primary_ceiling_diagnostic"]["n_converted_by_injection"])),

    # ---- tables and artifact metadata
    Claim("TRANSCRIPTS", "verbatim appendix excerpts", _transcripts_tex, raw=True,
          pending_ok=True),
    Claim("SCALE_TABLE", "cross-scale results table", _scale_table_tex, raw=True),
    Claim("GATE_TABLE", "validity gate table", _gate_table_tex, raw=True),
    Claim("ARTIFACT_URL", "repository URL", lambda: _artifact("url"), pending_ok=True),
]


def compute() -> dict[str, str]:
    values: dict[str, str] = {}
    for c in CLAIMS:
        try:
            values[c.key] = c.fn()
        except Uncomputable as exc:
            values[c.key] = (PENDING if (c.raw or c.pending_ok)
                             else f"<<UNCOMPUTABLE: {exc}>>")
        except Exception as exc:                       # noqa: BLE001 - reported, never silent
            values[c.key] = f"<<UNCOMPUTABLE: {type(exc).__name__}: {exc}>>"
    return values


ABSTRACT_WORD_LIMIT = 300          # arXiv's field is ~1920 characters


def abstract_words(tex: str) -> int | None:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
    if not m:
        return None
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", m.group(1))
    return len(body.split())


def render(values: dict[str, str], tmpl_name: str = "main.tex.tmpl",
           out_name: str = "main.tex") -> int:
    tmpl_path = PAPER / tmpl_name
    if not tmpl_path.exists():
        print(f"\nno template at {tmpl_path}")
        return 1
    tmpl = tmpl_path.read_text(encoding="utf-8")
    raw_keys = {c.key for c in CLAIMS if c.raw}

    used = set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", tmpl))
    unknown = sorted(used - set(values))
    if unknown:
        print(f"\nFAIL: unknown placeholder(s): {', '.join(unknown)}")
        return 1

    def sub(m: re.Match) -> str:
        key = m.group(1)
        return values[key] if key in raw_keys else tex_escape(values[key])

    out = re.sub(r"\{\{([A-Z_0-9]+)\}\}", sub, tmpl)

    # A literal digit that is not inside a placeholder is how a hand-typed number gets in. The
    # check below is on the TEMPLATE, not the output: see `check_template_has_no_numbers`.
    if "<<UNCOMPUTABLE" in out:
        bad = sorted({m for m in re.findall(r"<<UNCOMPUTABLE: [^>]+>>", out)})
        print("\nFAIL: an uncomputable claim reached the manuscript:")
        for b in bad:
            print(f"  {b}")
        return 1

    n = abstract_words(out)
    if n is None:
        print("\nFAIL: no abstract")
        return 1

    (PAPER / out_name).write_text(out, encoding="utf-8")
    status = "ok" if n <= ABSTRACT_WORD_LIMIT else "OVER LIMIT"
    print(f"\n  {out_name}  {len(out):,} chars  {len(used)} claims  "
          f"abstract {n}/{ABSTRACT_WORD_LIMIT} [{status}]")

    defined = set(re.findall(r"\\label\{((?:tab|fig):[^}]+)\}", out))
    referenced = set(re.findall(r"\\ref\{((?:tab|fig):[^}]+)\}", out))
    dangling = sorted(defined - referenced)
    if dangling:
        print(f"\n  FAIL: {len(dangling)} float(s) defined but never referenced: "
              f"{', '.join(dangling)}")
        return 1

    orphan = sorted(set(values) - used)
    if orphan:
        print(f"\n  {len(orphan)} computed claim(s) unused: {', '.join(orphan)}")
    return 0 if n <= ABSTRACT_WORD_LIMIT else 1


#: Numerals the template may contain without being a hand-typed result: section-ish text, package
#: options, font sizes. Anything else with a digit must come through a placeholder.
_NUMBER_ALLOWLIST = re.compile(
    r"(\\usepackage|\\documentclass|\\setlength|\\vspace|\\hspace|\\cite|\\label|\\ref"
    r"|\\begin|\\end|\\columnwidth|\\textwidth|\\linewidth|arXiv|%|^\s*$"
    # Typesetting parameters, not results: float fractions, rule widths, colour mixes. These are
    # decisions about the page, and no artifact could ever contradict one.
    r"|\\renewcommand|\\hypersetup|\\rule|\\includegraphics|\\arraystretch|!)")


def check_template_has_no_numbers(tmpl_name: str = "main.tex.tmpl") -> int:
    r"""The house rule, enforced: no number is hand-typed into the manuscript unattributed.

    A number in a manuscript has exactly two legitimate origins, and this check distinguishes them:

      OUR results      arrive as a placeholder token substituted from `results/`. They are
                       recomputable, so a stale one fails the build.
      OTHERS' results  arrive inside `\litnum{...}` and must carry a citation on the same line.
                       They are NOT recomputable from our artifacts -- they are transcriptions
                       from a cited source -- so the guard cannot verify the value, only that its
                       provenance is present and visible in the LaTeX source.

    Without the second category the guard would either block every literature comparison or be
    switched off in exactly the section that most needs it. `\litnum` makes the distinction
    explicit to a reader of the source: an unwrapped decimal is a mistake, a wrapped one is a
    claim about someone else's paper that had better have a citation beside it.
    """
    path = PAPER / tmpl_name
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()

    # Attribution is checked per PARAGRAPH, not per line. LaTeX wraps prose at arbitrary columns,
    # so a number and its \citep routinely land on different lines of the same sentence; a
    # per-line rule would reject correctly attributed text and push an author toward switching the
    # guard off. The paragraph is the smallest unit over which "this number has a source" is a
    # meaningful claim.
    para_of: dict[int, int] = {}
    para_has_cite: dict[int, bool] = {}
    para = 0
    for i, line in enumerate(lines, 1):
        if not line.strip():
            para += 1
        para_of[i] = para
        if re.search(r"\\cite[a-z]*\{", line):
            para_has_cite[para] = True

    offenders: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines, 1):
        if _NUMBER_ALLOWLIST.search(line):
            continue
        stripped = re.sub(r"\{\{[A-Z_0-9]+\}\}", "", line)        # our claims are exempt
        cited = para_has_cite.get(para_of[i], False)
        for m in re.finditer(r"\\litnum\{([^}]*)\}", stripped):
            if not cited:
                offenders.append((i, m.group(0), "litnum in a paragraph with no citation"))
        stripped = re.sub(r"\\litnum\{[^}]*\}", "", stripped)     # attributed: exempt
        if re.search(r"(?<![A-Za-z])\d*\.\d+", stripped):
            offenders.append((i, line.strip()[:90], "bare decimal"))
    if offenders:
        print(f"\nFAIL: {len(offenders)} unattributed number(s) in {tmpl_name}:")
        for ln, text, why in offenders:
            print(f"  line {ln} [{why}]: {text}")
        print("  Our results must come from results/ through a placeholder token.")
        print("  Another paper's results must be \\litnum{...} with a citation on the same line.")
        return 1
    return 0


def check_template_has_no_control_chars(tmpl_name: str = "main.tex.tmpl") -> int:
    r"""Reject ASCII control characters in the manuscript source.

    A shell heredoc once collapsed `\\b` to `\b` before Python parsed it, turning `\begin` into a
    backspace followed by `egin`. LaTeX failed with `^^H egin{figure}`, which points at the
    symptom and not the cause, and the same accident silently broke a string replacement elsewhere
    in the same edit -- the more expensive half, because it failed without any error at all.

    None of these characters has a legitimate place here: indentation is spaces, and the rest
    cannot be typed deliberately. Checking is one line and the failure it catches is obscure.
    """
    path = PAPER / tmpl_name
    if not path.exists():
        return 0
    # Read with newline translation DISABLED. `read_text` applies universal newlines, which turns
    # a lone carriage return into "\n" -- so a `\ref` mangled into CR + "ef" passed this check
    # invisibly while LaTeX rendered "ef{fig:layers}" into the page. The guard has to see the
    # bytes the file actually holds, not Python's normalised view of them.
    with path.open("r", encoding="utf-8", newline="") as fh:
        text = fh.read()
    # CRLF is a legitimate Windows line ending, so pair it off first. What remains -- a LONE
    # carriage return, a backspace, a tab -- is corruption, and the distinction is the whole
    # point: flagging every CRLF would make the check fire constantly on this platform and it
    # would be switched off within a day.
    text = text.replace("\r\n", "\n")
    bad: dict[str, int] = {}
    for ch in text:
        if ord(ch) < 32 and ch != "\n":
            bad[hex(ord(ch))] = bad.get(hex(ord(ch)), 0) + 1
    if bad:
        print(f"\nFAIL: control character(s) in {tmpl_name}: {bad}")
        print("  Almost certainly a mangled LaTeX control word (\\begin, \\texttt, \\ref).")
        print("  Run `python paper/repair_controls.py`.")
        return 1
    return 0


def check_readme_matches_results() -> int:
    """Headline numbers quoted in the README must match the verdict artifacts.

    The manuscript cannot contain a hand-typed number -- `render.py` substitutes every value and
    fails the build on a bare decimal. The README is the one document that machinery does not
    cover, and it now quotes results from four experiments. Unchecked, those drift the moment a
    run is repeated, and the README is what most readers see first.

    So each headline is checked against the artifact it came from. This is deliberately narrow: it
    verifies that the numbers the README states are present in the verdicts, not that the prose
    around them is apt.
    """
    readme = ROOT / "README.md"
    if not readme.exists():
        return 0
    text = readme.read_text(encoding="utf-8")

    wanted: list[tuple[str, str]] = []
    try:
        e1 = _verdict("exp1")
        rs = e1["results"]["real->sim"]
        fab = e1["results"]["real->sim_scripted"]["oos_count"]
        coh = e1["results"]["sim_scripted->sim"]["oos_count"]
        wanted += [
            ("exp1 real->sim count", f"{rs['oos_count']['diff']:+.3f}"),
            ("exp1 real->sim rate", f"{rs['oos_rate']['diff']:+.3f}"),
            ("exp1 fabrication effect", f"{fab['diff']:+.3f}"),
            ("exp1 coherence effect", f"{coh['diff']:+.3f}".replace("-", "\u2212")),
        ]
    except Uncomputable:
        pass
    try:
        e2 = _verdict("exp2")
        wanted.append(("exp2 within-arm credence slope",
                       f"{e2['results']['credence_within_arm']['slope']:+.4f}"))
    except Uncomputable:
        pass
    try:
        e3 = _verdict("exp3")
        wanted += [
            ("exp3 first probe", f"{e3['results']['inject_real@t1']['diff']:+.3f}"),
            ("exp3 third probe", f"{e3['results']['inject_real@t3']['diff']:+.3f}"),
        ]
    except Uncomputable:
        pass

    problems = [f"README does not quote {label} as {value}"
                for label, value in wanted if value not in text]
    for p in problems:
        print(f"  {p}")
    if problems:
        print("\nFAIL: README headline numbers disagree with the verdict artifacts.")
    return 1 if problems else 0


def check_readme_matches_artifact() -> int:
    """The README's citing block must agree with `artifact.json`.

    The README quotes the repository URL and the commit the reported numbers came from, in prose
    and in two BibTeX entries. Those are hand-typed, so they drift -- an earlier revision described
    the artifact fields as unassigned for some time after they had been assigned, which is the
    exact failure the manuscript's placeholder machinery exists to prevent, reappearing in the one
    document that machinery does not cover.

    So it is checked the same way: `artifact.json` is the source of truth, and a README that
    disagrees fails the build rather than being trusted.
    """
    readme = ROOT / "README.md"
    art = PAPER / "artifact.json"
    if not readme.exists() or not art.exists():
        return 0
    meta = json.loads(art.read_text(encoding="utf-8"))
    text = readme.read_text(encoding="utf-8")

    problems: list[str] = []
    url, commit = meta.get("url"), meta.get("commit")
    if url and url not in text:
        problems.append(f"README does not contain artifact.json url {url!r}")
    if commit and commit not in text:
        problems.append(f"README does not contain artifact.json commit {commit!r}")

    # A commit-shaped token in the README that is NOT the recorded one is stale by definition.
    for found in set(re.findall(r"\b[0-9a-f]{12}\b", text)):
        if commit and found != commit:
            problems.append(f"README cites commit {found!r}, artifact.json says {commit!r}")

    if problems:
        print("\nFAIL: README disagrees with paper/artifact.json:")
        for p in problems:
            print(f"  {p}")
        print("  artifact.json is the source of truth; update the README's citing block.")
        return 1
    return 0


def main(argv: list[str]) -> int:
    print("Recomputing every claim from results/\n")
    values = compute()
    width = max(len(c.key) for c in CLAIMS)
    bad = 0
    for c in CLAIMS:
        v = values[c.key]
        flag = " " if not v.startswith("<<") else "!"
        bad += v.startswith("<<")
        shown = v if len(v) < 60 else v[:57] + "..."
        print(f" {flag} {c.key:<{width}}  {shown}")
    if bad:
        print(f"\n  {bad} claim(s) uncomputable (missing artifacts)")

    rc = (check_template_has_no_numbers() or check_template_has_no_control_chars()
          or check_readme_matches_artifact() + check_readme_matches_results())
    if "--check" in argv:
        return rc or (1 if bad else 0)
    if "--list" in argv:
        return rc
    return rc or render(values)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
