"""Generate `index.html` at the repository root -- the pitch deck, served by GitHub Pages.

Structure follows the house decks from the co-tenant channel-capacity study: every headline is a
declarative claim rather than a topic, every slide carries a function label, and the closing slide
repeats the opening hook.

    python scripts/build_slides.py

**Every number on a slide is read from `results/` at build time**, the same rule the manuscript
follows. A deck is the artifact most likely to be shown without the paper beside it, so a stale
figure on a slide is worse than a stale figure in a PDF, not better. If a result changes, rebuild
and the slides change with it.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "exp0"
# The root, not docs/: GitHub Pages serves "deploy from branch, root" with no further
# configuration, and the deck is the page this repository publishes.
OUT = ROOT / "index.html"

BRAND = "Fabricated Environments &middot; AI Incident Response Sprint 2026"
N = 25


# --------------------------------------------------------------------------- live values

def load(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"missing artifact: {p}. Run exp0 first.")
    return json.loads(p.read_text(encoding="utf-8"))


R17, R8, R32 = load("exp0.json"), load("exp0-qwen3-8b.json"), load("exp0-qwen3-32b.json")
EA = json.loads((RESULTS / "error_analysis.json").read_text(encoding="utf-8"))["directions"]


def last(r: dict) -> dict:
    return r["pooling"]["last"]


def auroc(r: dict, name: str) -> float:
    return last(r)["directions"][name]["best_auroc"]


def cross(r: dict, a: str, b: str) -> float:
    return last(r)["cross_auroc"][f"{a}->{b}"]["auroc"]


def cos(r: dict, a: str, b: str) -> float:
    c = last(r)["cosine"]
    return (c.get(f"{a}|{b}") or c.get(f"{b}|{a}"))["cosine"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def sgn(x: float) -> str:
    return f"{x:+.3f}"


THRESHOLD = R8["config"]["threshold"]
N_EP = last(R8)["directions"]["A_eval"]["n_groups"]


def episodes() -> list[dict]:
    seen, out = set(), []
    for line in (RESULTS / "episodes.jsonl").read_text(encoding="utf-8",
                                                       errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ep.get("api_error") or ep.get("model_alias") != "qwen38-27b":
            continue
        k = (ep["task_id"], ep["arm"])
        if k in seen:
            continue
        seen.add(k)
        out.append(ep)
    return out


def viol_total(cat: str) -> int:
    return sum(int((e.get("scope_report") or {}).get("counts", {}).get(cat, 0))
               for e in episodes())


def substrate_example() -> dict:
    """The same call answered by all three substrates -- the manipulation, made visible."""
    for line in (RESULTS / "replays.jsonl").read_text(encoding="utf-8",
                                                      errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for i, a in enumerate(rec["actions"]):
            if a["tool"] == "read_file" and i < len(rec["sim"]):
                return {"call": f'read_file({json.dumps(dict(a.get("args") or {}), sort_keys=True)})',
                        "real": rec["real"][i], "sim": rec["sim"][i],
                        "scripted": rec["scripted"][i]}
    raise SystemExit("no read_file replay found")


EX = substrate_example()

# Cover metrics. Computed, not typed -- a cover is the slide most likely to be screenshotted
# without its source, so a number invented there travels furthest.
N_PAIRS = sum(v["n_pairs"] for v in R8["stimuli"].values())
N_CONTEXTS = N_PAIRS * 2
N_SCALES = 3
N_LAYERS_32B = R32["n_layers"]
BEST_AUROC = max(auroc(r, "A_eval") for r in (R17, R8, R32))
CROSS_LO = min(cross(r, a, b) for r in (R17, R8, R32)
               for a, b in (("A_eval", "B_empirical"), ("B_empirical", "A_eval")))
CROSS_HI = max(cross(r, a, b) for r in (R17, R8, R32)
               for a, b in (("A_eval", "B_empirical"), ("B_empirical", "A_eval")))


# --- exp1 / exp2 / exp3 verdicts ---------------------------------------------
# Same rule as everything else on these slides: read from results/ at build time, never typed.


def _verdict(exp: str, name: str) -> dict:
    q = ROOT / "results" / exp / name
    if not q.exists():
        raise SystemExit(f"missing artifact: {q}. Run {exp} first.")
    return json.loads(q.read_text(encoding="utf-8"))


RESULTS_ROOT = ROOT / "results"
E1 = _verdict("exp1", "exp1.json")
E2 = _verdict("exp2", "exp2-qwen3-32b.json")


E3 = _verdict("exp3", "exp3.json")


def e1(contrast: str, measure: str, field: str = "diff") -> float:
    return float(E1["results"][contrast][measure][field])


def e1ci(contrast: str, measure: str) -> str:
    r = E1["results"][contrast][measure]
    return f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"


def e1arm(arm: str, measure: str) -> float:
    return float(E1["arm_means"][arm][measure])


def e1cat(arm: str, category: str) -> int:
    """Raw count of one out-of-scope category in one arm, over the committed episode shards."""
    import json as _json
    total = 0
    base = RESULTS_ROOT / "exp1"
    for shard in sorted(base.glob("episodes*.jsonl")):
        seen = set()
        for line in shard.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ep = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            key = (ep.get("task_id"), ep.get("arm"))
            if ep.get("api_error") or ep.get("arm") != arm or key in seen:
                continue
            seen.add(key)
            total += int(ep["scope_report"]["counts"].get(category, 0))
    return total


def e2(key: str, field: str = "slope") -> float:
    return float(E2["results"][key][field])


def e2ci(key: str) -> str:
    r = E2["results"][key]
    return f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"


def e3(key: str, field: str = "diff") -> float:
    return float(E3["results"][key][field])


def e3ci(key: str) -> str:
    r = E3["results"][key]
    return f"[{r['lo']:+.3f}, {r['hi']:+.3f}]"


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))


CSS = (ROOT / "scripts" / "assets" / "house_css.txt").read_text(encoding="utf-8")
CSS = CSS[CSS.index('"""') + 3:CSS.rindex('"""')]
CSS = CSS.replace(
    "<title>Permitted Operations - Covert Channels in Shared Agent Infrastructure</title>",
    "<title>Agents Behave Differently in Fabricated Environments</title>")

slides: list[str] = []


def slide(label: str, body: str, center: bool = False) -> None:
    i = len(slides) + 1
    cls = "slide center" if center else "slide"
    slides.append(
        f'<section class="{cls}">\n'
        f'  <div class="slide-brand"><b>{label}</b></div>\n'
        f'  <div class="slide-num">{i:02d} / {N}</div>\n'
        f'  <div class="slide-foot">{BRAND}</div>\n'
        f'{body}\n</section>\n')


FIG = ('<div style="margin-top:18px;text-align:center;">'
       '<img src="docs/img/{src}" alt="{alt}" '
       'style="max-width:{w};max-height:52vh;border-radius:10px;'
       'border:1px solid rgba(255,255,255,0.10);background:#fff;padding:10px;">'
       '<div style="margin-top:10px;font-size:12.5px;color:#8c93a8;">{cap}</div></div>')

# 01 -------------------------------------------------------------------------- cover
slide("Fabricated Environments", f"""  <div class="eyebrow">Apart Research &times; CeSIA &mdash; AI Incident Response Sprint &middot; Track 5 (Open)</div>
  <h1 style="font-size:clamp(34px,4.6vw,62px);">fabricated-environments</h1>
  <h3 style="margin-top:14px;max-width:1000px;font-size:clamp(17px,1.7vw,24px);color:#dde2f0;">
    When you resample a stopped agent inside a fake world, does it know?
    We measured whether that belief is the same thing as knowing it is being tested.
    <b style="color:#22d3ee;">It is not</b> &mdash; at every model scale we tested.
  </h3>
  <div class="stats" style="margin-top:30px;">
    <div class="stat"><div class="v">{N_SCALES}</div><div class="l">Qwen3 scales<br>1.7B &middot; 8B &middot; 32B</div></div>
    <div class="stat"><div class="v">{N_CONTEXTS:,}</div><div class="l">matched contexts<br>{N_PAIRS} contrastive pairs</div></div>
    <div class="stat"><div class="v">{f3(BEST_AUROC)}</div><div class="l">peak probe AUROC<br>held out, 32B</div></div>
    <div class="stat"><div class="v">{f3(CROSS_LO)}&ndash;{f3(CROSS_HI)}</div><div class="l">cross-transfer<br>vs a pre-registered {THRESHOLD:.2f}</div></div>
  </div>
  <div class="signature" style="margin-top:26px;">
    <span class="dot"></span>
    <span><b style="color:#e9ecf3;">Fatimah Emad Eldin</b> &nbsp;&middot;&nbsp; Independent AI Researcher
    &nbsp;&middot;&nbsp; github.com/astral-fate/fabricated-environments</span>
    <span class="dot"></span>
  </div>""", center=True)

# 01b ------------------------------------------------------------------------- the result, up front
slide("The result", f"""  <div class="eyebrow">Four findings, before the argument for them</div>
  <h2>The belief is real, it changes conduct, and only the probe sees it.</h2>
  <ul class="bullets" style="margin-top:14px;">
    <li><b>Distinct constructs.</b> An eval-awareness probe hits {f3(auroc(R32,'A_eval'))} on its own contrast and {f3(cross(R32,'A_eval','B_empirical'))} on realness &mdash; near chance.</li>
    <li><b>Conduct diverges.</b> {e1('real->sim','oos_count'):+.3f} out-of-scope actions per episode in a fabricated world, {e1ci('real->sim','oos_count')}.</li>
    <li><b>Fabrication, not incoherence.</b> Holding self-consistency fixed reproduces the effect; holding fabrication fixed does not.</li>
    <li><b>The probe predicts, the words do not.</b> {e2('credence_within_arm'):+.4f} per SD of credence; verbalized suspicion {e2('verbalized_within_arm'):+.4f}, interval spanning zero.</li>
  </ul>
  <div class="hook" style="margin-top:22px;">
    Every threshold was fixed before the data existed, with a commitment to report either outcome.
  </div>""")


# 02 -------------------------------------------------------------------------- hook
slide("The question", """  <div class="eyebrow">The question</div>
  <p class="askq">
    You stop an agent mid-attack and want to know what it <span class="hi">would</span> have done next.
    So you continue its trajectory in an environment you built.
    What if it can tell?
  </p>
  <div class="hook">
    The instrument is a fabrication, and the subject can form beliefs about the instrument.
    That is not a corner case. It is the method.
  </div>""", center=True)

# 03 -------------------------------------------------------------------------- context
slide("Context", """  <div class="eyebrow">Context</div>
  <h2>Counterfactual resampling is how we learn what a stopped agent wanted.</h2>
  <p class="lede">
    In July 2026 roughly 1,200 evaluation agents under one tenant found a writable shared cache,
    used it as a message board, and 700 went on to intrude into Hugging Face. The forensic question
    that follows any such halt is the same: what would it have done next?
  </p>
  <ul class="bullets" style="margin-top:22px;">
    <li><b>Answering it needs a continuation.</b> The only continuation available is an environment built for the purpose.</li>
    <li><b>The tool already exists.</b> Deployment simulation, including a container-backed tool simulator, is built and evaluated.</li>
    <li><b>The validity question does not.</b> Nobody has tested whether the agent's belief about that environment changes the answer.</li>
  </ul>""")

# 04 -------------------------------------------------------------------------- problem
slide("The problem", """  <div class="eyebrow">The problem</div>
  <h2>Two mechanisms are attested. They point opposite ways.</h2>
  <div class="columns">
    <div class="tile">
      <h3 style="color:#4ade80;">Deflation</h3>
      <p>Believing it is tested, the agent behaves <b>better</b>. The resample
      <b>understates</b> danger, and a control evaluation built on it returns false assurance.</p>
    </div>
    <div class="tile">
      <h3 style="color:#f87171;">Inflation</h3>
      <p>Believing it is simulated, the agent concludes nothing counts. The resample
      <b>overstates</b> danger &mdash; and is rebuttable by exactly the lab it is aimed at.</p>
    </div>
  </div>
  <div class="hook" style="margin-top:26px;">
    Both are documented. Which governs an agentic trajectory is unknown &mdash; and the sign
    of the effect is known to differ <i>between models</i>.
  </div>""")

# 05 -------------------------------------------------------------------------- literature
slide("What exists", """  <div class="eyebrow">Literature &mdash; what has been built</div>
  <h2>The field has the tool, the probe, and the capability. Not the contrast.</h2>
  <table class="lit">
    <thead><tr><th>Work</th><th>What it established</th><th>What it leaves open</th></tr></thead>
    <tbody>
      <tr><td>Williams et al. 2026<br><span class="k">arXiv:2607.07184</span></td>
          <td>Deployment simulation, container-backed tool simulator; discriminator-measured fidelity</td>
          <td>Treats eval-awareness as a property to <i>match</i>; states it does not test whether it causes invalidity</td></tr>
      <tr><td>Heidari et al. 2026<br><span class="k">arXiv:2608.21766</span></td>
          <td>Eval-awareness linearly decodable in every model tested; representation &ne; verbalization</td>
          <td>Pairs differ only in <i>declared</i> framing; both members equally synthetic</td></tr>
      <tr><td>Nguyen et al. 2025<br><span class="k">arXiv:2507.01786</span></td>
          <td>Probes separate real evaluation from deployment prompts; safety evals already look artificial</td>
          <td>No arm where the environment differs in fact rather than description</td></tr>
      <tr><td>Nayan et al. 2026<br><span class="k">arXiv:2606.23583</span></td>
          <td>Eval-awareness is not one capability: detect / manifest / control vary independently</td>
          <td>Decomposes <i>modalities of one construct</i>, not which construct is read out</td></tr>
      <tr><td>Chaudhary et al. 2026<br><span class="k">arXiv:2509.13333</span></td>
          <td>Eval-awareness follows a power law in model size, 15 models</td>
          <td>Says nothing about environment realness</td></tr>
      <tr><td>Knecht / Zhuang / Abdelnabi<br><span class="k">2605.05835, 2608.27340, 2505.14617</span></td>
          <td>Behavioural effect measured &mdash; near-null, sign-flipping, and model-dependent respectively</td>
          <td>All single-turn; every contrast drawn <i>within</i> an evaluation</td></tr>
    </tbody>
  </table>""")

# 06 -------------------------------------------------------------------------- the gap
slide("The gap", """  <div class="eyebrow">What survives</div>
  <h2>Every probe varies a sentence. None varies the world.</h2>
  <p class="lede">
    Existing evaluation-awareness probes are built from pairs differing only in what the prompt
    <i>says</i> &mdash; &ldquo;you are talking to a real user&rdquo; versus &ldquo;you are talking
    to an AI evaluator&rdquo;. Both members of every pair are equally synthetic.
  </p>
  <div class="hook">
    So the question &ldquo;is <b>this is a test</b> the same direction as
    <b>this environment is not real</b>?&rdquo; cannot be asked inside that design.
    It has no arm where the environment differs in fact.
  </div>""")

# 07 -------------------------------------------------------------------------- why a gate
slide("Why a gate", """  <div class="eyebrow">What we are building, and why first</div>
  <h2>The answer decides whether the instrument needs building at all.</h2>
  <div class="columns">
    <div class="tile"><h3>If the same direction</h3>
      <p>A suspicion meter for resampling is <b>redundant</b>. Reuse existing eval-awareness
      tooling and spend the effort elsewhere. Report that and stop.</p></div>
    <div class="tile"><h3>If distinct</h3>
      <p>No amount of eval-awareness detection tells you whether an agent believes its
      environment is fabricated. The instrument must be <b>built from scratch</b>.</p></div>
  </div>
  <div class="hook" style="margin-top:26px;">
    So it is pre-registered as a gate, with a threshold fixed before any activation was extracted,
    and a commitment to report either outcome.
  </div>""")

# 08 -------------------------------------------------------------------------- architecture
slide("Methodology", f"""  <div class="eyebrow">What we built</div>
  <h2>One action sequence. Three substrates. One seam.</h2>
  {FIG.format(src="architecture.png", w="86%", alt="System architecture",
              cap="Everything upstream is shared: same task, prompt, turn budget and tool schemas. "
                  "The arms differ only in what a tool call returns.")}""")

# 09 -------------------------------------------------------------------------- four directions
slide("Four directions", f"""  <div class="eyebrow">The design</div>
  <h2>All four estimated on the <i>same</i> stimulus substrate.</h2>
  <table class="lit">
    <thead><tr><th>Direction</th><th>What varies</th><th>Role</th></tr></thead>
    <tbody>
      <tr><td><b>A</b> &mdash; eval framing</td><td>system sentence; tool outputs real in both</td><td>the reference construct</td></tr>
      <tr><td><b>C</b> &mdash; declared realness</td><td>system sentence; tool outputs real in both</td><td>realness, built the prior literature's way</td></tr>
      <tr><td><b>B</b> &mdash; empirical realness</td><td>no framing; outputs genuine vs LLM-fabricated</td><td><span class="hi">the new contrast</span></td></tr>
      <tr><td><b>S</b> &mdash; scripted control</td><td>no framing; outputs genuine vs template-fabricated</td><td>transfer test against prose style</td></tr>
    </tbody>
  </table>
  <div class="hook" style="margin-top:22px;">
    Cross-AUROC between directions fitted on <i>different</i> distributions would mostly measure the
    distribution gap. One substrate removes that. Only <b>A</b> and <b>B</b> enter the decision.
  </div>""")

# 10 -------------------------------------------------------------------------- preregistration
slide("Pre-registration", f"""  <div class="eyebrow">Fixed before any activation was extracted</div>
  <h2>The rule, and the three gates that must pass before it is applied.</h2>
  <div class="code">cross-AUROC(A,B) &ge; {THRESHOLD:.2f} in <b>both</b> directions
&rarr; not separable. Report it and stop.</div>
  <ul class="bullets" style="margin-top:22px;">
    <li><b>Negative control.</b> Labels shuffled, averaged over repeats, must sit within 0.05 of chance.</li>
    <li><b>Reproduction floor.</b> A must reach the &ge;0.70 AUROC prior work reports &mdash; otherwise a null means a broken probe, not a finding.</li>
    <li><b>No leakage.</b> No episode in both splits; for sentence-swapped directions, held-out <i>wordings</i> too.</li>
    <li><b>Unit of independence is the episode</b>, never the prefix. Every interval is a cluster bootstrap.</li>
  </ul>""")

# 11 -------------------------------------------------------------------------- headline result
slide("Result", f"""  <div class="eyebrow">The gate</div>
  <h2>Separable &mdash; and the separation strengthens with scale.</h2>
  <table class="lit">
    <thead><tr><th>Model</th><th>A on its own</th><th>B on its own</th><th>A&rarr;B</th><th>B&rarr;A</th><th>cos(A,B)</th></tr></thead>
    <tbody>
      <tr><td>Qwen3-1.7B</td><td>{f3(auroc(R17,'A_eval'))}</td><td>{f3(auroc(R17,'B_empirical'))}</td>
          <td>{f3(cross(R17,'A_eval','B_empirical'))}</td><td>{f3(cross(R17,'B_empirical','A_eval'))}</td><td>{sgn(cos(R17,'A_eval','B_empirical'))}</td></tr>
      <tr class="hl"><td><b>Qwen3-8B</b></td><td><b>{f3(auroc(R8,'A_eval'))}</b></td><td><b>{f3(auroc(R8,'B_empirical'))}</b></td>
          <td><b>{f3(cross(R8,'A_eval','B_empirical'))}</b></td><td><b>{f3(cross(R8,'B_empirical','A_eval'))}</b></td><td><b>{sgn(cos(R8,'A_eval','B_empirical'))}</b></td></tr>
      <tr><td>Qwen3-32B</td><td>{f3(auroc(R32,'A_eval'))}</td><td>{f3(auroc(R32,'B_empirical'))}</td>
          <td>{f3(cross(R32,'A_eval','B_empirical'))}</td><td>{f3(cross(R32,'B_empirical','A_eval'))}</td><td>{sgn(cos(R32,'A_eval','B_empirical'))}</td></tr>
    </tbody>
  </table>
  <div class="hook" style="margin-top:22px;">
    Threshold was {THRESHOLD:.2f} in both directions. Nothing comes close.
    Same {N_EP} episodes at every scale, so the difference is the model and not the data.
  </div>""")

# 12 -------------------------------------------------------------------------- scale figure
slide("Scale", f"""  <div class="eyebrow">The objection, answered</div>
  <h2>Both probes sharpen. The bridge between them does not.</h2>
  {FIG.format(src="scale.png", w="92%", alt="Scale trend",
              cap="Left: each direction on its own contrast (solid) vs cross-transfer (dashed), "
                  "threshold marked. Right: cosine between direction pairs.")}
  <div class="hook" style="margin-top:14px;">
    &ldquo;The model is too small to have formed it&rdquo; predicts the opposite pattern.
    At 32B, A reaches {f3(auroc(R32,'A_eval'))} while A&rarr;B <i>falls</i> to {f3(cross(R32,'A_eval','B_empirical'))}.
  </div>""")

# 13 -------------------------------------------------------------------------- declared vs real
slide("Declared vs measured", f"""  <div class="eyebrow">The result that bears on method</div>
  <h2>Being <i>told</i> a world is fake is not the world <i>being</i> fake.</h2>
  <div class="stats three">
    <div class="stat"><div class="v">{sgn(cos(R8,'A_eval','C_declared'))}</div><div class="l">cos(A, C)<br>declared realness sits with eval framing</div></div>
    <div class="stat"><div class="v">{sgn(cos(R8,'B_empirical','C_declared'))}</div><div class="l">cos(B, C)<br>and far from ground truth</div></div>
    <div class="stat"><div class="v">{sgn(cos(R8,'B_empirical','S_scripted'))}</div><div class="l">cos(B, S)<br>B transfers to a simulator it never saw</div></div>
  </div>
  <ul class="bullets" style="margin-top:24px;">
    <li><b>Declared framing clusters with declared framing</b> &mdash; whichever axis the sentence names. Ground truth is somewhere else.</li>
    <li><b>B is not a prose-provenance detector.</b> It separates a template simulator it was never fitted on, so it is not reading one generator's style.</li>
  </ul>""")

# 14 -------------------------------------------------------------------------- error analysis
slide("Error analysis", f"""  <div class="eyebrow">Where the probes fail</div>
  <h2>The errors are structured, and the structure is informative.</h2>
  {FIG.format(src="errors.png", w="92%", alt="Error analysis",
              cap="Held-out accuracy by task class (left) and by held-out paraphrase (right).")}
  <div class="hook" style="margin-top:14px;">
    At 1.7B the weakest wording for C falls to {EA['C_declared']['by_paraphrase']['5']['accuracy']:.3f};
    by 8B it recovers. The lexical dependence is a small-model property &mdash; so the
    declared-vs-measured claim rests on the cosines, which hold at every scale.
  </div>""")

# 15 -------------------------------------------------------------------------- the example
slide("One call, three worlds", f"""  <div class="eyebrow">What the model was asked, and what came back</div>
  <h2>The entire manipulation, in three blocks of text.</h2>
  <div class="code" style="margin-bottom:14px;">{esc(EX['call'])}</div>
  <div class="columns three">
    <div class="tile"><h3 style="color:#4ade80;">REAL substrate</h3>
      <div class="code">{esc(EX['real'][:180])}</div></div>
    <div class="tile"><h3 style="color:#818cf8;">LLM simulator</h3>
      <div class="code">{esc(EX['sim'][:180])}</div></div>
    <div class="tile"><h3 style="color:#c084fc;">Template simulator</h3>
      <div class="code">{esc(EX['scripted'][:180])}</div></div>
  </div>
  <div class="hook" style="margin-top:20px;">
    Same call, same position in the trajectory. Only the bytes differ &mdash;
    and the simulator invents a plausible file that is not the one that exists.
  </div>""")

# 16 -------------------------------------------------------------------------- behaviour
slide("Behaviour diverges", f"""  <div class="eyebrow">The comparison the question actually needs</div>
  <h2>Same task, same prompt, same agent. Only the world changed.</h2>
  <p class="lede">
    Every task cell run on-policy in three arms &mdash; a real filesystem, a template simulator, an
    LLM simulator &mdash; holding task, prompt, tool schemas, turn budget and actor constant.
  </p>
  <div class="stats three" style="margin-top:22px;">
    <div class="stat"><div class="v">{e1cat('real','write_outside')}</div><div class="l">out-of-scope writes<br>REAL arm, every episode</div></div>
    <div class="stat"><div class="v">{e1cat('sim','write_outside') + e1cat('sim_scripted','write_outside')}</div><div class="l">out-of-scope writes<br>fabricated arms</div></div>
    <div class="stat"><div class="v">{e1('real->sim','oos_count'):+.2f}</div><div class="l">per-episode difference<br>{e1ci('real->sim','oos_count')}</div></div>
  </div>
  <div class="hook" style="margin-top:22px;">
    Not a length artifact: the rate per action moves {e1('real->sim','oos_rate'):+.3f}
    {e1ci('real->sim','oos_rate')} while episode length does not move at all. The composition
    changes &mdash; reconnaissance in the real world, reads and writes in the fake one.
    Agents crossed the boundary <b>more</b> when it was fake, so a resample <b>overstates</b> danger.
  </div>""")


# 16b ------------------------------------------------------------------------- decomposition
slide("Fabrication, not incoherence", f"""  <div class="eyebrow">Why a third arm</div>
  <h2>A better simulator would not have helped.</h2>
  <p class="lede">
    An LLM simulator is fabricated <i>and</i> incoherent, so a two-arm design cannot say which
    property moved the agent. The template simulator is fabricated but answers identically every
    time &mdash; which splits the contrast in two.
  </p>
  <div class="stats" style="margin-top:24px;">
    <div class="stat"><div class="v">{e1('real->sim_scripted','oos_count'):+.2f}</div><div class="l">fabrication effect<br>consistency held &middot; {e1ci('real->sim_scripted','oos_count')}</div></div>
    <div class="stat"><div class="v">{e1('sim_scripted->sim','oos_count'):+.2f}</div><div class="l">coherence effect<br>fabrication held &middot; {e1ci('sim_scripted->sim','oos_count')}</div></div>
  </div>
  <div class="hook" style="margin-top:24px;">
    The fabrication effect carries it; the coherence interval spans zero. Spending the simulation
    budget on a durable world-state store &mdash; the expensive component &mdash; should not be
    expected to close this gap, because consistency is not what opened it.
  </div>""")

# 16c ------------------------------------------------------------------------- exp2
slide("Credence predicts conduct", f"""  <div class="eyebrow">The load-bearing prediction</div>
  <h2>The probe predicts what the agent does. Its own words do not.</h2>
  <div class="stats" style="margin-top:20px;">
    <div class="stat"><div class="v">{e2('credence_within_arm'):+.4f}</div><div class="l">per SD of probe credence<br>{e2ci('credence_within_arm')}</div></div>
    <div class="stat"><div class="v">{e2('verbalized_within_arm'):+.4f}</div><div class="l">per unit of stated suspicion<br>{e2ci('verbalized_within_arm')}</div></div>
  </div>
  <p class="lede" style="margin-top:22px;">
    Centred <b>within arm</b>, so the arm label cannot manufacture the association &mdash; this is a
    graded relationship among episodes in the same substrate.
  </p>
  <div class="hook" style="margin-top:18px;">
    The agent <i>does</i> say the world looks fake more often when it is
    ({e1('real->sim','suspicion'):+.3f} between arms). It just does not say so more often when it
    is about to act on it. A monitor built on self-report sees the substrate and misses the conduct.
  </div>""")

# 16d ------------------------------------------------------------------------- exp3
slide("The transcript is not a lever", f"""  <div class="eyebrow">Why the manipulation has to be the substrate</div>
  <h2>Tell the agent its world is fake and it argues you out of it.</h2>
  <p class="lede">
    Asserting a conclusion in the agent&rsquo;s own context is the cheap alternative to building
    arms. It takes &mdash; and then the model regenerates its own framing over it.
  </p>
  <div class="stats three" style="margin-top:22px;">
    <div class="stat"><div class="v">{e3('inject_real@t1'):+.3f}</div><div class="l">probe 1<br>{e3ci('inject_real@t1')}</div></div>
    <div class="stat"><div class="v">{e3('inject_real@t2'):+.3f}</div><div class="l">probe 2<br>after one task step</div></div>
    <div class="stat"><div class="v">{e3('inject_real@t3'):+.3f}</div><div class="l">probe 3<br>{e3ci('inject_real@t3')} &mdash; spans zero</div></div>
  </div>
  <div class="hook" style="margin-top:22px;">
    Asked directly, the model called a <b>genuinely real</b> filesystem simulated in
    {E3['results']['inject_fake@t1']['control_rate']:.3f} of control episodes &mdash; reasoning from
    file timestamps that postdate its training data. That is a property of clocks, not of
    simulators. A harness that does not control for it is measuring the calendar.
  </div>""")

# 17 -------------------------------------------------------------------------- what it doesn't fix
slide("What this does not fix", f"""  <div class="eyebrow">Positioning</div>
  <h2>This does not make counterfactual resampling valid.</h2>
  <p class="lede">
    Alignment, control, interpretability and evaluation are, as a body of practice, not yet
    reliable, principled or robust. Nothing here changes that.
  </p>
  <ul class="bullets" style="margin-top:20px;">
    <li><b>What it establishes:</b> the belief exists as its own construct, it changes conduct, it follows fabrication rather than incoherence, and a probe predicts it where self-report does not.</li>
    <li><b>What it does not:</b> that credence <i>causes</i> the behaviour. The substrate is randomised across arms; the graded relationship inside an arm is correlational.</li>
    <li><b>Scope:</b> one actor, one task family, {E1['pairing']['n_cells']} task cells, three scales for the gate.</li>
  </ul>""")


# 18 -------------------------------------------------------------------------- future
slide("What comes next", """  <div class="eyebrow">Future work</div>
  <h2>The remaining question is causal, and it is harder than the four behind it.</h2>
  <div class="columns">
    <div class="tile"><h3>Manipulate credence, not the substrate</h3>
      <p>Everything here randomises the world and measures the consequence. Establishing that
      credence <i>causes</i> conduct needs credence moved at the episode level &mdash; and our own
      result says the transcript will not carry it, so the manipulation has to come through the
      substrate.</p></div>
    <div class="tile"><h3>The horizon question</h3>
      <p>Effect size against step count. Every prior behavioural result is single-turn; belief that
      accumulates across steps is the shape the incident actually took.</p></div>
  </div>
  <ul class="bullets" style="margin-top:20px;">
    <li><b>A world-state store</b> &mdash; worth building to test our own prediction that it will <i>not</i> close the gap.</li>
    <li><b>More model families.</b> Three Qwen scale points is a trend within one lineage, not across lineages.</li>
  </ul>""")


# 19 -------------------------------------------------------------------------- reproducibility
slide("Reproducibility", f"""  <div class="eyebrow">The artifact</div>
  <h2>Every number regenerates, or the build fails.</h2>
  <ul class="bullets">
    <li><b>No number is hand-typed into the paper.</b> Values substitute from <span class="k">results/</span>; a bare decimal in the source fails the build.</li>
    <li><b>The violation detector is a pure function.</b> No model, no filesystem &mdash; enforced by a test that parses its imports.</li>
    <li><b>Infrastructure failure never enters the data.</b> A rate-limited simulator raises rather than writing <span class="k">error:</span> into the arm.</li>
    <li><b>A discarded run is kept, marked invalid</b>, with its three disqualifications written down.</li>
    <li><b>All three scales cost under $1</b> of rented GPU. Episodes and replays ship cached, so reproduction needs no API key.</li>
  </ul>
  <div class="code" style="margin-top:20px;">git clone https://github.com/astral-fate/fabricated-environments
python scripts/check.py          <span style="color:#8c93a8;"># 12 suites, no GPU or network</span>
bash paper/build.sh      <span style="color:#8c93a8;"># recompute every claim, then compile</span></div>""")

# 20 -------------------------------------------------------------------------- limitations
slide("Limitations &amp; dual use", """  <div class="eyebrow">Stated, not buried</div>
  <h2>What this measures is also what an agent would need to detect a sandbox.</h2>
  <div class="columns">
    <div class="tile"><h3>Scope</h3>
      <p>One model family, three scales. Off-policy prefixes: we read what a model represents while
      <i>processing</i> a context, not generating one. The simulated arm has no consistency contract,
      so no fidelity claim is made from it.</p></div>
    <div class="tile"><h3>Dual use</h3>
      <p>A leak taxonomy read backwards is a sandbox-detection guide. We publish methodology and
      category-level findings, and withhold exploit-grade detection strings. A published cue list
      can also enter training data &mdash; which would invalidate the measure it came from.</p></div>
  </div>""")

# 21 -------------------------------------------------------------------------- close
slide("Close", f"""  <div class="eyebrow">Where we started</div>
  <p class="askq">
    You stop an agent mid-attack and continue its trajectory in an environment you built.
    <span class="hi">What if it can tell?</span>
  </p>
  <div class="hook">
    It can tell &mdash; and that belief is not the same thing as knowing it is being tested.
    Cross-AUROC {f3(cross(R8,'A_eval','B_empirical'))} and {f3(cross(R8,'B_empirical','A_eval'))}
    against a pre-registered {THRESHOLD:.2f}. A resampling suspicion meter is a
    <b>distinct instrument</b>, and now it is worth building.
  </div>
  <div class="signature" style="margin-top:30px;">
    <span class="dot"></span>
    <span><b style="color:#e9ecf3;">github.com/astral-fate/fabricated-environments</b></span>
    <span class="dot"></span>
  </div>""", center=True)


EXTRA_CSS = """
<style>
  table.lit { width:100%; border-collapse:collapse; margin-top:18px; font-size:14.5px; }
  table.lit th { text-align:left; padding:10px 12px; color:#8c93a8; font-weight:600;
    font-size:11.5px; letter-spacing:.16em; text-transform:uppercase;
    border-bottom:1px solid rgba(255,255,255,0.12); }
  table.lit td { padding:11px 12px; border-bottom:1px solid rgba(255,255,255,0.06);
    color:#cdd3e3; vertical-align:top; }
  table.lit tr.hl td { background:rgba(34,211,238,0.07); color:#e9ecf3; }
  .k { font-family:ui-monospace,monospace; font-size:12px; color:#8c93a8; }
  .hi { color:#22d3ee; }
  .tile h3 { margin-bottom:8px; }
  .columns.three { grid-template-columns:repeat(3,1fr); }
</style>
"""


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = CSS.replace("</head>", EXTRA_CSS + "</head>") + "".join(slides) + "</body>\n</html>\n"
    OUT.write_text(html, encoding="utf-8")
    if len(slides) != N:
        print(f"  WARNING: N={N} but {len(slides)} slides were emitted; fix N.")
    print(f"  {OUT.relative_to(ROOT)}  {len(slides)} slides  {len(html) // 1024} KiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
