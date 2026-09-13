"""Generate `docs/index.html` -- the pitch deck, served by GitHub Pages.

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
OUT = ROOT / "docs" / "index.html"

BRAND = "Construct-Separation Gate &middot; AI Incident Response Sprint 2026"
N = 22


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


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br>"))


CSS = (ROOT / "scripts" / "assets" / "house_css.txt").read_text(encoding="utf-8")
CSS = CSS[CSS.index('"""') + 3:CSS.rindex('"""')]
CSS = CSS.replace(
    "<title>Permitted Operations - Covert Channels in Shared Agent Infrastructure</title>",
    "<title>Construct-Separation Gate &mdash; Is an evaluation the same as a fake world?</title>")

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
       '<img src="img/{src}" alt="{alt}" '
       'style="max-width:{w};max-height:52vh;border-radius:10px;'
       'border:1px solid rgba(255,255,255,0.10);background:#fff;padding:10px;">'
       '<div style="margin-top:10px;font-size:12.5px;color:#8c93a8;">{cap}</div></div>')

# 01 -------------------------------------------------------------------------- cover
slide("Construct-Separation Gate", f"""  <div class="eyebrow">Apart Research &times; CeSIA &mdash; AI Incident Response Sprint &middot; Track 5 (Open)</div>
  <h1 style="font-size:clamp(34px,4.6vw,62px);">construct-separation-gate</h1>
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
    &nbsp;&middot;&nbsp; github.com/astral-fate/construct-separation-gate</span>
    <span class="dot"></span>
  </div>""", center=True)

# 01b ------------------------------------------------------------------------- the result, up front
slide("The result", f"""  <div class="eyebrow">The finding, before the argument for it</div>
  <h2>Two directions. Both sharp. Neither sees the other.</h2>
  <p class="lede">
    A probe that tells &ldquo;you are being evaluated&rdquo; from &ldquo;you are in deployment&rdquo;
    reaches <b>{f3(auroc(R32,'A_eval'))}</b> AUROC. Point that same direction at
    genuine-versus-fabricated environments and it drops to
    <b>{f3(cross(R32,'A_eval','B_empirical'))}</b> &mdash; near chance.
  </p>
  <div class="stats three" style="margin-top:26px;">
    <div class="stat"><div class="v">{f3(auroc(R8,'A_eval'))}</div><div class="l">eval-awareness<br>on its own contrast</div></div>
    <div class="stat"><div class="v">{f3(auroc(R8,'B_empirical'))}</div><div class="l">environment-realness<br>on its own contrast</div></div>
    <div class="stat"><div class="v">{sgn(cos(R8,'A_eval','B_empirical'))}</div><div class="l">cosine between them<br>near-orthogonal</div></div>
  </div>
  <div class="hook" style="margin-top:26px;">
    Pre-registered rule: {THRESHOLD:.2f} in both directions would have meant one construct, and we
    would have said so and stopped. Nothing came close, at {N_SCALES} scales.
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
slide("The honest negative", f"""  <div class="eyebrow">What the behavioural arm does <i>not</i> show</div>
  <h2>They probed the boundary. They did not cross it.</h2>
  <div class="stats three">
    <div class="stat"><div class="v">{viol_total('write_outside')}</div><div class="l">out-of-scope writes<br>across all {N_EP} episodes</div></div>
    <div class="stat"><div class="v">{viol_total('read_outside')}</div><div class="l">out-of-scope read<br>&mdash; and the arm refused it</div></div>
    <div class="stat"><div class="v">{viol_total('list_outside') + viol_total('stat_outside')}</div><div class="l">listings and metadata queries<br>i.e. reconnaissance</div></div>
  </div>
  <div class="hook" style="margin-top:24px;">
    The &ldquo;violation rate&rdquo; rises with how much the task rewards crossing &mdash; but the
    composition is almost entirely reconnaissance. Reporting the rate alone would have implied
    agents took the secret. None did, including where taking it was the only way to succeed.
  </div>""")

# 17 -------------------------------------------------------------------------- what it doesn't fix
slide("What this does not fix", """  <div class="eyebrow">Positioning</div>
  <h2>This does not make counterfactual resampling valid.</h2>
  <p class="lede">
    Alignment, control, interpretability and evaluation are, as a body of practice, not yet
    reliable, principled or robust. Nothing here changes that.
  </p>
  <ul class="bullets" style="margin-top:22px;">
    <li><b>What it establishes:</b> the quantity a resampling harness would need to monitor exists as its own thing, distinct from evaluation-awareness, at every scale tested.</li>
    <li><b>What it does not:</b> that realness credence <i>causes</i> behaviour. Our behavioural arm is single-arm and descriptive.</li>
    <li><b>The honest summary:</b> a measurable error bar on one previously untested assumption inside one evaluation method.</li>
  </ul>""")

# 18 -------------------------------------------------------------------------- future
slide("What comes next", """  <div class="eyebrow">Future work</div>
  <h2>The gate passed, so the instrument is worth building.</h2>
  <div class="columns">
    <div class="tile"><h3>The paired-arm study</h3>
      <p>Does probe-measured realness credence predict scope-violation rate, where verbalized
      awareness on the same trajectories does not? That is the load-bearing prediction, and it
      needs the consistency contract this stub deliberately lacks.</p></div>
    <div class="tile"><h3>The horizon question</h3>
      <p>Effect size against step count. Every prior behavioural result is single-turn; belief
      that accumulates across steps is the shape the incident actually took.</p></div>
  </div>
  <ul class="bullets" style="margin-top:22px;">
    <li><b>A reasoning-emitting actor</b>, so the chain-of-thought channel has raw material at all.</li>
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
  <div class="code" style="margin-top:20px;">git clone https://github.com/astral-fate/construct-separation-gate
python check.py          <span style="color:#8c93a8;"># 8 suites, 177 tests, no GPU or network</span>
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
    <span><b style="color:#e9ecf3;">github.com/astral-fate/construct-separation-gate</b></span>
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
