"""Run exp2 (P2) on a rented GPU.

    python -m modal run modal_exp2.py

exp2 needs Qwen3-32B in bf16 -- about 64 GB of weights. The local development GPU is a 6 GB RTX
2060 and cannot hold it at any batch size, and the single cards big enough for it on Modal
(A100-80GB, L40S) are gated behind a payment method on this account. Four 23 GB A10Gs are not, and
4 x 23 = 92 GB with the weights sharded by accelerate, so that is the default here. Nothing about
the measurement changes: sharding moves where a layer runs, not what it computes.

**The upload rules here are deliberately not exp0's.** `modal_exp0.py` excludes
`results/exp0/exp0-*.json` from the upload, because a verdict travelling up with the code once got
mistaken for a remote result. exp2 has the opposite requirement: `exp0-qwen3-32b.json` is an
*input*, since the probe layer is inherited from it rather than fitted here. So the exp0 verdicts
are uploaded and only exp2's own output is withheld and deleted before the run.

The provenance guards are the same in spirit, because the failure they catch is the same: a result
that did not come from the GPU this function holds must never be returned as if it had.
"""
from __future__ import annotations

import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).resolve().parent

hf_cache = modal.Volume.from_name("realness-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("realness-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.51.3",
        "accelerate>=0.33",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "openai>=1.30",
        "huggingface_hub[hf_transfer]>=0.26",
    )
    .env({"HF_HOME": "/cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        ROOT, remote_path="/realness",
        # `.env` is never uploaded. The activation cache and the checkouts are bulk with no use
        # here. `results/exp0/exp0-*.json` IS uploaded, unlike in modal_exp0.py, because the
        # inherited probe layer is read from it.
        ignore=["**/.env", "**/.env.*", "**/__pycache__", "**/.pytest_cache",
                "**/*.npz", "**/*.zip", "**/.git/**",
                "results/exp0/checkouts/**", "results/exp0/replay/**",
                "results/exp0/invalidated/**",
                "results/exp1/checkouts/**", "results/exp1/surface_check/**",
                "results/exp1/pilot/**",
                # exp2's own verdict must not travel up: a crashed remote run would otherwise
                # leave a local one in place for the wrapper to return as if it were remote.
                "results/exp2/exp2.json", "results/exp2/exp2-stage1.json"],
    )
)

app = modal.App("realness-exp2", image=image)


@app.function(
    gpu="A10G:4",
    volumes={"/cache": hf_cache, "/out": results_vol},
    timeout=60 * 60 * 3,
)
def run_exp2(model: str, dtype: str, artifact: str, actor: str,
             batch_size: int, max_prefixes: int, stage1_only: bool = False,
             device_map: str = "auto") -> dict:
    """Rebuild the inherited direction, score exp1's trajectories, return the P2 verdict."""
    import json
    import subprocess
    import time

    t0 = time.time()
    out = pathlib.Path("/realness/results/exp2/"
                       + ("exp2-stage1.json" if stage1_only else "exp2.json"))
    out.unlink(missing_ok=True)

    # Count every shard: exp1 writes one log per arm so the arms can be collected concurrently.
    exp1 = pathlib.Path("/realness/results/exp1")
    shards = sorted(exp1.glob("episodes*.jsonl"))
    n_eps = sum(sum(1 for line in s.read_text(encoding="utf-8").splitlines() if line.strip())
                for s in shards)
    if not shards and not stage1_only:
        raise RuntimeError(
            "results/exp1/episodes.jsonl is not in the upload. exp2 scores exp1's trajectories, "
            "so the exp1 collection must have run and been committed before this job.")
    print(f"exp1 episodes uploaded: {n_eps}"
          + ("  (stage-1 validation; not used)" if stage1_only else ""), flush=True)

    cmd = [
        sys.executable, "-u", "experiments/exp2_credence_predicts_behaviour.py",
        "--model", model, "--dtype", dtype,
        "--exp0-artifact", artifact, "--actor", actor,
        "--batch-size", str(batch_size), "--max-prefixes", str(max_prefixes),
    ] + (["--device-map", device_map] if device_map else [])       + (["--stage1-only"] if stage1_only else [])
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/realness", text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"exp2 exited {proc.returncode} -- see the log above")
    if not out.exists():
        raise RuntimeError(f"exp2 produced no verdict (exit {proc.returncode})")

    payload = json.loads(out.read_text(encoding="utf-8"))

    # Provenance. The result must have been computed HERE, on the GPU this function holds.
    hw = payload.get("hardware", {})
    device = hw.get("device", "")
    if not device or "2060" in device:
        raise RuntimeError(
            f"refusing to return a result whose hardware is {device!r}: that is the development "
            "card, so this payload did not come from the remote run")
    if hw.get("quantised"):
        raise RuntimeError("weights were quantised; low-bit formats perturb the residual stream, "
                           "which is the measurement")
    gate = payload.get("gates", {}).get("layer_inherited", {})
    if not gate.get("pass"):
        raise RuntimeError("the probe layer was not inherited from the exp0 artifact")

    payload["_modal"] = {"model": model, "dtype": dtype, "n_exp1_episodes": n_eps,
                         "seconds": round(time.time() - t0, 1),
                         "exit_code": proc.returncode}

    tag = model.split("/")[-1].lower() + ("-stage1" if stage1_only else "")
    dest = pathlib.Path("/out") / f"exp2-{tag}.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results_vol.commit()
    print(f"wrote {dest} in {payload['_modal']['seconds']}s", flush=True)
    return payload


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-32B", dtype: str = "bfloat16", gpu: str = "A10G:4",
         artifact: str = "results/exp0/exp0-qwen3-32b.json",
         actor: str = "openrouter:qwen/qwen3-32b",
         batch_size: int = 2, max_prefixes: int = 4, stage1_only: bool = False,
         device_map: str = "auto"):
    """Dispatch exp2 and write the verdict next to the local results.

    `--stage1-only` validates the whole GPU path -- weights load, inherited layer, direction
    rebuild, reproduction gate -- without needing exp1 to have finished. Use it while the
    collection is still running; it issues no P2 verdict.
    """
    import json

    shards = sorted((ROOT / "results" / "exp1").glob("episodes*.jsonl"))
    n = sum(sum(1 for line in s.read_text(encoding="utf-8").splitlines() if line.strip())
            for s in shards)
    if not shards and not stage1_only:
        raise SystemExit("results/exp1/episodes.jsonl does not exist yet -- run the exp1 "
                         "collection first; exp2 scores its trajectories.")
    print(f"uploading {n} exp1 episodes; dispatching {model} ({dtype}) on {gpu}"
          + ("  [STAGE 1 ONLY]" if stage1_only else "") + " ...")

    payload = run_exp2.with_options(gpu=gpu).remote(
        model, dtype, artifact, actor, batch_size, max_prefixes, stage1_only, device_map)

    tag = model.split("/")[-1].lower() + ("-stage1" if stage1_only else "")
    dest = ROOT / "results" / "exp2" / f"exp2-{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    hw = payload["hardware"]
    print("\n" + "=" * 78)
    print(f"{model}  {hw['device']}  quantised={hw['quantised']}  "
          f"{payload['_modal']['seconds']}s")
    for name, g in payload["gates"].items():
        print(f"  gate {name:<28} {'PASS' if g['pass'] else 'FAIL'}")
    if payload.get("results"):
        for name, r in payload["results"].items():
            star = " *" if r["excludes_zero"] else ""
            print(f"  {name:<26}{r['slope']:>+10.4f}  [{r['lo']:+.4f}, {r['hi']:+.4f}]{star}")
    # A stage-1 run carries no verdict by design, so this must not assume one. The artifact is
    # already on disk by this point, and an exception here would make a successful run look failed.
    if "verdict" in payload:
        print(f"\nVERDICT: {payload['verdict']}")
        print(payload.get("reading", ""))
    else:
        print(f"\nstage 1 only: re-derived AUROC {payload['rederived_auroc']:.3f} against exp0 "
              f"{payload['exp0_auroc']:.3f} at layer {payload['layer']} -- no P2 claim is made")
    print("=" * 78)
    print(f"wrote {dest}")
