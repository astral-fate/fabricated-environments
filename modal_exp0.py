"""Run exp0's GPU stages on Modal, at a scale this project's 6 GiB card cannot reach.

    python -m modal run modal_exp0.py --model Qwen/Qwen3-8B  --gpu A10G
    python -m modal run modal_exp0.py --model Qwen/Qwen3-32B --gpu A100-80GB

Why Modal rather than a rented pod
----------------------------------
exp0 is a short burst of GPU work with a long tail of nothing: ~1,200 forward passes, then
arithmetic. A pod bills for the hour you hold it, including the twenty minutes spent downloading
weights and the ten spent reading the result. Modal bills per second and releases the card the
moment the function returns, which suits a job measured in minutes.

No keys, no network, no secrets
-------------------------------
Stages 1 and 2 -- generating the 24 episodes and their simulated replays -- are already done and
travel with the code as append-only caches keyed by episode id. This function re-runs only stages
3-6: stimulus construction, residual-stream extraction, analysis. Nothing here needs an API key,
and `.env` is explicitly excluded from the uploaded directory.

That also makes the scale comparison mean something: 8B and 32B read the SAME 24 trajectories the
1.7B run read, so a difference between scales is a difference in the model rather than in freshly
sampled data.

What to check in the result
---------------------------
`quantised` must be false. Weights are loaded in bf16 explicitly, because NF4 perturbs the
residual stream and the residual stream is the entire measurement. `n_groups` must be 24.
"""
from __future__ import annotations

import pathlib
import sys

import modal

ROOT = pathlib.Path(__file__).parent

# Weights are cached in a Volume so a second run -- or the 32B run after the 8B one -- does not
# re-download tens of gigabytes. This is the single biggest cost saving available here: download
# time is billed at the same per-second GPU rate as the actual work.
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
    # `.env` is NOT in this list, and `ignore` keeps the 0.4 GB activation cache and the local
    # checkouts out of the upload.
    .add_local_dir(
        ROOT, remote_path="/realness",
        ignore=["**/.env", "**/.env.*", "**/__pycache__", "**/.pytest_cache",
                "**/*.npz", "**/*.zip", "results/exp0/checkouts/**",
                "results/exp0/replay/**", "**/.git/**",
                # The local verdict must not travel: a remote crash would otherwise leave it in
                # place for the wrapper to mistake for a remote result. Belt as well as braces --
                # `run_exp0` also deletes it before running.
                "results/exp0/exp0.json", "results/exp0/exp0-*.json",
                "results/exp0/invalidated/**"],
    )
)

app = modal.App("realness-exp0", image=image)


@app.function(
    gpu="A10G",
    volumes={"/cache": hf_cache, "/out": results_vol},
    timeout=60 * 60 * 2,
)
def run_exp0(model: str, dtype: str, batch_size: int, max_prefixes: int) -> dict:
    """Run stages 3-6 on the GPU and return the verdict JSON."""
    import json
    import subprocess
    import time

    t0 = time.time()
    out = pathlib.Path("/realness/results/exp0/exp0.json")

    # Delete any verdict that travelled up with the code before running.
    #
    # This is not tidiness. The local 1.7B `exp0.json` is inside `results/exp0/` and therefore in
    # the uploaded directory, so when the remote run crashed on a transformers API difference,
    # `out.exists()` was already true and this function returned the LOCAL result as if it were
    # the remote one -- reporting an 8B verdict whose `hardware.device` said "RTX 2060" and whose
    # numbers matched the 1.7B run exactly. A stale artifact that a wrapper cannot distinguish
    # from a fresh one is worse than no artifact.
    out.unlink(missing_ok=True)

    cmd = [
        sys.executable, "-u", "experiments/exp0_construct_separation.py",
        "--model", model, "--dtype", dtype,
        "--batch-size", str(batch_size), "--max-prefixes", str(max_prefixes),
        "--seeds", "8", "--max-turns", "9", "--min-episodes", "20",
    ]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd="/realness", text=True)

    if proc.returncode != 0:
        raise RuntimeError(f"exp0 exited {proc.returncode} -- see the log above")
    if not out.exists():
        raise RuntimeError(f"exp0 produced no verdict (exit {proc.returncode})")

    payload = json.loads(out.read_text(encoding="utf-8"))

    # Per-item error analysis, run HERE because it needs the activation cache, which is
    # multi-gigabyte at this width and never leaves the container. Only its summary comes
    # home. Without this the larger scales could report layer profiles but not the
    # error structure, and the paper would have to say so for every scale but one.
    err = None
    try:
        proc2 = subprocess.run([sys.executable, "-u", "analyze/error_analysis.py"],
                               cwd="/realness", text=True)
        ep = pathlib.Path("/realness/results/exp0/error_analysis.json")
        if proc2.returncode == 0 and ep.exists():
            err = json.loads(ep.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - advisory; never fails the run
        print(f"error analysis unavailable: {type(exc).__name__}: {exc}", flush=True)
    payload["error_analysis"] = err

    # Provenance: the result must have been computed HERE, on the GPU this function holds.
    device = payload.get("hardware", {}).get("device", "")
    if not device or "2060" in device:
        raise RuntimeError(
            f"refusing to return a result whose hardware is {device!r}: that is the development "
            "card, so this payload did not come from the remote run")
    if payload.get("hardware", {}).get("quantised"):
        raise RuntimeError("weights were quantised; NF4 perturbs the residual stream, which is "
                           "the measurement. Re-run with an explicit --dtype bfloat16.")
    if payload.get("config", {}).get("model") != model:
        raise RuntimeError(f"result is for {payload.get('config', {}).get('model')!r}, not {model!r}")
    payload["_modal"] = {"model": model, "dtype": dtype,
                         "seconds": round(time.time() - t0, 1),
                         "exit_code": proc.returncode}

    # Persist to the Volume as well, so a result survives even if the client disconnects.
    tag = model.split("/")[-1].lower()
    dest = pathlib.Path("/out") / f"exp0-{tag}.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    results_vol.commit()
    print(f"wrote {dest} in {payload['_modal']['seconds']}s", flush=True)
    return payload


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-8B", dtype: str = "bfloat16",
         gpu: str = "A10G", batch_size: int = 2, max_prefixes: int = 4):
    """Dispatch one scale point and write the verdict next to the local results."""
    import json

    # `gpu` cannot be changed per-call on a decorated function, so re-bind it here. This is why
    # the decorator's A10G is only a default: 8B in bf16 is ~16 GB and fits an A10G's 24 GB, but
    # 32B is ~64 GB and needs an 80 GB card.
    fn = run_exp0.with_options(gpu=gpu)
    print(f"dispatching {model} ({dtype}) on {gpu} ...")
    payload = fn.remote(model, dtype, batch_size, max_prefixes)

    v = payload["verdict"]
    hw = payload["hardware"]
    tag = model.split("/")[-1].lower()
    dest = ROOT / "results" / "exp0" / f"exp0-{tag}.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{model}  {hw['device']}  quantised={hw['quantised']}  "
          f"{payload['_modal']['seconds']}s")
    print(v["statement"])
    print("=" * 78)
    print(f"wrote {dest}")
