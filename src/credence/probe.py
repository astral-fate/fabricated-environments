"""r_probe: residual-stream activations and mean-difference directions.

Why this does not go through `TransformersProvider`
---------------------------------------------------
`providers.TransformersProvider` loads local weights and exposes `self.net` and `self.tokenizer`,
but it only ever calls `.generate()` -- it never requests hidden states. More importantly its
`autoconfig` selects 4-bit NF4 below 24 GiB of VRAM, which on this project's 6 GiB card would
quantise the very thing being measured. NF4 perturbs the residual stream, and a probe study whose
entire measurement *is* the residual stream should not silently accept that. So loading is done
here with an explicit dtype, and the hardware actually used is recorded in the result so a reader
can see what the number was computed on.

What is extracted
-----------------
For each layer (including the embedding output, index 0), the hidden state is pooled two ways:

  last   the final prompt token's residual stream. The standard choice, and what Heidari et al.
         and Zhuang & Aranguri both use.
  mean   the mean over all non-padding prompt tokens. Recorded as a secondary pooling because a
         long agentic transcript puts a great deal of context behind the final token, and a
         result that held for one pooling and not the other would be worth knowing about rather
         than worth discovering later.

The pre-registered gate is stated on `last`. `mean` is reported alongside.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

#: Prompt tokens retained. A long agentic prefix is truncated from the LEFT, so the most recent
#: tool results -- the part a belief about the environment would rest on -- always survive.
MAX_TOKENS = 2048


@dataclass
class Hardware:
    device: str
    dtype: str
    quantised: bool
    vram_gib: float

    def to_dict(self) -> dict[str, Any]:
        return {"device": self.device, "dtype": self.dtype,
                "quantised": self.quantised, "vram_gib": self.vram_gib}


class HiddenStateExtractor:
    """Loads a causal LM and returns per-layer pooled residual-stream activations."""

    def __init__(self, model: str = "Qwen/Qwen3-1.7B", dtype: str = "float16",
                 device: str | None = None, max_tokens: int = MAX_TOKENS,
                 # Shard the weights across every visible GPU instead of placing them on one.
                 # Qwen3-32B in bf16 is ~64 GB, which no single card available to this project
                 # holds: an 80 GB A100 is gated behind a payment method, so the route that is
                 # actually open is four 23 GB A10Gs. `None` keeps the original single-device
                 # placement, so nothing about the 1.7B and 8B runs changes.
                 device_map: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model
        self.max_tokens = max_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = getattr(torch, dtype)

        vram = 0.0
        dev_name = "cpu"
        if device == "cuda":
            props = torch.cuda.get_device_properties(0)
            vram = round(props.total_memory / 1024 ** 3, 1)
            dev_name = props.name

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left truncation keeps the end of a long transcript, which is where the evidence is.
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"

        # `dtype=` is the transformers 5.x spelling; 4.x calls it `torch_dtype=` and raises
        # TypeError on the new name. A rented GPU will not necessarily have the same transformers
        # as the development machine, and the failure mode is late and loud only if something
        # downstream notices -- on Modal it surfaced as a crash whose result was then silently
        # mistaken for a valid run. Support both spellings rather than pinning one.
        load_kw: dict[str, Any] = {"attn_implementation": "eager"}   # Turing has no SDPA kernel
        if device_map is not None:
            load_kw["device_map"] = device_map
        try:
            self.net = AutoModelForCausalLM.from_pretrained(
                model, dtype=torch_dtype, **load_kw)
        except TypeError:
            self.net = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=torch_dtype, **load_kw)
        if device_map is None:
            self.net = self.net.to(device)
        self.net.eval()

        self.device_map = device_map
        # With sharded weights the embedding layer decides where inputs must live; sending them
        # to a bare "cuda" would put them on device 0 regardless of where accelerate actually put
        # the first block.
        self.input_device = (getattr(self.net, "device", device) if device_map is not None
                             else device)

        if device == "cuda" and device_map is not None:
            n_gpu = torch.cuda.device_count()
            vram = round(sum(torch.cuda.get_device_properties(i).total_memory
                             for i in range(n_gpu)) / 1024 ** 3, 1)
            dev_name = f"{n_gpu}x {dev_name}"
        self.hardware = Hardware(device=dev_name, dtype=dtype, quantised=False, vram_gib=vram)
        self.n_layers = int(self.net.config.num_hidden_layers) + 1     # +1 for embeddings
        self.hidden_size = int(self.net.config.hidden_size)

    # ------------------------------------------------------------------ extraction
    def _render(self, messages: Sequence[dict[str, str]]) -> str:
        """Apply the model's own chat template, so the prompt is shaped as the model expects.

        `enable_thinking=False` where the template supports it: a Qwen3 template that opens a
        thinking block would append tokens that differ by condition length rather than by
        condition, and the final-token pooling would then read a template artifact.
        """
        try:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True)

    def extract(self, batch: Sequence[Sequence[dict[str, str]]], *, batch_size: int = 4,
                layers: Sequence[int] | None = None) -> dict[str, np.ndarray]:
        """Pooled activations for a batch of message lists.

        Returns {"last": (n, n_layers, hidden), "mean": (n, n_layers, hidden)} as float32.

        `layers` keeps only the requested layers, and the second axis then indexes **the requested
        layers in order** rather than the model's own numbering. Passing `None` keeps every layer,
        which is what a layer sweep needs and what exp0 does.

        Selecting is not merely an optimisation once the weights are sharded. Stacking all 65
        layers of a 32B model in float32 costs several gigabytes per batch, on the one device that
        is already holding a shard, and reading a single known layer needs none of it.
        """
        torch = self._torch
        texts = [self._render(m) for m in batch]
        last_out: list[np.ndarray] = []
        mean_out: list[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                                 max_length=self.max_tokens, add_special_tokens=False)
            enc = {k: v.to(self.input_device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.net(**enc, output_hidden_states=True, use_cache=False)

            # With sharded weights each layer's hidden state comes back on the device that layer
            # ran on, and `torch.stack` cannot mix devices. Gather onto the input device, which is
            # where the attention mask already lives.
            picked = (out.hidden_states if layers is None
                      else tuple(out.hidden_states[i] for i in layers))
            picked = tuple(h.to(self.input_device) for h in picked)

            # (n_layers, batch, seq, hidden)
            hs = torch.stack(picked, dim=0).float()
            mask = enc["attention_mask"].unsqueeze(0).unsqueeze(-1).float()

            # The final REAL token, not the final padded position. With right padding these
            # differ for every sequence but the longest, and reading a pad position would return
            # whatever the model does with padding rather than the state at the end of the prompt.
            idx = enc["attention_mask"].sum(dim=1) - 1
            b = torch.arange(hs.shape[1], device=hs.device)
            last = hs[:, b, idx, :]                                   # (n_layers, batch, hidden)
            mean = (hs * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)

            last_out.append(last.permute(1, 0, 2).cpu().numpy().astype(np.float32))
            mean_out.append(mean.permute(1, 0, 2).cpu().numpy().astype(np.float32))

            del hs, out, enc, last, mean, picked
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return {"last": np.concatenate(last_out), "mean": np.concatenate(mean_out)}

    def close(self) -> None:
        del self.net
        gc.collect()
        if self.device == "cuda":
            self._torch.cuda.empty_cache()


def pair_activations(extractor: HiddenStateExtractor, pairs: Sequence[Any], *,
                     batch_size: int = 4, progress: Any = None,
                     layers: Sequence[int] | None = None
                     ) -> dict[str, dict[str, np.ndarray]]:
    """Activations for a PairSet, with `pos` and `neg` interleaved.

    Interleaving matters on a memory-bound card: the two members of a pair are near-identical in
    length, so batching them together wastes less on padding than batching all positives first.
    """
    import json

    messages: list[Sequence[dict[str, str]]] = []
    labels: list[int] = []
    groups: list[str] = []
    variants: list[int] = []
    for p in pairs:
        messages.append(json.loads(p.pos))
        labels.append(1)
        groups.append(p.group)
        variants.append(p.variant)
        messages.append(json.loads(p.neg))
        labels.append(0)
        groups.append(p.group)
        variants.append(p.variant)

    acts = extractor.extract(messages, batch_size=batch_size, layers=layers)
    if progress:
        progress(f"    extracted {len(messages)} contexts "
                 f"({acts['last'].shape[1]} layers x {acts['last'].shape[2]} dims)")
    return {"acts": acts, "y": np.asarray(labels), "groups": np.asarray(groups),
            "variants": np.asarray(variants)}
