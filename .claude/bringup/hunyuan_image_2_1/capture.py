# SPDX-License-Identifier: Apache-2.0
"""One-pass CPU capture of HunyuanImage 2.1 (Distilled) pipeline component I/O.

Non-invasive: registers forward hooks on the live component instances and wraps
vae.decode. Does NOT mutate any installed-package source. Dumps a per-component
I/O spec + exact parameter counts to io_spec.json / capture.log.
"""
import json
import logging
import os
import sys
import traceback

import torch

REPO = "hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(OUT_DIR, "capture.log")
SPEC = os.path.join(OUT_DIR, "io_spec.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("capture")

spec = {"components": {}, "denoise_steps_observed": 0, "errors": []}


def _describe(obj, depth=0):
    """Return a JSON-able shape/dtype summary of a (possibly nested) value."""
    if isinstance(obj, torch.Tensor):
        return {"kind": "tensor", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (list, tuple)):
        if depth > 3:
            return {"kind": "seq", "len": len(obj)}
        return {"kind": "seq", "items": [_describe(x, depth + 1) for x in obj]}
    if isinstance(obj, dict):
        if depth > 3:
            return {"kind": "dict", "keys": list(obj.keys())}
        return {"kind": "dict", "fields": {str(k): _describe(v, depth + 1) for k, v in obj.items()}}
    if obj is None:
        return {"kind": "none"}
    # objects with a .sample / .last_hidden_state attr (diffusers/transformers outputs)
    for attr in ("sample", "last_hidden_state"):
        if hasattr(obj, attr):
            return {"kind": type(obj).__name__, attr: _describe(getattr(obj, attr), depth + 1)}
    return {"kind": type(obj).__name__, "repr": repr(obj)[:80]}


def make_pre_hook(name, store):
    def hook(module, args, kwargs):
        if store["inputs"] is None:  # capture first call only
            store["inputs"] = {
                "args": [_describe(a) for a in args],
                "kwargs": {k: _describe(v) for k, v in kwargs.items()},
            }
            log.info("FWD_IN  %-16s args=%s kwargs=%s", name,
                     store["inputs"]["args"], list(kwargs.keys()))
        store["calls"] += 1
    return hook


def make_post_hook(name, store):
    def hook(module, args, kwargs, output):
        if store["outputs"] is None:
            store["outputs"] = _describe(output)
            log.info("FWD_OUT %-16s out=%s", name, store["outputs"])
    return hook


def main():
    log.info("Loading pipeline %s on CPU (bfloat16)...", REPO)
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    pipe.to("cpu")
    log.info("Pipeline class: %s", type(pipe).__name__)

    # ---- exact param counts per component ----
    for name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        comp = getattr(pipe, name, None)
        store = {"inputs": None, "outputs": None, "calls": 0, "params": None,
                 "class": type(comp).__name__ if comp is not None else None}
        if comp is not None and hasattr(comp, "parameters"):
            store["params"] = sum(p.numel() for p in comp.parameters())
            # also record inner language_model params (Qwen vision tower is dropped)
            inner = getattr(comp, "language_model", None)
            if inner is not None and hasattr(inner, "parameters"):
                store["language_model_params"] = sum(p.numel() for p in inner.parameters())
                store["language_model_class"] = type(inner).__name__
        spec["components"][name] = store

    # ---- register hooks ----
    handles = []
    for name in ("text_encoder", "text_encoder_2", "transformer"):
        comp = getattr(pipe, name, None)
        if comp is None:
            continue
        store = spec["components"][name]
        handles.append(comp.register_forward_pre_hook(make_pre_hook(name, store), with_kwargs=True))
        handles.append(comp.register_forward_hook(make_post_hook(name, store), with_kwargs=True))
        # also hook the inner language_model (what the loader returns for text_encoder)
        inner = getattr(comp, "language_model", None)
        if inner is not None:
            sub = {"inputs": None, "outputs": None, "calls": 0,
                   "class": type(inner).__name__}
            spec["components"][name + ".language_model"] = sub
            handles.append(inner.register_forward_pre_hook(make_pre_hook(name + ".language_model", sub), with_kwargs=True))
            handles.append(inner.register_forward_hook(make_post_hook(name + ".language_model", sub), with_kwargs=True))

    # transformer calls == denoise steps (distilled => no CFG doubling, but count anyway)
    # ---- wrap vae.decode (not a forward) ----
    vae = pipe.vae
    vstore = spec["components"]["vae"]
    _orig_decode = vae.decode

    def decode_wrap(*args, **kwargs):
        if vstore["inputs"] is None:
            vstore["inputs"] = {
                "args": [_describe(a) for a in args],
                "kwargs": {k: _describe(v) for k, v in kwargs.items()},
            }
            log.info("DECODE_IN  vae args=%s kwargs=%s", vstore["inputs"]["args"], list(kwargs.keys()))
        out = _orig_decode(*args, **kwargs)
        if vstore["outputs"] is None:
            vstore["outputs"] = _describe(out)
            log.info("DECODE_OUT vae out=%s", vstore["outputs"])
        vstore["calls"] += 1
        return out

    vae.decode = decode_wrap

    # ---- run one tiny pass ----
    log.info("Running pipeline: steps=2, height=64, width=64")
    try:
        with torch.no_grad():
            pipe(
                "a small red cube on a wooden table",
                num_inference_steps=2,
                height=64,
                width=64,
                output_type="latent",  # skip full VAE/image post if heavy; we wrapped decode separately
            )
    except Exception as e:
        log.warning("height=64 pass failed (%s); retrying at 512x512", repr(e)[:160])
        spec["errors"].append("64px: " + repr(e)[:300])
        try:
            with torch.no_grad():
                pipe(
                    "a small red cube on a wooden table",
                    num_inference_steps=2,
                    height=512,
                    width=512,
                )
        except Exception as e2:
            log.error("512px pass also failed: %s", traceback.format_exc())
            spec["errors"].append("512px: " + repr(e2)[:300])

    spec["denoise_steps_observed"] = spec["components"]["transformer"]["calls"]

    for h in handles:
        h.remove()
    vae.decode = _orig_decode

    with open(SPEC, "w") as f:
        json.dump(spec, f, indent=2)
    log.info("Wrote %s", SPEC)
    log.info("DONE. denoise_steps=%d", spec["denoise_steps_observed"])
    for n, s in spec["components"].items():
        log.info("  %-26s params=%s calls=%s", n, s.get("params"), s.get("calls"))


if __name__ == "__main__":
    main()
