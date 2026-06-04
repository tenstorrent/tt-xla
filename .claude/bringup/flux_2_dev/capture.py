"""One-pass CPU capture of FLUX.2-dev per-component I/O shapes/dtypes.

Runtime monkey-patch only -- does NOT mutate installed package source.
Writes structured JSONL records directly to a file (bypasses the logging
module, which other libs reconfigure on import and silently swallow).
"""
import functools
import inspect
import json

import torch
from diffusers import (
    Flux2Pipeline,
    Flux2Transformer2DModel,
    AutoencoderKLFlux2,
)
from transformers import Mistral3ForConditionalGeneration

REPO = "black-forest-labs/FLUX.2-dev"
OUT = "/proj_sw/user_dev/ctr-akannan/2_jun_yyz/tt-xla/.claude/bringup/flux_2_dev/io_spec_raw.jsonl"
_FH = open(OUT, "w")


def emit(rec):
    _FH.write(json.dumps(rec, default=str) + "\n")
    _FH.flush()


def _describe(v):
    if torch.is_tensor(v):
        return {"shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return [_describe(x) for x in v]
    if isinstance(v, dict):
        return {k: _describe(x) for k, x in v.items()}
    if hasattr(v, "sample") and torch.is_tensor(getattr(v, "sample")):
        return {"_obj": type(v).__name__, "sample": _describe(v.sample)}
    if hasattr(v, "last_hidden_state") and torch.is_tensor(getattr(v, "last_hidden_state")):
        return {"_obj": type(v).__name__,
                "last_hidden_state": _describe(v.last_hidden_state)}
    if hasattr(v, "logits") and torch.is_tensor(getattr(v, "logits")):
        return {"_obj": type(v).__name__, "logits": _describe(v.logits)}
    return {"_repr": type(v).__name__}


_STEP = {"transformer": 0}


def wrap(cls, method_name, tag, count_key=None):
    orig = getattr(cls, method_name)
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        if count_key:
            _STEP[count_key] += 1
        try:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            ins = {k: _describe(v) for k, v in bound.arguments.items() if k != "self"}
        except Exception as e:
            ins = {"_bind_err": str(e)}
        emit({"event": "FWD_IN", "tag": tag, "class": type(self).__name__,
              "call": _STEP.get(count_key) if count_key else None, "inputs": ins})
        out = orig(self, *args, **kwargs)
        emit({"event": "FWD_OUT", "tag": tag, "class": type(self).__name__,
              "output": _describe(out)})
        return out

    setattr(cls, method_name, wrapper)


def main():
    wrap(Flux2Transformer2DModel, "forward", "transformer", count_key="transformer")
    wrap(AutoencoderKLFlux2, "decode", "vae_decode")
    wrap(AutoencoderKLFlux2, "encode", "vae_encode")
    wrap(Mistral3ForConditionalGeneration, "forward", "text_encoder")

    emit({"event": "LOADING", "repo": REPO})
    pipe = Flux2Pipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    pipe.to("cpu")
    emit({"event": "LOADED",
          "components": [n for n in pipe.components if pipe.components[n] is not None]})

    gen = torch.Generator(device="cpu").manual_seed(0)
    out = pipe(
        prompt="a small red cube on a wooden table",
        num_inference_steps=2,
        height=64,
        width=64,
        guidance_scale=4.0,
        generator=gen,
        output_type="pt",
    )
    emit({"event": "PASS_DONE", "denoise_steps_observed": _STEP["transformer"],
          "pipe_out": _describe(out)})
    emit({"event": "CAPTURE_OK"})
    _FH.close()
    print("CAPTURE_OK steps=%d -> %s" % (_STEP["transformer"], OUT))


if __name__ == "__main__":
    main()
