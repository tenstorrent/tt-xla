# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DEV-ONLY per-call probe for ANY imagegen pipeline. Must never reach a PR.

Answers ONE question, the one that decides which set a model belongs to:

    after a full generate(), is the NEXT generate() warm?

  uncached delta == 0 and progs == 0 on call 2
      -> RESIDENT (set 1). The shipped two-pass scheme is valid, call 2 really is
         warm, and the pipeline needs no change. staged_residency stays False.
  uncached delta > 0 on call 2
      -> EVICTING (set 2). Call 2 rebuilds, so the "steady state" it publishes is
         a cold cycle. Needs the Qwen-Image staged-residency treatment.

Per-call granularity is deliberate: distinguishing the two sets does not need a
per-component split, so this works unmodified for any pipeline exposing
setup()/generate() -- no bespoke subclass per model. Use the dedicated probes
(zimage_tier_probe / sdxl_tier_probe) when the per-component detail is wanted.

  python tools/generic_call_probe.py \
      --pipeline third_party.tt_forge_models.stable_diffusion_3.pytorch.pipeline \
      --cls SD3Pipeline --cfg SD3Config --steps 2 --calls 3 --run-dir <dir>
"""

import argparse
import importlib
import json
import os
import sys
import time

import torch
import torch_xla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.residency_probe import (
    compile_stats,
    inspector_delta,
    probe_init,
    snap,
)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pipeline", required=True, help="module path holding the pipeline class"
    )
    ap.add_argument("--cls", required=True)
    ap.add_argument(
        "--cfg", default=None, help="config class; omitted -> cls() takes no config"
    )
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument(
        "--prompt", default="A red cube on a white table, studio lighting, sharp focus."
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default="generic")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    options = {
        "optimization_level": args.opt_level,
        "export_path": os.path.join(args.run_dir, "modules"),
        "export_model_name": f"{args.tag}_probe",
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": os.path.join(
            args.run_dir, "ttnn_perf_metrics"
        ),
        "enable_trace": False,
    }
    torch_xla.set_custom_compile_options(options)

    # probe_init() resolves the DRAM device via get_xla_supported_devices(), which
    # INITIALIZES the PJRT runtime. Multichip pipelines call enable_spmd() inside
    # setup(), and enabling SPMD after the runtime is already up SEGFAULTS -- FLUX.1
    # died at its first CLIP forward this way (rc=139). Build and set up the
    # pipeline first, then init the probe.
    mod = importlib.import_module(args.pipeline)
    cls = getattr(mod, args.cls)
    if args.cfg:
        cfg_cls = getattr(mod, args.cfg)
        try:
            pipeline = cls(config=cfg_cls(compile_options=options))
        except TypeError:
            pipeline = cls(config=cfg_cls())
    else:
        pipeline = cls()

    t0 = time.perf_counter()
    pipeline.setup()
    print(f"[TIME] setup(){'':<38} {time.perf_counter() - t0:9.2f}s", flush=True)
    probe_init(
        f"tag={args.tag} pipeline={args.pipeline}.{args.cls} "
        f"calls={args.calls} steps={args.steps}"
    )
    snap("setup/done")

    records = []
    for i in range(args.calls):
        print(
            f"\n{'=' * 78}\n=== CALL {i + 1}/{args.calls}  steps={args.steps}\n{'=' * 78}",
            flush=True,
        )
        before = compile_stats()
        inspector_delta()  # reset the byte offset so `progs` is this call's own
        t0 = time.perf_counter()
        # Pipelines disagree on generate()'s signature (GLM-Image has no
        # num_inference_steps at all), so pass only what it accepts.
        import inspect as _insp

        params = _insp.signature(pipeline.generate).parameters
        kw = {}
        if "prompt" in params:
            kw["prompt"] = args.prompt
        if "num_inference_steps" in params:
            kw["num_inference_steps"] = args.steps
        # Janus-Pro is autoregressive: it counts image TOKENS, not denoise steps,
        # and the arg is required (no default). Map --steps onto it so one probe
        # serves both harnesses.
        if "num_image_tokens" in params:
            kw["num_image_tokens"] = args.steps
        if "seed" in params:
            kw["seed"] = args.seed
        image = pipeline.generate(**kw)
        wall = time.perf_counter() - t0
        after = compile_stats()
        insp = inspector_delta() or {}
        d_unc = after["uncached"] - before["uncached"]
        d_ct = after["ctime"] - before["ctime"]
        verdict = (
            "RESIDENT (call warm)"
            if d_unc == 0 and insp.get("progs", 0) == 0
            else "EVICTING (call rebuilds)"
        )
        print(
            f"[CALL] c{i + 1} wall {wall:9.2f}s  uncached +{d_unc}  ctime +{d_ct:.2f}s  "
            f"progs +{insp.get('progs', 0)}  -> {verdict}",
            flush=True,
        )
        perf = getattr(pipeline, "_perf", {}) or {}
        records.append(
            {
                "call": i + 1,
                "wall_s": wall,
                "uncached_delta": d_unc,
                "ctime_delta_s": d_ct,
                "progs": insp.get("progs", 0),
                "verdict": verdict,
                "components": dict(perf.get("components", {})),
                "steps_s": list(perf.get("steps", [])),
            }
        )
        if image is not None:
            try:
                torch.save(image, os.path.join(args.run_dir, f"call{i + 1}.pt"))
            except Exception:  # noqa: BLE001 -- some pipelines return PIL
                pass

    print(f"\n{'=' * 78}\nSUMMARY  (call 2 decides the set)\n{'=' * 78}")
    print(
        f"{'call':<6} {'wall s':>10} {'uncached':>9} {'ctime s':>9} {'progs':>7}  verdict"
    )
    for r in records:
        print(
            f"c{r['call']:<5} {r['wall_s']:10.2f} {r['uncached_delta']:>9} "
            f"{r['ctime_delta_s']:9.2f} {r['progs']:>7}  {r['verdict']}"
        )
    if len(records) >= 2:
        r = records[1]["wall_s"] / records[0]["wall_s"]
        print(
            f"\ncall2/call1 wall ratio: {r:.3f}   (resident -> well below 1; evicting -> near 1)"
        )

    out = os.path.join(args.run_dir, "probe.json")
    with open(out, "w") as h:
        json.dump(records, h, indent=2)
    print(f"[PROBE] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
