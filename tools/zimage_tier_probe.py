# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DEV-ONLY probe: does Z-Image's benchmark warmup pass actually warm anything?

Must never reach a PR. See tools/residency_probe.py for the three signals.

THE CLAIM UNDER TEST
--------------------
tests/benchmark/benchmarks/imagegen_benchmark.py runs a warmup
``generate_fn(prompt, 1)`` (:104) then a steady ``generate_fn(prompt, N)``
(:111) and publishes only the steady pass. Z-Image destroys every component at
the end of its stage (``del compiled, module; gc.collect(); torch_xla.sync()``),
and eviction discards the compiled graph -- the executable pins the weight
buffers. So the steady pass rebuilds everything and its published numbers are
cold.

THE EXPERIMENT
--------------
Three generate() calls in ONE process (tier 3 exists only within a residency,
and the counters are process-scoped):

    C1  1 step   mirrors the driver's warmup pass, verbatim
    C2  N steps  mirrors the driver's steady pass, shrunk
    C3  N steps  nothing in the driver does this -- the PLATEAU proof

Three calls, not two, because the on-disk kernel cache may already have been
orphaned by a HAL/compiler-hash bump. If it is live: C1/C2/C3 are all tier 2 and
the plateau is proven three ways. If orphaned: C1 is tier 1 and C2/C3 tier 2,
which hands us the tier1->tier2 ratio for free AND still proves the plateau. The
run self-detects which world it is in from C1's [TIER] verdict.

Three steps, because step 1 carries the transformer build and steps 2-3 give a
variance estimate rather than a single point. Contamination at the shipped N=50
is then computed analytically, not measured.

FREE TIER-3 SAMPLE
------------------
_encode() is called twice inside the text-encoder residency -- once for PROMPT,
once for NEGATIVE_PROMPT (the empty string). tokenize_prompt pads to
max_length=512 unconditionally, so both present identical device shapes and the
second is a genuine tier-3 forward. The shipped pipeline folds both into one
timer (pipeline.py:191). Splitting it costs no hardware time at all.

WHAT WOULD FALSIFY THE CLAIM
----------------------------
  encode #1 shows uncached +0 / TIER3     nothing was rebuilt; claim collapses
  encode #2 shows uncached >0             the two encodes do not share a graph
  C2/C1 <= 0.2                            the warmup really did warm it
  C3 << C2                                warmth accrues; use a 3-pass scheme
  uncached +0 and TIER3 everywhere while  the cost is WEIGHT UPLOAD, not compile
    wall time is still hundreds of s      -- the whole diagnosis is wrong
                                          (this is why [TIME] upload is logged)
"""

import argparse
import gc
import inspect
import json
import os
import shutil
import sys
import time

import torch
import torch_xla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dev-only import, kept out of module scope in production code but fine here:
# this file IS the dev tool.
from tools.residency_probe import probe_init, snap, snap_module, stage_timer

from third_party.tt_forge_models.z_image.pytorch.pipeline import (  # noqa: E402
    DTYPE,
    GUIDANCE_SCALE,
    LATENT_CHANNELS,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    VAE_SCALE_FACTOR,
    TextEncoderWrapper,
    TransformerWrapper,
    VaeDecodeWrapper,
    ZImageConfig,
    ZImageTTPipeline,
    calculate_shift,
    load_text_encoder,
    load_transformer,
    load_vae,
)

MIN_FREE_GIB = 20.0


def use_local_snapshot():
    """Point every loader at the local snapshot DIRECTORY instead of the repo id.

    Why: two independent hub calls make a repo-id run fragile, and BOTH were hit.
      - AutoTokenizer.from_pretrained(REPO_ID, subfolder="tokenizer") resolves a
        PreTrainedConfig first, but Z-Image has no tokenizer/config.json upstream.
        Online that is a 404 it shrugs off; offline it is a hard error.
      - load_transformer -> diffusers calls hf_api.model_info() to enumerate
        variant files, which raises OfflineModeIsEnabled offline and died on
        "[Errno -3] Temporary failure in name resolution" online, 4 minutes into
        a run.
    Loading from a local directory takes neither path: no config lookup, no
    model_info, no DNS. Verified all five loaders OK with HF_HUB_OFFLINE=1.

    This changes only WHERE the same files are read from. Same weights, same
    config, same graphs -- it cannot affect any timing being measured. Host-load
    time is page-cache dependent anyway (measured 16.10s cold vs 0.57s warm).
    """
    import glob

    from third_party.tt_forge_models.z_image.pytorch import pipeline as _pipe
    from third_party.tt_forge_models.z_image.pytorch.src import model_utils as _mu

    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    pattern = os.path.join(
        hf_home, "hub", "models--Tongyi-MAI--Z-Image", "snapshots", "*"
    )
    snaps = sorted(glob.glob(pattern))
    if not snaps:
        raise SystemExit(f"[PREFLIGHT] FAIL no Z-Image snapshot under {pattern}")
    snap = snaps[-1]
    # model_utils resolves REPO_ID at call time; ZImageConfig reads the pipeline
    # module's global in its __init__ body. Patch both.
    _mu.REPO_ID = snap
    _pipe.REPO_ID = snap
    print(f"[PROBE-INIT] local_snapshot={snap}", flush=True)
    return snap


class ProbeZImagePipeline(ZImageTTPipeline):
    """ZImageTTPipeline with generate() instrumented.

    The override is a faithful copy of the shipped generate() plus snaps, so
    what is measured is provably the shipped code path. The shared pipeline is
    NOT modified.

    Results accumulate in ``self._probe`` (append-only) rather than ``_perf``,
    which the shipped code clobbers at the top of every call (pipeline.py:171).
    """

    def __init__(self, config, *, stop_after=None, vae_warm_iters=0):
        super().__init__(config)
        self._probe = []
        self._stop_after = stop_after
        self._vae_warm_iters = vae_warm_iters
        self._call = 0

    def _rec(self, **kw):
        kw["call"] = self._call
        self._probe.append(kw)

    def generate(
        self, prompt=PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, seed=SEED
    ):
        self._call += 1
        c = self._call
        do_cfg = GUIDANCE_SCALE > 0
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "transformer_step",
            "total": None,
        }
        t_total_start = time.perf_counter()
        snap(f"c{c}/generate/entry")

        with torch.no_grad():
            # ---- Text encoder (Qwen3) ------------------------------------
            t = time.perf_counter()
            text_encoder = TextEncoderWrapper(load_text_encoder(DTYPE)).eval()
            host_load = time.perf_counter() - t
            print(
                f"[TIME] c{c} text_encoder host load (from_pretrained)   {host_load:9.2f}s",
                flush=True,
            )
            snap_module(f"c{c}/text_encoder/loaded-cpu", text_encoder)

            t = time.perf_counter()
            text_encoder = text_encoder.to(self._device)
            upload = time.perf_counter() - t
            print(
                f"[TIME] c{c} text_encoder weight upload (.to(device))   {upload:9.2f}s",
                flush=True,
            )
            snap(f"c{c}/text_encoder/placed")

            te_compiled = torch.compile(text_encoder, backend="tt")

            # COLD: first forward in this residency, carries the build.
            t0 = time.perf_counter()
            t = time.perf_counter()
            cap_pos = self._encode(prompt, te_compiled)
            te_cold = time.perf_counter() - t
            print(
                f"[TIME] c{c} text_encoder forward #1 (positive)         {te_cold:9.2f}s",
                flush=True,
            )
            snap(f"c{c}/text_encoder/ran-positive")

            # WARM: same residency, identical padded shapes -> tier 3.
            te_warm = None
            if do_cfg:
                t = time.perf_counter()
                cap_neg = self._encode(NEGATIVE_PROMPT, te_compiled)
                te_warm = time.perf_counter() - t
                print(
                    f"[TIME] c{c} text_encoder forward #2 (negative)         {te_warm:9.2f}s",
                    flush=True,
                )
                snap(f"c{c}/text_encoder/ran-negative")
            else:
                cap_neg = None

            # Shipped behaviour: both forwards folded into ONE published number.
            self._perf["components"]["text_encoder"] = time.perf_counter() - t0
            self._rec(
                stage="text_encoder",
                host_load_s=host_load,
                upload_s=upload,
                cold_s=te_cold,
                warm_s=te_warm,
                published_s=self._perf["components"]["text_encoder"],
            )

            del te_compiled, text_encoder
            gc.collect()
            torch_xla.sync()
            snap(f"c{c}/text_encoder/evicted")

            if self._stop_after == "text_encoder":
                self._perf["total"] = time.perf_counter() - t_total_start
                return None

            # ---- Latents + timesteps (verbatim from the shipped pipeline) --
            vsf = self.config.vae_scale_factor
            latent_h = 2 * (int(self.config.height) // (vsf * 2))
            latent_w = 2 * (int(self.config.width) // (vsf * 2))
            generator = torch.Generator(device="cpu").manual_seed(
                seed if seed is not None else SEED
            )
            latents = torch.randn(
                (1, LATENT_CHANNELS, latent_h, latent_w),
                generator=generator,
                dtype=torch.float32,
            )
            image_seq_len = (latent_h // 2) * (latent_w // 2)
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            set_ts_kwargs = {}
            if "mu" in inspect.signature(self.scheduler.set_timesteps).parameters:
                set_ts_kwargs["mu"] = mu
            self.scheduler.set_timesteps(
                num_inference_steps, device="cpu", **set_ts_kwargs
            )
            self.scheduler.set_begin_index(0)
            timesteps = self.scheduler.timesteps

            # ---- Denoising loop (transformer) ------------------------------
            t = time.perf_counter()
            transformer = TransformerWrapper(load_transformer(DTYPE)).eval()
            host_load = time.perf_counter() - t
            print(
                f"[TIME] c{c} transformer host load (from_pretrained)    {host_load:9.2f}s",
                flush=True,
            )
            snap_module(f"c{c}/transformer/loaded-cpu", transformer)

            t = time.perf_counter()
            transformer = transformer.to(self._device)
            upload = time.perf_counter() - t
            print(
                f"[TIME] c{c} transformer weight upload (.to(device))    {upload:9.2f}s",
                flush=True,
            )
            snap(f"c{c}/transformer/placed")

            tf_compiled = torch.compile(transformer, backend="tt")
            step_halves = []
            for i, ts in enumerate(timesteps):
                timestep = ((1000 - ts.expand(1)) / 1000).to(DTYPE)
                latent_input = latents.to(DTYPE)

                # CFG runs two forwards per step. cap_pos and cap_neg are
                # mask-trimmed to DIFFERENT lengths, so they are two distinct
                # graphs -- step 1 therefore carries TWO cold builds, and the
                # shipped single per-step timer folds them together. Record the
                # halves separately; the timed region is unchanged because each
                # _forward already ends in .cpu().
                t0 = time.perf_counter()
                pos = self._forward(tf_compiled, latent_input, timestep, cap_pos)
                pos_s = time.perf_counter() - t0
                neg_s = None
                if do_cfg:
                    t_neg = time.perf_counter()
                    neg = self._forward(tf_compiled, latent_input, timestep, cap_neg)
                    neg_s = time.perf_counter() - t_neg
                    pred = pos + GUIDANCE_SCALE * (pos - neg)
                else:
                    pred = pos
                step_s = time.perf_counter() - t0
                step_halves.append((pos_s, neg_s))
                self._perf["steps"].append(step_s)
                half = f"(pos {pos_s:.2f}s"
                half += f" + neg {neg_s:.2f}s)" if neg_s is not None else ")"
                print(
                    f"[TIME] c{c} transformer step {i + 1}/{num_inference_steps} {half:<26} {step_s:9.2f}s",
                    flush=True,
                )
                snap(f"c{c}/transformer/step{i + 1}")

                noise_pred = (-pred).squeeze(2)
                latents = self.scheduler.step(
                    noise_pred.to(torch.float32), ts, latents, return_dict=False
                )[0]

            self._rec(
                stage="transformer",
                host_load_s=host_load,
                upload_s=upload,
                steps_s=list(self._perf["steps"]),
                step_halves_s=step_halves,
            )

            del tf_compiled, transformer
            gc.collect()
            torch_xla.sync()
            snap(f"c{c}/transformer/evicted")

            if self._stop_after == "transformer":
                self._perf["total"] = time.perf_counter() - t_total_start
                return None

            # ---- VAE decode ------------------------------------------------
            t = time.perf_counter()
            vae_wrapper = VaeDecodeWrapper(load_vae(DTYPE)).eval()
            host_load = time.perf_counter() - t
            print(
                f"[TIME] c{c} vae host load (from_pretrained)            {host_load:9.2f}s",
                flush=True,
            )
            if self.config.vae_tiling and hasattr(vae_wrapper.vae, "enable_tiling"):
                vae_wrapper.vae.enable_tiling()
            snap_module(f"c{c}/vae/loaded-cpu", vae_wrapper)

            t = time.perf_counter()
            vae_wrapper = vae_wrapper.to(self._device)
            upload = time.perf_counter() - t
            print(
                f"[TIME] c{c} vae weight upload (.to(device))            {upload:9.2f}s",
                flush=True,
            )
            snap(f"c{c}/vae/placed")

            vae_compiled = torch.compile(vae_wrapper, backend="tt")

            # COLD
            t0 = time.perf_counter()
            image = vae_compiled(latents.to(self._device)).cpu().float()
            vae_cold = time.perf_counter() - t0
            self._perf["components"]["vae"] = vae_cold
            print(
                f"[TIME] c{c} vae decode #1 (cold)                       {vae_cold:9.2f}s",
                flush=True,
            )
            snap(f"c{c}/vae/ran")

            # WARM: synthetic in-residency repeats. Outputs discarded, so the
            # functional result is unchanged at any iteration count.
            vae_warm = []
            for k in range(self._vae_warm_iters):
                t = time.perf_counter()
                extra = vae_compiled(latents.to(self._device)).cpu().float()
                vae_warm.append(time.perf_counter() - t)
                print(
                    f"[TIME] c{c} vae decode #{k + 2} (warm)                       {vae_warm[-1]:9.2f}s",
                    flush=True,
                )
                snap(f"c{c}/vae/warm{k + 1}")
                del extra

            self._rec(
                stage="vae",
                host_load_s=host_load,
                upload_s=upload,
                cold_s=vae_cold,
                warm_s=vae_warm,
            )

            del vae_compiled, vae_wrapper
            gc.collect()
            torch_xla.sync()
            snap(f"c{c}/vae/evicted")

        self._perf["total"] = time.perf_counter() - t_total_start
        self._rec(stage="__total__", total_s=self._perf["total"])
        snap(f"c{c}/generate/exit")
        return image


def preflight(run_dir):
    """Fail loud before touching hardware."""
    problems = []
    if not os.path.exists("/dev/tenstorrent"):
        problems.append("no /dev/tenstorrent")
    if not os.environ.get("TT_METAL_CACHE"):
        problems.append(
            "TT_METAL_CACHE unset -- runs would use $HOME and not be reproducible"
        )
    # NOT offline. Z-Image's repo genuinely has no tokenizer/config.json, and
    # transformers 5.5.1 resolves a PreTrainedConfig before the tokenizer
    # (tokenization_auto.py:689). With network it HEADs, gets a 404 and moves on;
    # under HF_HUB_OFFLINE=1 "does not exist upstream" is indistinguishable from
    # "not cached", so it hard-errors. hf_hub DOES write .no_exist markers but
    # does not consult them when offline. Verified both ways 2026-08-24.
    # Instead assert the snapshot is already local, so no multi-GB download can
    # sneak into a timed run -- only HEAD calls for files that do not exist.
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        problems.append("HF_HOME unset")
    else:
        snap_dir = os.path.join(
            hf_home, "hub", "models--Tongyi-MAI--Z-Image", "snapshots"
        )
        if not os.path.isdir(snap_dir):
            problems.append(f"Z-Image snapshot not cached under {snap_dir}")
    free = shutil.disk_usage(run_dir).free / float(1 << 30)
    if free < MIN_FREE_GIB:
        problems.append(
            f"only {free:.1f}GiB free (need {MIN_FREE_GIB}); /proj_sw is at 95%"
        )
    if problems:
        for p in problems:
            print(f"[PREFLIGHT] FAIL {p}", file=sys.stderr)
        sys.exit(1)
    print(
        f"[PREFLIGHT] ok, {free:.1f}GiB free, devices={sorted(os.listdir('/dev/tenstorrent'))}",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="1,3,3", help="steps per generate() call")
    ap.add_argument(
        "--stop-after", default=None, choices=[None, "text_encoder", "transformer"]
    )
    ap.add_argument("--vae-warm-iters", type=int, default=0)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default="probe")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    preflight(args.run_dir)
    calls = [int(x) for x in args.calls.split(",")]

    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options(
        {
            # opt_level=1 keeps GroupNorm native so the 1280x720 VAE decode does
            # not OOM (#4755). The benchmark uses 1; anything else is not
            # comparable.
            "optimization_level": 1,
            "export_path": os.path.join(args.run_dir, "modules"),
            "export_model_name": "zimage_probe",
            # Match the shipped benchmark's option set exactly (see
            # imagegen_benchmark.py:85-92) so the numbers stay comparable --
            # ttnn perf metrics are ON there, and they are not free.
            "ttnn_perf_metrics_enabled": True,
            "ttnn_perf_metrics_output_file": os.path.join(
                args.run_dir, "ttnn_perf_metrics"
            ),
            "enable_trace": False,
        }
    )

    use_local_snapshot()
    probe_init(f"tag={args.tag} calls={calls} vae_warm_iters={args.vae_warm_iters}")

    pipeline = ProbeZImagePipeline(
        ZImageConfig(),
        stop_after=args.stop_after,
        vae_warm_iters=args.vae_warm_iters,
    )
    pipeline.setup()

    for i, steps in enumerate(calls):
        print(
            f"\n{'=' * 78}\n=== CALL {i + 1}/{len(calls)}  steps={steps}\n{'=' * 78}",
            flush=True,
        )
        t0 = time.perf_counter()
        image = pipeline.generate(prompt=PROMPT, num_inference_steps=steps, seed=SEED)
        wall = time.perf_counter() - t0
        print(f"[CALL] c{i + 1} wall {wall:9.2f}s  steps={steps}", flush=True)
        pipeline._probe.append(
            {"call": i + 1, "stage": "__call__", "wall_s": wall, "steps": steps}
        )
        if image is not None:
            torch.save(image, os.path.join(args.run_dir, f"call{i + 1}.pt"))

    out = os.path.join(args.run_dir, "probe.json")
    with open(out, "w") as handle:
        json.dump(pipeline._probe, handle, indent=2)
    print(f"\n[PROBE] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
