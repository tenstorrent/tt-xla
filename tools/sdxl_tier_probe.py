# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DEV-ONLY probe: SDXL-Lightning, the CONTRAST CASE. Must never reach a PR.

Z-Image destroys each component (`del compiled, module; gc.collect();
torch_xla.sync()`), so eviction provably discards the compiled graph and the
driver's warmup pass cannot help. SDXL-Lightning is the opposite shape and is
the one place the shipped two-pass scheme might legitimately work:

  - `model.compile(backend="tt")` is called ONCE in load_models() during
    setup(), not per generate().
  - Eviction is `module.to("cpu")` ONLY -- no del, no gc, no dynamo reset. The
    module object and its compiled backend stay alive across calls.

THE QUESTION: does .to("cpu") -> .to(device) preserve the compiled graph?
  If YES  -> call 2 is genuinely warm, the two-pass scheme is VALID here, and
             the issue must scope itself to destroy-per-stage pipelines.
  If NO   -> the same defect as Z-Image, reached by a different mechanism, and
             the "modules stay alive" reasoning is a false comfort.
Either answer is worth having; this is the axis under test.

SECOND DIFFERENCE THAT MATTERS: unlike Z-Image (whose text encoder runs twice
per residency for CFG), EVERY SDXL one-shot component runs exactly ONE forward.
So text_encoder_1, text_encoder_2 and vae have NO natural warm counterpart at
all -- all three need a synthetic in-residency repeat. Z-Image got two of its
three warm numbers for free; SDXL gets none.

Mirrors the driver exactly: warmup generate(1 step) then steady generate(4),
plus a third call the driver never makes, as the plateau control.
"""

import argparse
import json
import os
import shutil
import sys
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "benchmark",
    ),
)

from tools.residency_probe import (
    probe_init,
    snap,
    snap_module,
    stage_timer,
)  # noqa: E402

from benchmarks.sdxl_lightning_pipeline import (  # noqa: E402
    SDXLLightningConfig,
    SDXLLightningPipeline,
)

MIN_FREE_GIB = 20.0
PROMPT = "A girl smiling"


class ProbeSDXLPipeline(SDXLLightningPipeline):
    """Instrumented copy of generate(). The shipped file is NOT modified.

    Adds, per one-shot component: a snap around the cold forward, then
    `warm_iters` extra forwards inside the SAME residency (before the
    .to("cpu")), outputs discarded. The functional path is untouched.
    """

    def __init__(self, config, *, warm_iters=1):
        super().__init__(config)
        self._probe = []
        self._warm_iters = warm_iters
        self._call = 0

    def _rec(self, **kw):
        kw["call"] = self._call
        self._probe.append(kw)

    def _warm(self, name, fn):
        """Extra in-residency forwards, discarded. Inert at warm_iters=0."""
        out = []
        c = self._call
        for k in range(self._warm_iters):
            t = time.perf_counter()
            extra = fn()
            out.append(time.perf_counter() - t)
            print(
                f"[TIME] c{c} {name} warm #{k + 1}{'':<20} {out[-1]:9.2f}s", flush=True
            )
            snap(f"c{c}/{name}/warm{k + 1}")
            del extra
        return out

    def generate(self, prompt, num_inference_steps=4, seed=None):
        self._call += 1
        c = self._call
        batch_size = 1
        device = xm.xla_device()
        self._perf = {
            "components": {},
            "steps": [],
            "step_metric_name": "unet_step",
            "total": None,
        }
        t_total_start = time.perf_counter()
        snap(f"c{c}/generate/entry")

        with torch.no_grad():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed) if seed is not None else generator.seed()

            # ---- Text encoder 1 -------------------------------------------
            tokens_1 = self.tokenizer(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device="cpu")
            if self.config.text_encoder_on_tt:
                t = time.perf_counter()
                self.text_encoder = self.text_encoder.to(device)
                print(
                    f"[TIME] c{c} te1 .to(device){'':<27} {time.perf_counter() - t:9.2f}s",
                    flush=True,
                )
                tokens_1 = tokens_1.to(device=device)
            snap_module(f"c{c}/te1/placed", self.text_encoder)

            t0 = time.perf_counter()
            prompt_embeds_1 = self.text_encoder(tokens_1)
            if self.config.text_encoder_on_tt:
                prompt_embeds_1 = prompt_embeds_1.to("cpu")
            te1_cold = time.perf_counter() - t0
            self._perf["components"]["text_encoder_1"] = te1_cold
            print(
                f"[TIME] c{c} te1 forward (COLD){'':<24} {te1_cold:9.2f}s", flush=True
            )
            snap(f"c{c}/te1/ran")

            te1_warm = self._warm("te1", lambda: self.text_encoder(tokens_1).to("cpu"))
            self._rec(stage="text_encoder_1", cold_s=te1_cold, warm_s=te1_warm)

            if self.config.text_encoder_on_tt:
                self.text_encoder = self.text_encoder.to(
                    "cpu"
                )  # THE EVICTION UNDER TEST
            snap(f"c{c}/te1/evicted-to-cpu")

            # ---- Text encoder 2 -------------------------------------------
            tokens_2 = self.tokenizer_2(
                [prompt],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device="cpu")
            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to(device)
                tokens_2 = tokens_2.to(device=device)
            snap_module(f"c{c}/te2/placed", self.text_encoder_2)

            t0 = time.perf_counter()
            prompt_embeds_2, pooled_prompt_embeds = self.text_encoder_2(tokens_2)
            if self.config.text_encoder_2_on_tt:
                prompt_embeds_2 = prompt_embeds_2.to("cpu")
                pooled_prompt_embeds = pooled_prompt_embeds.to("cpu")
            te2_cold = time.perf_counter() - t0
            self._perf["components"]["text_encoder_2"] = te2_cold
            print(
                f"[TIME] c{c} te2 forward (COLD){'':<24} {te2_cold:9.2f}s", flush=True
            )
            snap(f"c{c}/te2/ran")

            def _te2():
                a, b = self.text_encoder_2(tokens_2)
                return (a.to("cpu"), b.to("cpu"))

            te2_warm = self._warm("te2", _te2)
            self._rec(stage="text_encoder_2", cold_s=te2_cold, warm_s=te2_warm)

            if self.config.text_encoder_2_on_tt:
                self.text_encoder_2 = self.text_encoder_2.to("cpu")
            snap(f"c{c}/te2/evicted-to-cpu")

            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
            add_text_embeds = pooled_prompt_embeds
            add_time_ids = self._get_add_time_ids(prompt_embeds.dtype).to("cpu")

            self.scheduler.set_timesteps(num_inference_steps, device="cpu")
            timesteps = self.scheduler.timesteps
            latent_shape = (
                batch_size,
                4,
                self.config.latents_height,
                self.config.latents_width,
            )
            latents = torch.randn(
                latent_shape, generator=generator, dtype=torch.float32
            ).to("cpu")
            latents = latents * self.scheduler.init_noise_sigma

            # ---- UNet denoise loop ----------------------------------------
            if self.config.unet_on_tt:
                self.unet = self.unet.to(device)
                unet_eh = prompt_embeds.to(torch.bfloat16).to(device)
                unet_te = add_text_embeds.to(torch.bfloat16).to(device)
                unet_ti = add_time_ids.to(torch.bfloat16).to(device)
            else:
                unet_eh, unet_te, unet_ti = prompt_embeds, add_text_embeds, add_time_ids
            snap_module(f"c{c}/unet/placed", self.unet)

            for i, t_step in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t_step)
                if self.config.unet_on_tt:
                    unet_sample = latent_model_input.to(torch.bfloat16).to(device)
                    unet_t = t_step.to(torch.bfloat16).to(device)
                else:
                    unet_sample, unet_t = latent_model_input, t_step

                t0 = time.perf_counter()
                noise_pred = self.unet(unet_sample, unet_t, unet_eh, unet_te, unet_ti)
                if self.config.unet_on_tt:
                    noise_pred = noise_pred.to("cpu").to(torch.float32)
                step_s = time.perf_counter() - t0
                self._perf["steps"].append(step_s)
                print(
                    f"[TIME] c{c} unet step {i + 1}/{num_inference_steps}{'':<26} {step_s:9.2f}s",
                    flush=True,
                )
                snap(f"c{c}/unet/step{i + 1}")

                latents = self.scheduler.step(
                    noise_pred, t_step, latents, return_dict=False
                )[0]

            self._rec(stage="unet", steps_s=list(self._perf["steps"]))
            if self.config.unet_on_tt:
                self.unet = self.unet.to("cpu")
            snap(f"c{c}/unet/evicted-to-cpu")

            # ---- VAE decode ------------------------------------------------
            latents = latents / self.vae.vae.config.scaling_factor
            if self.config.vae_on_tt:
                torch_xla.set_custom_compile_options(
                    {**self.config.compile_options, "optimization_level": 1}
                )
                self.vae = self.vae.to(device)
                latents = latents.to(device)
            snap_module(f"c{c}/vae/placed", self.vae)

            t0 = time.perf_counter()
            image = self.vae(latents)
            if self.config.vae_on_tt:
                image = image.to("cpu")
            vae_cold = time.perf_counter() - t0
            self._perf["components"]["vae"] = vae_cold
            print(f"[TIME] c{c} vae decode (COLD){'':<25} {vae_cold:9.2f}s", flush=True)
            snap(f"c{c}/vae/ran")

            vae_warm = self._warm("vae", lambda: self.vae(latents).to("cpu"))
            self._rec(stage="vae", cold_s=vae_cold, warm_s=vae_warm)

            if self.config.vae_on_tt:
                self.vae = self.vae.to("cpu")
                torch_xla.set_custom_compile_options(self.config.compile_options)
            snap(f"c{c}/vae/evicted-to-cpu")

            self._perf["total"] = time.perf_counter() - t_total_start
            self._rec(stage="__total__", total_s=self._perf["total"])
            snap(f"c{c}/generate/exit")
            return image


def preflight(run_dir):
    problems = []
    if not os.path.exists("/dev/tenstorrent"):
        problems.append("no /dev/tenstorrent")
    if not os.environ.get("TT_METAL_CACHE"):
        problems.append("TT_METAL_CACHE unset")
    free = shutil.disk_usage(run_dir).free / float(1 << 30)
    if free < MIN_FREE_GIB:
        problems.append(f"only {free:.1f}GiB free")
    if problems:
        for p in problems:
            print(f"[PREFLIGHT] FAIL {p}", file=sys.stderr)
        sys.exit(1)
    print(f"[PREFLIGHT] ok, {free:.1f}GiB free", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="1,4,4")
    ap.add_argument("--warm-iters", type=int, default=1)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tag", default="sdxl")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)
    preflight(args.run_dir)
    calls = [int(x) for x in args.calls.split(",")]

    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    options = {
        # The shipped benchmark uses optimization_level=0 for SDXL
        # (test_imagegen.py::test_sdxl_lightning), unlike Z-Image's 1.
        "optimization_level": 0,
        "export_path": os.path.join(args.run_dir, "modules"),
        "export_model_name": "sdxl_probe",
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": os.path.join(
            args.run_dir, "ttnn_perf_metrics"
        ),
        "enable_trace": False,
    }
    torch_xla.set_custom_compile_options(options)
    probe_init(f"tag={args.tag} calls={calls} warm_iters={args.warm_iters}")

    pipeline = ProbeSDXLPipeline(
        SDXLLightningConfig(compile_options=options), warm_iters=args.warm_iters
    )
    with stage_timer("setup() [loads + compile(backend=tt) for all 4]"):
        pipeline.setup()
    snap("setup/done")

    for i, steps in enumerate(calls):
        print(
            f"\n{'=' * 78}\n=== CALL {i + 1}/{len(calls)}  steps={steps}\n{'=' * 78}",
            flush=True,
        )
        t0 = time.perf_counter()
        image = pipeline.generate(prompt=PROMPT, num_inference_steps=steps, seed=42)
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
