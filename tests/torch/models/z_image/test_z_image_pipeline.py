# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image — end-to-end text-to-image pipeline (Tongyi-MAI/Z-Image).

Stitches the three validated single-chip components into the full generation
flow, mirroring diffusers ``ZImagePipeline.__call__``:

    Qwen3 text encoder -> ZImageTransformer2DModel denoising loop (CFG +
    FlowMatchEulerDiscreteScheduler) -> AutoencoderKL decode -> PIL image.

Each component compiles with the ``tt`` backend and runs on a single
Blackhole chip. Source inference parameters (prompt, 1280x720, 50 steps,
guidance_scale=4.0) are used so the run produces a realistic image.

Every stage is gated on PCC against a CPU twin fed the same inputs the device
saw: the prompt embeds once, the conditional noise prediction on the first
``PCC_CHECK_STEPS`` denoise steps, and the decoded pixels once. The trajectory is
always advanced with the *device* output (deployment behavior), so a PCC drop
anywhere shows up as a test failure rather than a silently degraded image.

The encoder and VAE goldens run on the host copy before that component is placed
on device, so they cost no second copy. The transformer's twin is a second ~6.2B
module, so it is loaded lazily at the first checked step and dropped once the
checked steps are done.
"""

from __future__ import annotations

import gc
import inspect
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category, ModelGroup, TTArch, get_torch_device_arch

from third_party.tt_forge_models.z_image.pytorch.src.model_utils import (
    CFG_NORMALIZATION,
    DTYPE,
    GUIDANCE_SCALE,
    HEIGHT,
    LATENT_CHANNELS,
    MAX_SEQUENCE_LENGTH,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    PROMPT,
    REPO_ID,
    SEED,
    VAE_SCALE_FACTOR,
    WIDTH,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)

# --- diffusers.pipelines.z_image.pipeline_z_image helpers (inlined) ---------


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


# --- PCC gating ---------------------------------------------------------------

# Denoise steps that get a CPU twin + PCC assert. The twin is a second ~6.2B
# module and one CPU forward dominates the step time, so gate the leading steps
# (where a numerical break shows up first) instead of all 50. Only the
# conditional (positive) forward is gated; the negative branch shares the same
# weights and graph.
PCC_CHECK_STEPS = 4

# Every stage clears the default, so no stage needs a relaxed gate.
PCC_THRESHOLD = 0.99

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _assert_pcc(stage: str, device_out, golden_out, threshold: float) -> float:
    pcc = _pcc(device_out, golden_out)
    logger.info("[PCC] {}: pcc={:.6f} (threshold {})", stage, pcc, threshold)
    assert pcc >= threshold, f"{stage} PCC {pcc:.6f} below threshold {threshold}"
    return pcc


# --- TT-compilable component wrappers (tensor in / tensor out) ---------------


class TextEncoderWrapper(torch.nn.Module):
    """Qwen3 encoder -> penultimate hidden state (hidden_states[-2])."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-2]


class CondTransformerWrapper(torch.nn.Module):
    """Single (conditional) transformer pass; cap_feats is (L, D)."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, latents, timestep, cap_feats):
        x_list = list(latents.unsqueeze(2).unbind(dim=0))
        t = timestep.reshape(-1).to(dtype=latents.dtype)
        out = self.transformer(x_list, t, [cap_feats], return_dict=False)[0]
        return torch.stack([o.float() for o in out], dim=0)


class VaeDecodeWrapper(torch.nn.Module):
    """Undo latent scaling, then AutoencoderKL.decode -> pixels."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        z = latents.to(dtype=self.vae.dtype)
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]


# --- Pipeline ---------------------------------------------------------------


class ZImagePipeline:
    """Self-contained Z-Image text-to-image pipeline for TT bring-up."""

    def __init__(self, dtype: torch.dtype = DTYPE):
        self.dtype = dtype
        self.vae_scale_factor = VAE_SCALE_FACTOR
        self._twin_transformer = None
        self.pccs = {}

    def setup(self):
        self.tokenizer = load_tokenizer()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            REPO_ID, subfolder="scheduler"
        )

        # Components are loaded on CPU and placed on-device one at a time (see
        # _on_device): the ~6.2B transformer, the Qwen3 text encoder and the VAE
        # do not all fit resident on a single Blackhole (issue #4756), so each is
        # placed -> used -> evicted in turn, keeping peak DRAM ~= max(component).
        self.text_encoder = TextEncoderWrapper(load_text_encoder(self.dtype)).eval()
        self.transformer = CondTransformerWrapper(load_transformer(self.dtype)).eval()
        self.vae_decoder = VaeDecodeWrapper(load_vae(self.dtype)).eval()
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self._device = torch_xla.device()

    def _to_tt(self, x):
        return x.to(device=self._device)

    @staticmethod
    def _to_cpu(x):
        return x.to("cpu")

    @contextmanager
    def _on_device(self, module):
        """Place one component on-device (compiled), then evict it afterwards.

        Keeps only a single heavy component resident at a time (see setup): the
        module is moved to device and compiled with the tt backend for the
        duration of the block, then moved back to CPU and its compiled graph
        dropped so the next component has the DRAM headroom it needs.
        """
        module.to(self._device)
        compiled = torch.compile(module, backend="tt")
        try:
            yield compiled
        finally:
            module.to("cpu")
            del compiled
            gc.collect()
            torch_xla.sync()

    def _encode_prompt(self, prompt: str, encoder, device=None) -> torch.Tensor:
        """Tokenize (chat template) -> penultimate hidden state, mask-trimmed.

        ``device=None`` runs on CPU, which is how the golden is produced before
        the encoder is placed on device.
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        text_inputs = self.tokenizer(
            [text],
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask.bool()
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        hidden = encoder(input_ids, attention_mask)
        hidden = self._to_cpu(hidden)
        mask = text_inputs.attention_mask[0].bool()
        # Ragged, mask-trimmed embedding for this prompt: (valid_len, dim).
        return hidden[0][mask].to(self.dtype)

    def _transformer_step(self, transformer, latents, timestep, cap_feats, device=None):
        """One batch=1 transformer pass; returns CPU fp32 (1, C, 1, H, W).

        ``device=None`` runs on CPU, which is how the per-step golden is produced.
        """
        if device is not None:
            latents = latents.to(device)
            timestep = timestep.to(device)
            cap_feats = cap_feats.to(device)
        out = transformer(latents, timestep, cap_feats)
        return self._to_cpu(out).float()

    def _cpu_twin_transformer(self):
        """Second copy of the denoiser on CPU, loaded on first use.

        Only resident while the checked steps run; released by
        ``_release_twin_transformer`` right after.
        """
        if self._twin_transformer is None:
            logger.info("[load] CPU twin: transformer ({})", self.dtype)
            self._twin_transformer = CondTransformerWrapper(
                load_transformer(self.dtype)
            ).eval()
        return self._twin_transformer

    def _release_twin_transformer(self):
        if self._twin_transformer is not None:
            logger.info("[free] CPU twin: transformer (checked steps done)")
            self._twin_transformer = None
            gc.collect()

    def generate(
        self,
        prompt: str = PROMPT,
        negative_prompt: str = NEGATIVE_PROMPT,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        cfg_normalization: bool = CFG_NORMALIZATION,
        seed: int = SEED,
        output_type: str = "pil",
    ):
        do_cfg = guidance_scale > 0
        cpu = torch.device("cpu")

        with torch.no_grad():
            # 1. Text encoding (Qwen3, penultimate layer); encoder resident only
            #    for this block, then evicted before the transformer is placed.
            # Golden first, while the encoder is still on host: costs no copy.
            logger.info("[STAGE] text_encoder: start")
            golden_cap_pos = self._encode_prompt(prompt, self.text_encoder)
            with self._on_device(self.text_encoder) as encoder:
                cap_pos = self._encode_prompt(prompt, encoder, self._device)
                cap_neg = (
                    self._encode_prompt(negative_prompt, encoder, self._device)
                    if do_cfg
                    else None
                )
            self.pccs["text_encoder"] = _assert_pcc(
                "text_encoder", cap_pos, golden_cap_pos, PCC_THRESHOLD
            )
            del golden_cap_pos
            logger.info("[STAGE] text_encoder: done")

            # 2. Latents (fp32 on CPU; scheduler math stays fp32).
            latent_h = 2 * (int(height) // (self.vae_scale_factor * 2))
            latent_w = 2 * (int(width) // (self.vae_scale_factor * 2))
            generator = torch.Generator(device="cpu").manual_seed(seed)
            latents = torch.randn(
                (1, LATENT_CHANNELS, latent_h, latent_w),
                generator=generator,
                dtype=torch.float32,
                device=cpu,
            )

            # 3. Timesteps with resolution-dependent shift (mu).
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
            timesteps, _ = retrieve_timesteps(
                self.scheduler, num_inference_steps, cpu, **set_ts_kwargs
            )
            self.scheduler.set_begin_index(0)

            # 4. Denoising loop; transformer resident only for this block, then
            #    evicted before the VAE is placed.
            with self._on_device(self.transformer) as transformer:
                for i, t in enumerate(timesteps):
                    timestep = t.expand(1)
                    timestep = (1000 - timestep) / 1000
                    latent_input = latents.to(self.dtype)
                    timestep_input = timestep.to(self.dtype)

                    pos = self._transformer_step(
                        transformer, latent_input, timestep_input, cap_pos, self._device
                    )

                    if i < PCC_CHECK_STEPS:
                        golden_pos = self._transformer_step(
                            self._cpu_twin_transformer(),
                            latent_input,
                            timestep_input,
                            cap_pos,
                        )
                        self.pccs[f"transformer step {i + 1}"] = _assert_pcc(
                            f"transformer step {i + 1}/{num_inference_steps}",
                            pos,
                            golden_pos,
                            PCC_THRESHOLD,
                        )
                        del golden_pos
                        if i + 1 == PCC_CHECK_STEPS:
                            self._release_twin_transformer()

                    if do_cfg:
                        neg = self._transformer_step(
                            transformer,
                            latent_input,
                            timestep_input,
                            cap_neg,
                            self._device,
                        )
                        pred = pos + guidance_scale * (pos - neg)
                        if cfg_normalization and float(cfg_normalization) > 0.0:
                            ori = torch.linalg.vector_norm(pos)
                            new = torch.linalg.vector_norm(pred)
                            max_norm = ori * float(cfg_normalization)
                            if new > max_norm:
                                pred = pred * (max_norm / new)
                        noise_pred = pred
                    else:
                        noise_pred = pos

                    noise_pred = noise_pred.squeeze(2)
                    noise_pred = -noise_pred
                    latents = self.scheduler.step(
                        noise_pred.to(torch.float32), t, latents, return_dict=False
                    )[0]
                    logger.info(f"  denoise step {i + 1}/{num_inference_steps}")

            if output_type == "latent":
                return latents

            # 5. VAE decode (scaling folded into the wrapper); VAE resident only
            #    for this block.
            logger.info("[STAGE] vae: start")
            # Golden first, while the VAE is still on host: costs no copy.
            golden_image = self.vae_decoder(latents).float()
            with self._on_device(self.vae_decoder) as vae_decoder:
                image = vae_decoder(self._to_tt(latents))
                image = self._to_cpu(image).float()
            self.pccs["vae decode"] = _assert_pcc(
                "vae decode", image, golden_image, PCC_THRESHOLD
            )
            del golden_image
            logger.info("[STAGE] vae: done")
            return self.image_processor.postprocess(image, output_type=output_type)


def run_zimage_pipeline(
    output_path: str = "zimage_output.png",
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    output_type: str = "pil",
):
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    pipeline = ZImagePipeline()
    pipeline.setup()

    result = pipeline.generate(
        num_inference_steps=num_inference_steps,
        output_type=output_type,
    )

    if output_type == "latent":
        logger.info(f"Latent output shape: {result.shape}")
        return result

    image = result[0]
    image.save(output_path)
    logger.info(f"Image saved to {output_path} ({image.size})")
    return result


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="ZImage_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_z_image_pipeline():
    """Full Z-Image text-to-image e2e on a single Blackhole chip.

    optimization_level=1 keeps GroupNorm as native ttnn.group_norm so the VAE
    decode at 1280x720 does not OOM (issue #4755).
    """
    xr.set_device_type("TT")
    # The ~6.2B transformer + Qwen3 encoder + VAE fit a single Blackhole, but their
    # weights exceed the DRAM a single Wormhole (n150) provides, so this e2e is
    # Blackhole-only (issue #4756).
    if get_torch_device_arch() != TTArch.BLACKHOLE:
        pytest.skip(
            "Z-Image e2e runs on Blackhole only: the model weights exceed the "
            "DRAM a single Wormhole chip provides"
        )
    torch.manual_seed(SEED)

    output_path = "test_zimage_output.png"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    images = run_zimage_pipeline(output_path=output_path, output_type="pil")

    assert images is not None, "Pipeline returned None"
    assert len(images) == 1, f"Expected 1 image, got {len(images)}"
    assert isinstance(images[0], Image.Image), "Output is not a PIL image"
    assert output_file.exists(), "Output image was not saved"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
    logger.info(f"Z-Image e2e pipeline test passed ({width}x{height}).")
