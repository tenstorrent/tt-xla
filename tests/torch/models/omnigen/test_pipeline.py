# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""OmniGen — nightly e2e text-to-image pipeline test with per-step DiT PCC checks.

OmniGen is a unified image-generation DiT with a LLaMA-style backbone that
embeds text tokens internally. The transformer is the heavy net and runs
**tensor-parallel across a multi-chip mesh** (Megatron-1D on the ``"model"``
axis); the FlowMatchEuler scheduler and the AutoencoderKL VAE stay on CPU. This
drives the shared ``OmniGenPipeline`` from ``tt_forge_models`` (the same
pipeline the image-gen benchmark uses) end-to-end, gates its numerics per
denoising step and asserts the saved image dimensions.

Following tt-xla#5480 (Playground v2.5 / SDXL-Lightning) and the GLM-Image
pipeline test (tt-xla#5570), the DiT forward is wrapped so that after every TT
forward the *same* inputs are fed to a lazily loaded fp32 CPU twin of the
transformer and PCC is compared inline. The test fails fast the moment any DiT
step drops below ``PCC_THRESHOLD``. The pipeline itself keeps using the TT
outputs (real deployment behaviour); the CPU twin is a side reference fed the
same (bf16-rounded) inputs the TT DiT consumed.

The DiT is the only component that runs on TT in this pipeline — the scheduler
and the VAE both run on CPU here, so their TT correctness stays covered by the
standalone component tests in this directory rather than by this pipeline test.
"""

from pathlib import Path

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.omnigen.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.omnigen.pytorch.src.pipeline import (
    HEIGHT,
    WIDTH,
    OmniGenConfig,
    OmniGenPipeline,
)

VARIANT_NAME = ModelVariant.TRANSFORMER
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

SEED = 42
NUM_INFERENCE_STEPS = 30
PROMPT = "A realistic photo of a cat wearing sunglasses sitting on a sunny beach."
# bf16 DiT on TT vs a clean fp32 CPU reference. Mirrors tt-xla#5480's 0.99 gate;
# tune against the first nightly if the bf16/fp32 gap sits below this.
PCC_THRESHOLD = 0.99


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _patch_dynamo_id_match_guard() -> None:
    """Repair tt_torch's stale ``GuardBuilder.id_match_unchecked`` monkey-patch.

    ``python_package/tt_torch/utils.py`` (applied at ``tt_torch`` import) calls
    ``GuardManager.add_id_match_guard(id, verbose_parts)`` with two positional
    args, but the installed torch requires three
    (``id, verbose_parts, user_stack``). Any ``nn.Module`` ID_MATCH guard then
    raises ``InternalTorchDynamoError`` during ``torch.compile`` — OmniGen's
    ``rope`` (``OmniGenSuScaledRotaryEmbedding``) submodule triggers it.

    ``tt_torch/utils.py`` is not editable here, so reinstall a corrected
    ``id_match_unchecked`` (keeping tt_torch's XLA-tensor repr guard so sharded
    weights are not materialized during guard construction) and stamp it with
    tt_torch's own idempotency flag so ``apply_xla_dynamo_guard_repr_patch()``
    treats it as already-applied and never clobbers it back to the broken form.
    The ``add_id_match_guard`` call is arity-adaptive so this stays correct if
    the installed torch reverts to the two-arg signature.

    TODO: drop once tt_torch's guard patch is fixed upstream.
    """
    try:
        from torch._dynamo.guards import GuardBuilder, get_verbose_code_parts
        from torch._dynamo.source import LocalSource, TypeSource
        from torch._guards import Guard
    except ImportError:
        return

    def id_match_unchecked(self, guard, recompile_hint=None) -> None:
        if isinstance(guard.originating_source, TypeSource):
            return self.TYPE_MATCH(
                Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH)
            )

        ref = self.arg_ref(guard)
        val = self.get(guard)
        id_val = self.id_ref(val, guard.name)

        try:
            if isinstance(val, torch.Tensor) and val.device.type == "xla":
                type_repr = f"<{type(val).__name__} device={val.device}>"
            else:
                type_repr = repr(val)
        except Exception:
            type_repr = f"<{type(val).__name__}>"

        code = f"___check_obj_id({ref}, {id_val}), type={type_repr}"
        self._set_guard_export_info(guard, [code], provided_func_name="ID_MATCH")
        verbose_parts = get_verbose_code_parts(code, guard, recompile_hint)
        manager = self.get_guard_manager(guard)
        try:
            manager.add_id_match_guard(id_val, verbose_parts, guard.user_stack)
        except TypeError:
            manager.add_id_match_guard(id_val, verbose_parts)

        if isinstance(guard.originating_source, LocalSource):
            if isinstance(val, torch.nn.Module):
                local_name = guard.originating_source.local_name
                weak_id = self.lookup_weakrefs(val)
                if weak_id is not None:
                    self.id_matched_objs[local_name] = weak_id

    id_match_unchecked._tt_xla_guard_repr_patch = True
    GuardBuilder.id_match_unchecked = id_match_unchecked


def _attach_dit_pcc_check(pipeline: OmniGenPipeline) -> None:
    """Wrap the pipeline's DiT forward with an inline fp32-CPU-twin PCC check.

    Only the DiT transformer runs on TT, so it is the sole component gated here.
    After each TT forward the same inputs are replayed on a lazily loaded fp32
    CPU twin and PCC is asserted inline, so the test fails fast on the first
    diverging denoising step. OmniGen batches the classifier-free-guidance rows
    (cond + uncond) into a single forward, so there is one check per step.

    Must be called after ``pipeline.setup()`` — setup shards the DiT, moves it
    to the XLA device and installs the routed ``transformer.forward`` that
    ``generate()`` invokes.
    """
    transformer = pipeline.pipe.transformer
    # The routed (device) forward installed by setup(); captured before the
    # instance attribute below shadows it.
    orig_forward = transformer.forward

    twin = {"model": None}
    step = {"n": 0}

    def _cpu_twin():
        # Loaded on first use: a fresh fp32 CPU copy of the raw
        # OmniGenTransformer2DModel (via the loader's wrapper). It stays plain
        # fp32 (no bf16 cast, no MLP gate_up split) so it is a clean golden
        # reference for the bf16 TT DiT.
        if twin["model"] is None:
            logger.info("[PCC] loading CPU fp32 DiT twin")
            wrapper = ModelLoader(ModelVariant.TRANSFORMER).load_model(
                dtype_override=torch.float32
            )
            twin["model"] = wrapper.transformer
        return twin["model"]

    def _cpu(x):
        # Move to CPU and upcast floats to fp32 so the twin sees the same values
        # the TT DiT consumed; leave int/bool tensors (input_ids/position_ids)
        # untouched.
        if not isinstance(x, torch.Tensor):
            return x
        x = x.to("cpu")
        return x.float() if x.is_floating_point() else x

    def wrapped_forward(*args, **kwargs):
        # Real TT forward — the pipeline continues with this output.
        out = orig_forward(*args, **kwargs)
        device_sample = out[0] if isinstance(out, (tuple, list)) else out
        device_sample = device_sample.to("cpu").float()

        # Replay the same inputs on the fp32 CPU twin. The diffusers OmniGen
        # pipeline always calls the DiT with these keywords (see
        # pipeline_omnigen.OmniGenPipeline.__call__). ``input_img_latents`` and
        # ``input_image_sizes`` are the empty text-only structures and pass
        # through unchanged.
        golden = _cpu_twin()(
            hidden_states=_cpu(kwargs["hidden_states"]),
            timestep=_cpu(kwargs["timestep"]),
            input_ids=_cpu(kwargs["input_ids"]),
            input_img_latents=kwargs["input_img_latents"],
            input_image_sizes=kwargs["input_image_sizes"],
            attention_mask=_cpu(kwargs["attention_mask"]),
            position_ids=_cpu(kwargs["position_ids"]),
            return_dict=False,
        )
        golden_sample = golden[0] if isinstance(golden, (tuple, list)) else golden
        golden_sample = golden_sample.to("cpu").float()

        step["n"] += 1
        pcc = _pcc(device_sample, golden_sample)
        logger.info(f"[PCC] dit forward {step['n']}: pcc={pcc:.6f}")
        assert (
            pcc >= PCC_THRESHOLD
        ), f"DiT forward {step['n']} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"

        return out

    transformer.forward = wrapped_forward
    pipeline.pipe.transformer = transformer


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_pipeline():
    """Run the OmniGen pipeline (DiT tensor-parallel) with per-step DiT PCC."""
    xr.set_device_type("TT")

    # Repair tt_torch's stale dynamo ID_MATCH guard patch before any compile.
    _patch_dynamo_id_match_guard()

    pipeline = OmniGenPipeline(
        config=OmniGenConfig(num_inference_steps=NUM_INFERENCE_STEPS)
    )
    pipeline.setup()

    # Gate the DiT (the only TT component) against an fp32 CPU twin per step.
    _attach_dit_pcc_check(pipeline)

    # ``generate`` runs the upstream sampling loop and returns a (1, 3, H, W)
    # float tensor in [0, 1] (output_type="pt").
    image = pipeline.generate(prompt=PROMPT, seed=SEED)

    array = (
        image[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).cpu()
    ).numpy()

    output_path = "omnigen_pipeline_output.png"
    Image.fromarray(array).save(output_path)

    assert Path(output_path).exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
