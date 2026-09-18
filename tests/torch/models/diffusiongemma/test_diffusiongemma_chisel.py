"""Per-component Chisel runs for DiffusionGemma (investigation, not a gate).

One component at a time, so op-level attribution is not buried in a 32k-op
whole-model trace, and so isolated-vs-accumulated can be read per component:

  isolated    -> is any single op broken?
  accumulated -> how much error has compounded by this component's output?

Inputs are REAL, never synthetic:
  vision_tower  the processed image (loader.load_vision_tower_inputs)
  embed_vision  a real CPU vision-tower last_hidden_state, captured by the
                loader from the full model on real image inputs (loader.py:178)
  encoder       the processed image+text prompt
They are cached to disk on first use and reused, so every run sees byte-identical
inputs. Component stats are printed so a wrong input is visible, not silent.

  pytest --enable-chisel -p chisel_fixups -svv \
    tests/torch/models/diffusiongemma/test_diffusiongemma_chisel.py \
    -k embed_vision
"""

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import get_mesh

from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.diffusiongemma.pytorch import loader as dg_loader

CACHE = Path(os.environ.get("DIFFGEMMA_INPUT_CACHE", "diffusiongemma_vision_logs/inputs"))
# Target chat-templated prompt length; 0 = use the loader default. Lets the
# 15-token and 277-token text cases run the same code path.
PROMPT_TOKENS = int(os.environ.get("DIFFGEMMA_PROMPT_TOKENS", "0"))


def _describe(name, t):
    if not torch.is_tensor(t):
        print(f"[input] {name}: {type(t).__name__}", flush=True)
        return
    f = t.detach().float()
    print(
        f"[input] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={f.mean():.6f} std={f.std():.6f} "
        f"min={f.min():.6f} max={f.max():.6f} "
        f"nan={int(f.isnan().sum())} inf={int(f.isinf().sum())}",
        flush=True,
    )


def _run(variant_name):
    loader_path = inspect.getsourcefile(dg_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.diffusiongemma.pytorch.loader import (
            ModelLoader,
            ModelVariant,
        )
        from third_party.tt_forge_models.diffusiongemma.pytorch.pipeline import (
            enable_spmd,
        )
        from tt_torch.moe_backend import (
            TT_MOE_BACKEND_NAME,
            register_tt_moe_backend,
        )

        variant = getattr(ModelVariant, variant_name)
        enable_spmd()
        register_tt_moe_backend()
        xr.set_device_type("TT")
        torch.manual_seed(0)

        loader = ModelLoader(variant)
        model = loader.load_model(dtype_override=torch.bfloat16)
        model.eval()
        loader.config._experts_implementation = TT_MOE_BACKEND_NAME

        CACHE.mkdir(parents=True, exist_ok=True)
        # token count in the name so the 15-tok and 277-tok text cases do not
        # overwrite each other's cache
        suffix = f"-{PROMPT_TOKENS}tok" if PROMPT_TOKENS else ""
        cached = CACHE / f"{variant.value}{suffix}.pt"
        if cached.exists():
            inputs = torch.load(cached, map_location="cpu", weights_only=False)
            print(f"[input] reused {cached}", flush=True)
        else:
            if PROMPT_TOKENS:
                from tests.torch.models.diffusiongemma._length_prompt import build_prompt

                prompt, got = build_prompt(loader, PROMPT_TOKENS)
                print(f"[input] length-controlled prompt: {got} tokens "
                      f"(target {PROMPT_TOKENS})", flush=True)
                inputs = loader.load_inputs(dtype_override=torch.bfloat16, prompt=prompt)
            else:
                inputs = loader.load_inputs(dtype_override=torch.bfloat16)
            torch.save(inputs, cached)
            print(f"[input] saved {cached}", flush=True)
        print(f"[input] seq_len={tuple(inputs['input_ids'].shape)}", flush=True)
        for k, v in inputs.items():
            _describe(k, v)

        # tensor-in / tensor-out wrappers so torch.compile can trace them
        if variant == ModelVariant.EMBED_VISION:
            class W(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, inputs_embeds):
                    return self.m(inputs_embeds=inputs_embeds)

            args = [inputs["inputs_embeds"]]
        elif variant == ModelVariant.VISION_TOWER:
            class W(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, pixel_values, pixel_position_ids):
                    return self.m(
                        pixel_values=pixel_values,
                        pixel_position_ids=pixel_position_ids,
                    ).last_hidden_state

            args = [inputs["pixel_values"], inputs["pixel_position_ids"]]
        else:
            is_image = "IMAGE" in variant_name

            class W(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, input_ids, attention_mask, mm_token_type_ids,
                            pixel_values=None, image_position_ids=None):
                    return self.m(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        mm_token_type_ids=mm_token_type_ids,
                        pixel_values=pixel_values,
                        image_position_ids=image_position_ids,
                    ).last_hidden_state

            args = [inputs["input_ids"], inputs["attention_mask"],
                    inputs["mm_token_type_ids"]]
            if is_image:
                args += [inputs["pixel_values"], inputs["image_position_ids"]]

        xla = xm.xla_device()
        mesh = get_mesh(*loader.get_mesh_config(xr.global_runtime_device_count()))
        model = model.to(xla)
        xs.set_global_mesh(mesh)
        specs = loader.load_shard_spec(model)
        print(f"[shard] {len(specs)} tensors sharded "
              f"({'replicated component' if not specs else 'sharded'})", flush=True)
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)

        compiled = torch.compile(W(model), backend="tt")
        print(f"[run] {variant.value} under Chisel ...", flush=True)
        with torch.no_grad():
            out = compiled(*(a.to(xla) for a in args))
            xm.mark_step()
            host = out.to("cpu")
        _describe("OUTPUT", host)


@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.llmbox
def test_chisel_embed_vision():
    _run("EMBED_VISION")


@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.llmbox
def test_chisel_vision_tower():
    _run("VISION_TOWER")


@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.llmbox
def test_chisel_encoder_image():
    _run(os.environ.get("DIFFGEMMA_CHISEL_VARIANT", "ENCODER_IMAGE"))
