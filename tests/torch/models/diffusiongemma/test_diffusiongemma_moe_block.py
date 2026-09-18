"""Module-level MoE block comparison, TT vs CPU, on real activations.

Chisel cannot validate the MoE chain on this model: every op in it fails for a
tool reason, not a hardware one --
  sparse_matmul           golden does a DENSE matmul ("sparsity is applied at
                          runtime on device"), so the routed-vs-all-experts
                          disagreement is by construction (PCC 0.59-0.89 on the
                          gate_up matmul, 0.9999 on the down matmul whose input
                          already carries the device's zeros)
  all_to_all_combine      golden shape model assumes the 32-token pad, not 256
  all_to_all_dispatch     permutation-dependent layout
  moe_expert_token_remap  golden returns uint16, device bfloat16 -> PCC 0
  topk                    PCC over categorical indices
So the MoE -- the largest untested block, running in all 30 layers -- is checked
here at module level instead: router + experts, device (expert-parallel tt_moe
across the 8-way mesh) vs CPU (the checkpoint's own expert loop).

Inputs are the REAL per-layer activations, captured from a CPU encoder forward
with a pre-hook on each layer's pre_feedforward_layernorm_2 -- which is exactly
`hidden_states_flat` in modeling_diffusion_gemma.py:625. Cached to disk so the
26B CPU forward is paid once.

  pytest -svv tests/torch/models/diffusiongemma/test_diffusiongemma_moe_block.py
  DIFFGEMMA_MOE_LAYERS=0,14,29 selects which layers to compare.
"""

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from infra.utilities.torch_multichip_utils import get_mesh

from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.diffusiongemma.pytorch import loader as dg_loader

CACHE = Path(os.environ.get("DIFFGEMMA_INPUT_CACHE", "diffusiongemma_vision_logs/inputs"))
VARIANT_NAME = os.environ.get("DIFFGEMMA_MOE_VARIANT", "ENCODER_IMAGE")
LAYERS = [int(x) for x in os.environ.get("DIFFGEMMA_MOE_LAYERS", ",".join(str(i) for i in range(30))).split(",")]
PROMPT_TOKENS = int(os.environ.get("DIFFGEMMA_PROMPT_TOKENS", "0"))

_EV = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_CF = PccConfig()


def _pcc(a, b):
    # float64 on both sides: the shipped _pcc runs float32 and can exceed 1.0
    return float(
        _EV._compare_pcc(
            a.detach().to("cpu").double(), b.detach().to("cpu").double(), _CF
        )
    )


def _stats(name, t):
    f = t.detach().float()
    print(
        f"[{name}] shape={tuple(t.shape)} dtype={t.dtype} mean={f.mean():.6f} "
        f"std={f.std():.6f} nan={int(f.isnan().sum())} inf={int(f.isinf().sum())}",
        flush=True,
    )


class MoEBlock(torch.nn.Module):
    """router + experts, exactly as modeling_diffusion_gemma.py:625-630 wires them.

    The router reads the UN-normed residual; the experts read the normed one.
    """

    def __init__(self, layer):
        super().__init__()
        self.norm = layer.pre_feedforward_layernorm_2
        self.router = layer.router
        self.experts = layer.experts

    def forward(self, hidden_states_flat):
        _, top_k_weights, top_k_index = self.router(hidden_states_flat)
        out = self.experts(self.norm(hidden_states_flat), top_k_index, top_k_weights)
        # return the routing decision too: a top-8-of-128 selection is DISCRETE, so
        # a ~1e-3 perturbation near a decision boundary flips an expert and changes
        # the output by far more than 1e-3. That is the amplification hypothesis.
        return out, top_k_index, top_k_weights


@pytest.mark.model_test
@pytest.mark.large
@pytest.mark.llmbox
def test_moe_block_tt_vs_cpu():
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

        variant = getattr(ModelVariant, VARIANT_NAME)
        is_image = "IMAGE" in VARIANT_NAME
        enable_spmd()
        register_tt_moe_backend()
        xr.set_device_type("TT")
        torch.manual_seed(0)

        CACHE.mkdir(parents=True, exist_ok=True)
        suffix = f"-{PROMPT_TOKENS}tok" if PROMPT_TOKENS else ""
        act_path = CACHE / f"moe_acts_{variant.value}{suffix}.pt"

        # ---- CPU side: golden block + real per-layer activations ----
        cl = ModelLoader(variant)
        cpu_enc = cl.load_model(dtype_override=torch.bfloat16)
        cpu_enc.eval()  # experts left on the checkpoint's own impl -> true reference

        if act_path.exists():
            acts = torch.load(act_path, map_location="cpu", weights_only=False)
            print(f"[acts] reused {act_path}", flush=True)
        else:
            inp_path = CACHE / f"{variant.value}{suffix}.pt"
            if inp_path.exists():
                inputs = torch.load(inp_path, map_location="cpu", weights_only=False)
            else:
                if PROMPT_TOKENS:
                    from tests.torch.models.diffusiongemma._length_prompt import (
                        build_prompt,
                    )

                    prompt, got = build_prompt(cl, PROMPT_TOKENS)
                    print(f"[input] length-controlled prompt: {got} tokens", flush=True)
                    inputs = cl.load_inputs(dtype_override=torch.bfloat16, prompt=prompt)
                else:
                    inputs = cl.load_inputs(dtype_override=torch.bfloat16)
                torch.save(inputs, inp_path)
            acts, handles = {}, []
            for i, layer in enumerate(cpu_enc.language_model.layers):
                def hook(mod, args, idx=i):
                    acts[idx] = args[0].detach().clone()
                handles.append(
                    layer.pre_feedforward_layernorm_2.register_forward_pre_hook(hook)
                )
            kw = dict(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                mm_token_type_ids=inputs["mm_token_type_ids"],
            )
            if is_image:
                kw["pixel_values"] = inputs["pixel_values"]
                kw["image_position_ids"] = inputs["image_position_ids"]
            print("[acts] capturing real per-layer MoE inputs (CPU forward) ...", flush=True)
            with torch.no_grad():
                cpu_enc(**kw)
            for h in handles:
                h.remove()
            torch.save(acts, act_path)
            print(f"[acts] saved {act_path} ({len(acts)} layers)", flush=True)

        goldens = {}
        for li in LAYERS:
            x = acts[li]
            _stats(f"input L{li}", x)
            with torch.no_grad():
                goldens[li] = MoEBlock(cpu_enc.language_model.layers[li])(x)
        del cpu_enc

        # ---- TT side: same block, expert-parallel across the mesh ----
        tl = ModelLoader(variant)
        tt_enc = tl.load_model(dtype_override=torch.bfloat16)
        tl.config._experts_implementation = TT_MOE_BACKEND_NAME
        xla = xm.xla_device()
        mesh = get_mesh(*tl.get_mesh_config(xr.global_runtime_device_count()))
        tt_enc = tt_enc.to(xla)
        xs.set_global_mesh(mesh)
        for tensor, spec in tl.load_shard_spec(tt_enc).items():
            xs.mark_sharding(tensor, mesh, spec)

        print("\n  layer   block_PCC    slot_match   set_match   bad_wt_mass")
        results = []
        for li in LAYERS:
            block = torch.compile(MoEBlock(tt_enc.language_model.layers[li]), backend="tt")
            with torch.no_grad():
                out, idx, wts = block(acts[li].to(xla))
                xm.mark_step()
                host, idx_h, wts_h = out.to("cpu"), idx.to("cpu"), wts.to("cpu")

            g_out, g_idx, g_wts = goldens[li]
            p = _pcc(host, g_out)

            # ordered: same expert in the same slot
            slot = (idx_h == g_idx).float().mean().item()
            # set: same 8 experts regardless of order (ties can reorder legitimately)
            a = torch.sort(idx_h, dim=-1).values
            b = torch.sort(g_idx, dim=-1).values
            sset = (a == b).all(dim=-1).float().mean().item()
            # how much routing weight sits on experts the two sides disagree about --
            # a flipped expert carrying ~0 weight is harmless, one carrying real mass is not
            total = g_wts.float().abs().sum().item()
            mism = (torch.sort(idx_h, -1).values != torch.sort(g_idx, -1).values)
            bad = (torch.sort(g_wts.float(), -1, descending=True).values * mism.float()).abs().sum().item()
            results.append((li, p, slot, sset, bad / total if total else 0.0))
            print(f"  {li:>5}   {p:.8f}   {slot:9.6f}   {sset:9.6f}   {bad/total if total else 0:9.6f}", flush=True)

        print(f"\n  worst block PCC : {min(r[1] for r in results):.8f}")
        print(f"  worst slot match: {min(r[2] for r in results):.6f}")
        print(f"  worst set match : {min(r[3] for r in results):.6f}")
        print(f"  max bad wt mass : {max(r[4] for r in results):.6f}", flush=True)
