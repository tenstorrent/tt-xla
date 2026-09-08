"""Layer-wise PCC probe: op-level fault vs accumulative decay.

Runs the ENCODER_IMAGE_ONLY variant on CPU and on TT with identical inputs and
compares, in execution order:
    pooler_output   vision tower -> embed_vision, i.e. the features the merge
                    scatters into the prompt embeddings
    hidden_states[k] output of encoder text layer k (30 of them)

READ:
  cliff at one k        -> that layer's op is at fault
  smooth monotonic fall -> accumulative decay through depth
  pooler clean but h[0] already low -> the masked_scatter merge or the
                                       bidirectional vision mask
"""
import inspect
import os

import torch

from tests.runner.requirements import RequirementsManager
from third_party.tt_forge_models.diffusiongemma.pytorch import loader as dg

with RequirementsManager.for_loader(inspect.getsourcefile(dg), framework="torch"):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr
    from infra.evaluators import PccConfig, TorchComparisonEvaluator
    from infra.evaluators.evaluation_config import ComparisonConfig
    from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
    from tt_torch.moe_backend import TT_MOE_BACKEND_NAME, register_tt_moe_backend

    from third_party.tt_forge_models.diffusiongemma.pytorch.loader import (
        ModelLoader, ModelVariant,
    )

    xr.set_device_type("TT")
    enable_spmd()
    register_tt_moe_backend()

    _EV = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
    _CF = PccConfig()
    def pcc(a, b):
        return float(_EV._compare_pcc(a.detach().to("cpu").float(),
                                      b.detach().to("cpu").float(), _CF))

    VARIANT = getattr(ModelVariant, os.environ.get("PROBE_VARIANT", "ENCODER_IMAGE_ONLY"))
    IS_IMAGE = "IMAGE" in os.environ.get("PROBE_VARIANT", "ENCODER_IMAGE_ONLY")
    print(f"variant: {VARIANT.value}  image={IS_IMAGE}", flush=True)

    class Probe(torch.nn.Module):
        """tensor-in / tensor-out so torch.compile can trace it."""
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, input_ids, attention_mask, mm_token_type_ids,
                    pixel_values=None, image_position_ids=None):
            out = self.enc(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
                output_hidden_states=True,
            )
            return tuple(out.hidden_states)

    # ---------- CPU golden ----------
    print("[1/3] loading CPU golden encoder ...", flush=True)
    cl = ModelLoader(VARIANT)
    cpu_enc = cl.load_model(dtype_override=torch.bfloat16)
    cpu_enc.eval()
    cl.config._experts_implementation = TT_MOE_BACKEND_NAME
    TARGET = int(os.environ.get("PROBE_TARGET_TOKENS", "0"))
    if TARGET:
        base = ("The sky appears blue because molecules in the air scatter blue "
                "light from the sun more than they scatter red light. ")
        words, prompt = base.split(), ""
        n = 0
        while True:
            trial = (prompt + " " + " ".join(words)).strip()
            t = cl.load_text_inputs(dtype_override=torch.bfloat16, prompt=trial)
            if t["input_ids"].shape[-1] >= TARGET:
                break
            prompt, n = trial, t["input_ids"].shape[-1]
        # trim back word-by-word to land as close to TARGET as possible
        w = trial.split()
        while len(w) > 1:
            t = cl.load_text_inputs(dtype_override=torch.bfloat16, prompt=" ".join(w))
            if t["input_ids"].shape[-1] <= TARGET:
                break
            w = w[:-1]
        inputs = cl.load_text_inputs(dtype_override=torch.bfloat16, prompt=" ".join(w))
        print(f"      length-controlled TEXT prompt: {inputs['input_ids'].shape[-1]} tokens "
              f"(target {TARGET})", flush=True)
    else:
        inputs = cl.load_inputs(dtype_override=torch.bfloat16)
    args = [inputs["input_ids"], inputs["attention_mask"],
            inputs["mm_token_type_ids"]]
    if IS_IMAGE:
        args += [inputs["pixel_values"], inputs["image_position_ids"]]
    args = tuple(args)
    with torch.no_grad():
        golden = Probe(cpu_enc)(*args)
    print(f"      golden tensors: {len(golden)} "
          f"(embeds + {len(golden)-1} layers)", flush=True)
    for i, g in enumerate(golden[:3]):
        print(f"      [{i}] {tuple(g.shape)} {g.dtype}", flush=True)
    del cpu_enc

    # ---------- TT ----------
    print("[2/3] loading + sharding TT encoder ...", flush=True)
    tl = ModelLoader(VARIANT)
    tt_enc = tl.load_model(dtype_override=torch.bfloat16)
    tl.config._experts_implementation = TT_MOE_BACKEND_NAME
    xla = xm.xla_device()
    mesh = get_mesh(*tl.get_mesh_config(xr.global_runtime_device_count()))
    tt_enc = tt_enc.to(xla)
    xs.set_global_mesh(mesh)
    for t, spec in tl.load_shard_spec(tt_enc).items():
        xs.mark_sharding(t, mesh, spec)
    tt_probe = torch.compile(Probe(tt_enc), backend="tt")
    dev_args = tuple(a.to(xla) for a in args)
    with torch.no_grad():
        got = tt_probe(*dev_args)
    xm.mark_step()

    # ---------- compare ----------
    print("[3/3] per-stage PCC (execution order)\n", flush=True)
    print(f"  {'stage':>22s}  {'pcc':>10s}   delta")
    prev = None
    rows = []
    for i, (g, t) in enumerate(zip(golden, got)):
        name = "embeds(post-merge)" if i == 0 else f"text layer {i-1:02d}"
        p = pcc(t, g)
        d = "" if prev is None else f"{p - prev:+.5f}"
        print(f"  {name:>22s}  {p:10.6f}   {d}", flush=True)
        rows.append((name, p))
        prev = p
    worst_step = min(
        ((rows[i][0], rows[i][1] - rows[i-1][1]) for i in range(1, len(rows))),
        key=lambda x: x[1],
    )
    print(f"\n  final: {rows[-1][1]:.6f}")
    print(f"  largest single-stage drop: {worst_step[0]} ({worst_step[1]:+.5f})")
