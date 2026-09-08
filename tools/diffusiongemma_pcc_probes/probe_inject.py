"""Decisive test of the seed-error-amplification claim.

Claim: the image path is not mis-executed; it merely ENTERS the 30-layer text
stack carrying ~0.0055 of error (its embeddings are computed by the vision tower
+ embed_vision, whereas text embeddings are an exact table lookup at PCC 1.0),
and the stack amplifies that seed ~22x to the observed 0.873.

Test: run the TT encoder twice on the same image inputs --
  (A) normally                       -> seed error present   (expect ~0.873)
  (B) with the CPU golden's merged embeddings injected via inputs_embeds, and
      pixel_values withheld so no re-merge happens -> seed error ZERO
If the claim holds, (B) recovers to roughly the text figure (~0.985).
If (B) still lands near 0.87, the claim is WRONG and image content genuinely
changes how the layers behave.

Also emits last_hidden_state (post final RMSNorm) to cross-check the probe
against the runner's number (0.8688), which compares that tensor, not the raw
last layer output.
"""
import inspect
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

    V = ModelVariant.ENCODER_IMAGE_ONLY

    class Normal(torch.nn.Module):
        def __init__(s, e): super().__init__(); s.e = e
        def forward(s, input_ids, attention_mask, mm_token_type_ids,
                    pixel_values, image_position_ids):
            o = s.e(input_ids=input_ids, attention_mask=attention_mask,
                    mm_token_type_ids=mm_token_type_ids,
                    pixel_values=pixel_values,
                    image_position_ids=image_position_ids,
                    output_hidden_states=True)
            return tuple(o.hidden_states) + (o.last_hidden_state,)

    class Injected(torch.nn.Module):
        """pixel_values withheld -> the merge block is skipped entirely, so the
        provided (already-merged, error-free) embeddings pass straight through."""
        def __init__(s, e): super().__init__(); s.e = e
        def forward(s, inputs_embeds, attention_mask, mm_token_type_ids):
            o = s.e(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    mm_token_type_ids=mm_token_type_ids,
                    output_hidden_states=True)
            return tuple(o.hidden_states) + (o.last_hidden_state,)

    print("[1/4] CPU golden ...", flush=True)
    cl = ModelLoader(V)
    cpu = cl.load_model(dtype_override=torch.bfloat16); cpu.eval()
    cl.config._experts_implementation = TT_MOE_BACKEND_NAME
    inp = cl.load_inputs(dtype_override=torch.bfloat16)
    args = (inp["input_ids"], inp["attention_mask"], inp["mm_token_type_ids"],
            inp["pixel_values"], inp["image_position_ids"])
    with torch.no_grad():
        golden = Normal(cpu)(*args)
    merged = golden[0]           # CPU post-merge embeddings: the error-free seed
    print(f"      golden {len(golden)} tensors; merged embeds {tuple(merged.shape)}", flush=True)
    del cpu

    print("[2/4] TT load + shard ...", flush=True)
    tl = ModelLoader(V)
    tt = tl.load_model(dtype_override=torch.bfloat16)
    tl.config._experts_implementation = TT_MOE_BACKEND_NAME
    xla = xm.xla_device()
    mesh = get_mesh(*tl.get_mesh_config(xr.global_runtime_device_count()))
    tt = tt.to(xla); xs.set_global_mesh(mesh)
    for t, sp in tl.load_shard_spec(tt).items():
        xs.mark_sharding(t, mesh, sp)

    print("[3/4] (A) normal TT run ...", flush=True)
    with torch.no_grad():
        a = torch.compile(Normal(tt), backend="tt")(*(x.to(xla) for x in args))
    xm.mark_step()

    print("[4/4] (B) injected-embeddings TT run ...", flush=True)
    with torch.no_grad():
        b = torch.compile(Injected(tt), backend="tt")(
            merged.to(xla), inp["attention_mask"].to(xla),
            inp["mm_token_type_ids"].to(xla))
    xm.mark_step()

    n = len(golden)
    print(f"\n  {'stage':>28s}  {'(A) normal':>11s}  {'(B) injected':>12s}")
    for i in (0, 1, 6, 11, 16, 21, 26, n - 2, n - 1):
        name = ("embeds(seed)" if i == 0 else
                "last_hidden_state(post-norm)" if i == n - 1 else
                f"text layer {i-1:02d}")
        print(f"  {name:>28s}  {pcc(a[i], golden[i]):11.6f}  {pcc(b[i], golden[i]):12.6f}", flush=True)
    print(f"\n  VERDICT: injected final = {pcc(b[n-1], golden[n-1]):.6f}")
    print(f"           normal   final = {pcc(a[n-1], golden[n-1]):.6f}")
    print("           text-path reference = 0.985059")
    print("  -> injected ~= text reference  => seed-error amplification CONFIRMED")
    print("  -> injected still ~0.87        => claim WRONG, content-dependent")
