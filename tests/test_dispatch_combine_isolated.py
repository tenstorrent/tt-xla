"""
Focused dispatch+combine testing — isolates dispatch/combine from sparse_matmul.

Tests:
  1. Baseline: Router + weighted sum (no dispatch/combine)
  2. Dispatch + per-expert learned output + combine (no matmul)
  3. Dispatch + einsum expert computation + combine

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_dispatch_combine_isolated.py 2>&1 | tee out_dc.txt
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt_torch.sparse_mlp import A2aSparseMLP, build_expert_mapping


def P(*a, **kw):
    print(*a, **kw, flush=True)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0:
        return 1.0
    a_mean = a - a.mean()
    b_mean = b - b.mean()
    num = (a_mean * b_mean).sum()
    den = (a_mean.norm() * b_mean.norm()).clamp(min=1e-12)
    return (num / den).item()


# ---- Shared infrastructure ----

class FakeRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_tok):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))
        self.bias = nn.Parameter(torch.zeros(num_experts))
        self.top_k = num_experts_per_tok

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        flat = hidden_states.view(B * S, H)
        logits = F.linear(flat, self.weight, self.bias)
        scores = F.softmax(logits, dim=-1)
        _, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        return scores, topk_indices


class FakeExperts(nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size * 2) * 0.02
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(num_experts, intermediate_size * 2)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))
        self.alpha = 1.702
        self.limit = 7.0


class FakeMLP(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_tok, intermediate_size):
        super().__init__()
        self.router = FakeRouter(hidden_size, num_experts, num_experts_per_tok)
        self.experts = FakeExperts(num_experts, hidden_size, intermediate_size)


def make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape,
             forced_experts=None, fix_mapping=False):
    """Create A2aSparseMLP with forced routing.
    fix_mapping: if True, use mesh_shape-aware expert_mapping that matches compound sharding."""
    torch.manual_seed(42)
    fake_mlp = FakeMLP(H, E, K, inter)

    class FakeConfig:
        hidden_size = H
        num_local_experts = E
        num_experts_per_tok = K

    mlp = A2aSparseMLP(
        fake_mlp,
        num_experts=E,
        num_experts_per_tok=K,
        num_devices=num_devices,
        dispatch_devices=dispatch_devices,
        cluster_axis=0,
        config=FakeConfig(),
    )
    if fix_mapping:
        # Fix expert_mapping to match compound sharding ("model", "batch") layout
        mlp.expert_mapping = build_expert_mapping(E, num_devices, mesh_shape=mesh_shape)
        P(f"  [fix_mapping] Using mesh_shape={mesh_shape} for expert_mapping")
    # Force routing to specific experts
    if forced_experts is None:
        forced_experts = list(range(K))
    with torch.no_grad():
        mlp.router.bias.fill_(-100.0)
        for i, e in enumerate(forced_experts):
            mlp.router.bias[e] = 100.0 + (K - i) * 10.0
    mlp.use_fused_remap = False
    return mlp


def get_mlp_shard_specs(mlp, extra_params=None):
    """Get shard specs for A2aSparseMLP parameters."""
    specs = {
        mlp.router.weight: (None, "batch"),
        mlp.router.bias: (None,),
        mlp.experts.gate_up_proj: (("model", "batch"), None, None),
        mlp.experts.gate_up_proj_bias: (("model", "batch"), None),
        mlp.experts.down_proj: (("model", "batch"), None, None),
        mlp.experts.down_proj_bias: (("model", "batch"), None),
    }
    if extra_params:
        specs.update(extra_params)
    return specs


# ---- Test 1: Baseline (no dispatch/combine) ----

class BaselineModule(nn.Module):
    """Router + weighted sum — no dispatch/combine, no experts.
    Each expert returns hidden_states unchanged (identity).
    Output = hidden_states * sum(topk_weights)."""

    def __init__(self, router, K):
        super().__init__()
        self.router = router
        self.K = K

    def forward(self, hidden_states):
        B, S, H = hidden_states.shape
        router_scores, router_indices = self.router(hidden_states)
        topk_weights = torch.gather(router_scores, dim=-1, index=router_indices)
        topk_weights = topk_weights.view(B, S, self.K)
        output = hidden_states * topk_weights.sum(dim=-1, keepdim=True)
        return output


# ---- Test 2: Dispatch + per-expert learned output + combine ----

class DispatchCombineBias(nn.Module):
    """Dispatch + combine with per-expert learned output vector.
    No matmul — each expert outputs a fixed [H]-dim vector for all tokens.
    Tests whether dispatch+combine correctly route/gather expert outputs."""

    def __init__(self, mlp, expert_value):
        super().__init__()
        self.mlp = mlp
        # expert_value: [E, H] — will be compound-sharded on E
        self.expert_value = expert_value

    def forward(self, hidden_states):
        mlp = self.mlp
        B, S, H = hidden_states.shape
        K = mlp.num_experts_per_tok
        E = mlp.num_experts

        router_scores, router_indices = mlp.router(hidden_states)
        x = hidden_states.view(B, 1, S, H)
        expert_indices = router_indices.view(B, 1, S, K)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x,
            expert_indices,
            mlp.expert_mapping,
            num_devices=mlp.dispatch_devices,
            cluster_axis=mlp.cluster_axis,
        )

        BD = dispatched.shape[1]

        # Expert "computation": expand per-expert value to [E, BD, S, H]
        # expert_value inherits compound sharding on E from shard_specs
        down_out = self.expert_value.view(E, 1, 1, H).expand(E, BD, S, H).contiguous()

        # Combine
        combined = torch.ops.tt.all_to_all_combine(
            down_out,
            metadata,
            mlp.expert_mapping,
            num_devices=mlp.dispatch_devices,
            cluster_axis=mlp.cluster_axis,
            num_experts_per_tok=K,
            output_shard_dim=1,
            expert_indices=expert_indices,
        )

        # Weighted sum
        topk_weights = torch.gather(router_scores, dim=-1, index=router_indices)
        topk_weights = topk_weights.view(B, S, K)
        topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)
        output = (combined * topk_weights).sum(dim=0)

        return output


# ---- Test 3: Dispatch + einsum + combine ----

class DispatchEinsumCombine(nn.Module):
    """Full MLP with einsum replacing sparse_matmul.
    Tests the complete dispatch → expert_computation → combine pipeline."""

    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states):
        mlp = self.mlp
        B, S, H = hidden_states.shape
        K = mlp.num_experts_per_tok
        E = mlp.num_experts

        router_scores, router_indices = mlp.router(hidden_states)
        x = hidden_states.view(B, 1, S, H)
        expert_indices = router_indices.view(B, 1, S, K)

        dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
            x,
            expert_indices,
            mlp.expert_mapping,
            num_devices=mlp.dispatch_devices,
            cluster_axis=mlp.cluster_axis,
        )

        BD = dispatched.shape[1]
        M = 32
        split_seq = S % M == 0 and S >= M
        if split_seq:
            dim_a, dim_b = BD, S // M
        else:
            dim_a, dim_b = BD // M, S

        if split_seq:
            hidden_4d = dispatched.view(BD, S // M, M, H)
        elif dim_b == 1:
            hidden_4d = dispatched.view(BD // M, 1, M, H)
        else:
            hidden_4d = dispatched.view(BD // M, M, S, H)
            hidden_4d = hidden_4d.permute(0, 2, 1, 3)

        # Gate+Up with einsum (compound-sharded E)
        gate_up_proj = mlp.experts.gate_up_proj  # [E, H, inter*2]
        gate_up_out = torch.einsum("abmh,ehn->abemn", hidden_4d, gate_up_proj)
        gate_up_out = gate_up_out + mlp.experts.gate_up_proj_bias.unsqueeze(1)

        # Activation
        inter = mlp.intermediate_size
        gate_out = gate_up_out[..., :inter]
        up_out = gate_up_out[..., inter:]
        gate_out = gate_out.clamp(max=mlp.limit)
        up_out = up_out.clamp(-mlp.limit, mlp.limit)
        glu = gate_out * torch.sigmoid(gate_out * mlp.alpha)
        activated = (up_out + 1) * glu

        # Down projection with einsum
        down_proj = mlp.experts.down_proj  # [E, inter, H]
        down_out = torch.einsum("abeml,elh->abemh", activated, down_proj)
        down_out = down_out + mlp.experts.down_proj_bias.unsqueeze(1)

        # Reshape for combine: [E, BD, S, H]
        if split_seq:
            down_out = down_out.permute(2, 0, 1, 3, 4).contiguous()
            down_out = down_out.view(E, BD, S, H)
        elif dim_b == 1:
            down_out = down_out.squeeze(1)
            down_out = down_out.permute(1, 0, 2, 3).contiguous()
            down_out = down_out.view(E, BD, H).unsqueeze(1)
        else:
            down_out = down_out.permute(2, 0, 3, 1, 4).contiguous()
            down_out = down_out.view(E, BD, S, H)

        # Combine
        decode_mode = dim_b == 1 and not split_seq
        combined = torch.ops.tt.all_to_all_combine(
            down_out,
            metadata,
            mlp.expert_mapping,
            num_devices=mlp.dispatch_devices,
            cluster_axis=mlp.cluster_axis,
            num_experts_per_tok=K,
            output_shard_dim=2 if decode_mode else 1,
            expert_indices=expert_indices,
        )

        # Weighted sum
        topk_weights = torch.gather(router_scores, dim=-1, index=router_indices)
        topk_weights = topk_weights.view(B, S, K)
        topk_weights = topk_weights.permute(2, 0, 1).unsqueeze(-1)
        output = (combined * topk_weights).sum(dim=0)

        return output


def run_test(label, model, input_tensor, mesh, shard_specs_fn):
    """Run model on CPU (inductor) and TT, compare PCC.
    shard_specs_fn: callable(model) -> dict[tensor, spec], called AFTER model.to(device)."""
    P(f"\n--- {label} ---")
    model = model.to(torch.float32)
    model.train()

    # CPU
    cpu_compiled = torch.compile(model, backend="inductor")
    with torch.set_grad_enabled(True):
        cpu_out = cpu_compiled(input_tensor)
    cpu_norm = cpu_out.float().norm().item()
    P(f"  [cpu] norm={cpu_norm:.4f}  shape={cpu_out.shape}")

    # TT
    tt_compiled = torch.compile(
        model,
        backend="tt",
        options={
            "tt_experimental_compile": False,
            "tt_enable_torch_fx_fusion_pass": False,
        },
    )
    device = torch_xla.device()
    model.to(device)

    shard_specs = shard_specs_fn(model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    tt_input = input_tensor.detach().clone().to(device)
    with torch.set_grad_enabled(True):
        tt_out = tt_compiled(tt_input)
    torch_xla.sync(wait=True)

    tt_out_cpu = tt_out.detach().to("cpu")
    tt_norm = tt_out_cpu.float().norm().item()
    P(f"  [tt]  norm={tt_norm:.4f}  shape={tt_out_cpu.shape}")

    fwd_pcc = pcc(cpu_out.detach(), tt_out_cpu)
    P(f"  PCC = {fwd_pcc:.6f}")
    return fwd_pcc


def main():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    xr.set_device_type("TT")
    torch_xla._XLAC._init_computation_client()

    n = xr.global_runtime_device_count()
    if n >= 32:
        mesh_shape = (4, 8)
    elif n >= 8:
        mesh_shape = (2, 4)
    else:
        raise RuntimeError(f"Need >=8 devices, got {n}")

    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))
    num_devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[0]
    P(f"[setup] {n} devices, mesh={mesh_shape}, dispatch_devices={dispatch_devices}")

    # Common params
    B, S, H, E, K, inter = 1, 32, 512, 32, 4, 512

    torch.manual_seed(123)
    input_tensor = torch.randn(B, S, H, dtype=torch.float32)

    # ===== Test 1: Baseline (no dispatch/combine) =====
    P("\n" + "=" * 60)
    P("TEST 1: Baseline — router + weighted hidden_states")
    P("=" * 60)
    mlp1 = make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape)
    model1 = BaselineModule(mlp1.router, K)

    def specs1_fn(m):
        return {
            m.router.weight: (None, "batch"),
            m.router.bias: (None,),
        }

    pcc1 = run_test("Baseline", model1, input_tensor, mesh, specs1_fn)

    # Shared shard_specs factory
    def make_dc_specs_fn(has_expert_value=False):
        def fn(m):
            mlp = m.mlp if hasattr(m, "mlp") else m
            specs = {
                mlp.router.weight: (None, "batch"),
                mlp.router.bias: (None,),
                mlp.experts.gate_up_proj: (("model", "batch"), None, None),
                mlp.experts.gate_up_proj_bias: (("model", "batch"), None),
                mlp.experts.down_proj: (("model", "batch"), None, None),
                mlp.experts.down_proj_bias: (("model", "batch"), None),
            }
            if has_expert_value:
                specs[m.expert_value] = (("model", "batch"), None)
            return specs
        return fn

    dc_specs = make_dc_specs_fn()
    dc_specs_ev = make_dc_specs_fn(has_expert_value=True)

    # Compound sharding ("model", "batch") on (4,8) mesh:
    # - FIXED mapping: expert e → device (e%4)*8 + (e//4)
    #   Column c has experts: c*4, c*4+1, c*4+2, c*4+3 (one per row)
    # - LINEAR mapping: expert e → device e
    #   Row 0 has experts 0-7, row 1 has 8-15, etc.
    #
    # dispatch/combine operate along rows (cluster_axis=0).
    # They can gather within a COLUMN (across rows), not across columns.
    # So selected experts should be in the SAME COLUMN for optimal routing.
    #
    # FIXED + [0,1,2,3]: all in col 0 (one per row) — BEST for row dispatch
    # FIXED + [0,8,16,24]: all in row 0, diff cols — BAD for row dispatch
    # LINEAR + [0,8,16,24]: (0,0),(1,0),(2,0),(3,0) = col 0 — BEST for row dispatch
    # LINEAR + [0,1,2,3]: (0,0),(0,1),(0,2),(0,3) = row 0 — BAD for row dispatch

    P("\n" + "=" * 60)
    P("TEST 2: FIXED mapping + experts [0,1,2,3] (col 0, diff rows — IDEAL)")
    P("=" * 60)
    mlp2 = make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape,
                     forced_experts=[0, 1, 2, 3], fix_mapping=True)
    torch.manual_seed(99)
    model2 = DispatchCombineBias(mlp2, nn.Parameter(torch.randn(E, H) * 0.1))
    pcc2 = run_test("FIXED + col-0 experts", model2, input_tensor, mesh, dc_specs_ev)

    P("\n" + "=" * 60)
    P("TEST 3: LINEAR mapping + experts [0,8,16,24] (col 0, diff rows — IDEAL)")
    P("=" * 60)
    mlp3 = make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape,
                     forced_experts=[0, 8, 16, 24], fix_mapping=False)
    torch.manual_seed(99)
    model3 = DispatchCombineBias(mlp3, nn.Parameter(torch.randn(E, H) * 0.1))
    pcc3 = run_test("LINEAR + col-0 experts", model3, input_tensor, mesh, dc_specs_ev)

    P("\n" + "=" * 60)
    P("TEST 4: FIXED mapping + experts [0,8,16,24] (row 0, diff cols — BAD)")
    P("=" * 60)
    mlp4 = make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape,
                     forced_experts=[0, 8, 16, 24], fix_mapping=True)
    torch.manual_seed(99)
    model4 = DispatchCombineBias(mlp4, nn.Parameter(torch.randn(E, H) * 0.1))
    pcc4 = run_test("FIXED + row-0 experts", model4, input_tensor, mesh, dc_specs_ev)

    P("\n" + "=" * 60)
    P("TEST 5: LINEAR mapping + experts [0,1,2,3] (row 0, diff cols — BAD)")
    P("=" * 60)
    mlp5 = make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape,
                     forced_experts=[0, 1, 2, 3], fix_mapping=False)
    torch.manual_seed(99)
    model5 = DispatchCombineBias(mlp5, nn.Parameter(torch.randn(E, H) * 0.1))
    pcc5 = run_test("LINEAR + row-0 experts", model5, input_tensor, mesh, dc_specs_ev)

    P("\n" + "=" * 60)
    P("TEST 6: FIXED + [0,1,2,3] + einsum (col 0, diff rows — IDEAL)")
    P("=" * 60)
    mlp6 = make_mlp(H, E, K, inter, num_devices, dispatch_devices, mesh_shape,
                     forced_experts=[0, 1, 2, 3], fix_mapping=True)
    model6 = DispatchEinsumCombine(mlp6)
    pcc6 = run_test("FIXED + col-0 + einsum", model6, input_tensor, mesh, dc_specs)

    # ===== Summary =====
    P(f"\n{'='*60}")
    P("SUMMARY:")
    P(f"  1. Baseline (no D/C):                    PCC = {pcc1:.6f}")
    P(f"  2. FIXED + [0,1,2,3] col-0 (IDEAL):     PCC = {pcc2:.6f}")
    P(f"  3. LINEAR + [0,8,16,24] col-0 (IDEAL):   PCC = {pcc3:.6f}")
    P(f"  4. FIXED + [0,8,16,24] row-0 (BAD):      PCC = {pcc4:.6f}")
    P(f"  5. LINEAR + [0,1,2,3] row-0 (BAD):       PCC = {pcc5:.6f}")
    P(f"  6. FIXED + [0,1,2,3] + einsum (IDEAL):   PCC = {pcc6:.6f}")
    P(f"{'='*60}")
    P("")
    P("IDEAL tests (2,3,6) should give PCC~1.0 if dispatch/combine work correctly.")
    P("BAD tests (4,5) should give lower PCC due to cross-column routing.")


if __name__ == "__main__":
    main()
