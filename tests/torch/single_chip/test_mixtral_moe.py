import os
import torch
import torch.nn as nn
import tt_torch
from infra.comparators.torch_comparator import TorchComparator
from transformers import AutoModelForCausalLM, AutoConfig
# from huggingface_hub import snapshot_download
# local_dir = snapshot_download(
#     "mistralai/Mixtral-8x7B-v0.1",
#     local_files_only=True  # 네트워크 없이 로컬 캐시만 사용
# )

from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import torch
import torch.nn as nn
import torch.nn.functional as nnF

class TTMoEBlock(nn.Module):
    """
    Tensor-only Mixtral-like MoE block:
      - Router: Linear(H -> E) + softmax + topk
      - Experts: SwiGLU MLP per expert (batched-gathered parameters)
      - No Python data-dependent control flow (safe for torch.compile)
    """

    def __init__(self,
                 hidden_size: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int = 2,
                 dtype: torch.dtype = torch.bfloat16,
                 device: torch.device | None = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype

        # Router: W_gate [E, H] (as Linear: weight [E, H])
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype, device=device)

        # Experts packed as batched weights so we can gather by token expert-index
        # NOTE: Shapes:
        #   w_up:   [E, H, F]
        #   w_gate: [E, H, F]   (for SwiGLU)
        #   w_down: [E, F, H]
        self.w_up   = nn.Parameter(torch.empty(num_experts, hidden_size, ffn_dim, dtype=dtype, device=device))
        self.w_gate = nn.Parameter(torch.empty(num_experts, hidden_size, ffn_dim, dtype=dtype, device=device))
        self.w_down = nn.Parameter(torch.empty(num_experts, ffn_dim, hidden_size, dtype=dtype, device=device))

        # (optional) biases — Mixtral MLPs usually have no bias; keep for completeness (zeros)
        self.b_up   = nn.Parameter(torch.zeros(num_experts, ffn_dim, dtype=dtype, device=device))
        self.b_gate = nn.Parameter(torch.zeros(num_experts, ffn_dim, dtype=dtype, device=device))
        self.b_down = nn.Parameter(torch.zeros(num_experts, hidden_size, dtype=dtype, device=device))

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier init is a decent default
        for e in range(self.num_experts):
            nn.init.xavier_uniform_(self.w_up[e])
            nn.init.xavier_uniform_(self.w_gate[e])
            nn.init.xavier_uniform_(self.w_down[e])
        nn.init.xavier_uniform_(self.gate.weight)

    @torch.no_grad()
    def from_hf_block(self, hf_block) -> None:
        """
        NOTE (future, once HF model loads):
          - Copy gate weights:
              self.gate.weight.copy_(hf_block.gate.weight)         # [E, H]
          - Pack per-expert MLP weights:
              # up / gate (SwiGLU) and down
              for e in range(E):
                  self.w_up[e].copy_(hf_block.experts[e].w1.weight.T)     # adjust shape if needed
                  self.w_gate[e].copy_(hf_block.experts[e].w3.weight.T)   # adjust shape if needed
                  self.w_down[e].copy_(hf_block.experts[e].w2.weight.T)   # adjust shape if needed
          - Biases if present in HF (usually no bias in Mixtral MLPs)
        """
        pass

    def _flatten(self, x: torch.Tensor):
        """Accept [B,S,H] or [N,H]; return (x2d[N,H], (B,S) or None)"""
        if x.dim() == 3:
            b, s, h = x.shape
            return x.reshape(b * s, h), (b, s)
        elif x.dim() == 2:
            return x, None
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.shape}")

    def _unflatten(self, y2d: torch.Tensor, shape_bs):
        """Inverse of _flatten"""
        if shape_bs is None:
            return y2d
        b, s = shape_bs
        return y2d.view(b, s, -1)

    def forward(self, hidden_states: torch.Tensor, *_, **__) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          hidden_states: [B,S,H] or [N,H] (dtype usually bf16)
        Returns:
          final_hidden_states: same shape as input
          router_logits: [N, E] (flattened time-batch)
        """
        x, bs_shape = self._flatten(hidden_states)  # x: [N,H]
        N, H = x.shape
        E, F = self.num_experts, self.ffn_dim
        assert H == self.hidden_size, "Hidden size mismatch"

        # Router
        #   logits: [N,E]
        #   weights/topk: weights [N,K], indices [N,K]
        router_logits = self.gate(x.to(self.dtype))                     # [N,E]
        routing_weights = torch.softmax(router_logits.to(torch.float32), dim=-1)  # use fp32 for softmax stability
        topk_val, topk_idx = torch.topk(routing_weights, k=self.top_k, dim=-1)  # [N,K], [N,K]
        # Normalize within top-k (optional but common in Mixtral)
        topk_val = (topk_val / topk_val.sum(dim=-1, keepdim=True)).to(self.dtype)  # [N,K], bf16

        # Collect per-token expert params via gather
        # For each k in 0..K-1, we gather weights for expert indices topk_idx[:, k]
        y_accum = torch.zeros(N, H, dtype=self.dtype, device=x.device)

        # Precompute token-as-batch forms
        x_1 = x.unsqueeze(1)  # [N,1,H]

        for k in range(self.top_k):
            e_idx = topk_idx[:, k]                        # [N]
            w_k   = topk_val[:, k].unsqueeze(1)           # [N,1]

            # Gather expert weights: [N,H,F], [N,H,F], [N,F,H]
            w_up_e   = self.w_up[e_idx]                   # [N,H,F]
            w_gate_e = self.w_gate[e_idx]                 # [N,H,F]
            w_down_e = self.w_down[e_idx]                 # [N,F,H]

            b_up_e   = self.b_up[e_idx]                   # [N,F]
            b_gate_e = self.b_gate[e_idx]                 # [N,F]
            b_down_e = self.b_down[e_idx]                 # [N,H]

            # Up & Gate (SwiGLU): u = x @ W_up; v = x @ W_gate
            # x_1: [N,1,H]; W_up_e: [N,H,F] -> [N,1,F] -> [N,F]
            u = torch.bmm(x_1, w_up_e).squeeze(1) + b_up_e        # [N,F]
            v = torch.bmm(x_1, w_gate_e).squeeze(1) + b_gate_e    # [N,F]
            phi = nnF.silu(v) * u                                    # [N,F]

            # Down: y = phi @ W_down  -> [N,H]
            y = torch.bmm(phi.unsqueeze(1), w_down_e).squeeze(1) + b_down_e  # [N,H]

            # Weight by router probability and accumulate across top-k
            y_accum = y_accum + (y * w_k)                          # [N,H]

        out = self._unflatten(y_accum, bs_shape)                    # [B,S,H] or [N,H]
        return out, router_logits.to(hidden_states.dtype)
    
def swap_mixtral_moe_blocks(model):
    for name, module in model.named_modules():
        if hasattr(module, "block_sparse_moe"):
            old = module.block_sparse_moe
            module.block_sparse_moe = TTMoEBlock.from_hf_block(old)
    return model

def load_mixtral_fullmodel():
    model_id = "mistralai/Mixtral-8x7B-v0.1"   # 이미 받아둔 로컬 경로
    offload_dir = "/tmp/mixtral_offload"
    os.makedirs(offload_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,        # 파라메터 메모리 절감
        low_cpu_mem_usage=True,            # peak 메모리 줄이기
        device_map="auto",                 # accelerate가 CPU/디스크로 자동 배치
        max_memory={"cpu": "50GiB"},       # cgroup 아래로 여유 잡기
        offload_folder=offload_dir,        # 남는 가중치는 디스크로
        offload_state_dict=True,           # state_dict도 디스크로
    )
    return model

# class MixtralRouter(nn.Module):
#     """Mixtral 라우팅만 수행 (게이트 -> softmax -> top-k 선택)"""
#     def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
#         super().__init__()
#         self.gate = nn.Linear(hidden_size, num_experts, bias=False)
#         self.top_k = top_k

#     @torch.no_grad()
#     def forward(self, hidden_states: torch.Tensor):
#         # hidden_states: [B, S, H]
#         logits = self.gate(hidden_states)                 # [B, S, E]
#         probs  = torch.softmax(logits.to(torch.float32), dim=-1)  # 안정성 위해 fp32
#         topk_w, topk_idx = torch.topk(probs, k=self.top_k, dim=-1)  # [B, S, K]
#         return topk_idx, topk_w, logits

# def test_mixtral_router():
#     # Mixtral 모델을 이미 받아둔 상태에서, 라우터만 떼와서 테스트
#     model_id = local_dir   # 이미 받아둔 로컬 경로
#     cfg = AutoConfig.from_pretrained(model_id)

#     full = load_mixtral_fullmodel()

#     # 예) 0번 레이어 라우터만 떼오기
#     router0 = MixtralRouter(
#         hidden_size=cfg.hidden_size,
#         num_experts=cfg.num_local_experts,         # 보통 8
#         top_k=getattr(cfg, "num_experts_per_tok", 2)
#     )
#     with torch.no_grad():
#         src_w = full.model.layers[0].block_sparse_moe.gate.weight
#         src_w = src_w.to(dtype=router0.gate.weight.dtype, device=router0.gate.weight.device)
#         router0.gate.weight.copy_(src_w)

#         if router0.gate.bias is not None and hasattr(full.model.layers[0].block_sparse_moe.gate, "bias"):
#             src_b = full.model.layers[0].block_sparse_moe.gate.bias
#             src_b = src_b.to(dtype=router0.gate.bias.dtype, device=router0.gate.bias.device)
#             router0.gate.bias.copy_(src_b)


# class MixtralRouter(nn.Module):
#     """
#     A simple router: Linear -> (top-k + softmax over top-k scores).
#     It only decides *where* to send each token, it does not execute experts.
#     """
#     def __init__(self, hidden_size: int, num_experts: int, bias: bool = False):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_experts = num_experts
#         self.gate = nn.Linear(hidden_size, num_experts, bias=bias)

#     @torch.no_grad()
#     def load_from_hf_router(self, hf_linear: nn.Linear):
#         """
#         Optional helper to copy weights from a HF router (gate) layer.
#         Use no_grad to avoid in-place op error on leaf variables that require grad.
#         """
#         self.gate.weight.copy_(hf_linear.weight)
#         if self.gate.bias is not None and hf_linear.bias is not None:
#             self.gate.bias.copy_(hf_linear.bias)

#     def forward(self, x: torch.Tensor):
#         """
#         x: [N, D] flattened tokens
#         returns:
#           logits: [N, E]
#         """
#         return self.gate(x)


# class ExpertFFN(nn.Module):
#     """
#     A lightweight expert MLP: Linear -> SiLU -> Linear
#     (You can swap GELU/SiLU, or implement SwiGLU-style experts if you want.)
#     """
#     def __init__(self, hidden_size: int, ffn_hidden_size: int):
#         super().__init__()
#         self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
#         self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

#     def forward(self, x: torch.Tensor):
#         return self.fc2(F.silu(self.fc1(x)))


# class SparseMoEBlock(nn.Module):
#     """
#     A minimal Mixtral-like MoE block:
#       - Router picks top-k experts per token.
#       - We run only those experts for the tokens assigned to them.
#       - We weight each expert's output by the router probability and sum.

#     Shapes:
#       input:  [B, T, D]
#       output: [B, T, D]
#     """
#     def __init__(
#         self,
#         hidden_size: int,
#         num_experts: int,
#         top_k: int = 2,
#         ffn_hidden_size: int | None = None,
#         bias_router: bool = False,
#     ):
#         super().__init__()
#         assert top_k >= 1 and top_k <= num_experts
#         self.hidden_size = hidden_size
#         self.num_experts = num_experts
#         self.top_k = top_k

#         if ffn_hidden_size is None:
#             # A common default is ~4x, but you can choose any width.
#             ffn_hidden_size = 4 * hidden_size

#         self.router = MixtralRouter(hidden_size, num_experts, bias=bias_router)
#         self.experts = nn.ModuleList(
#             [ExpertFFN(hidden_size, ffn_hidden_size) for _ in range(num_experts)]
#         )

#     @torch.no_grad()
#     def load_from_hf_moe_block(self, hf_block) -> None:
#         """
#         Optional helper:
#         Given a HF MoE block (e.g., `block_sparse_moe` in Mixtral),
#         copy router and expert weights.
#         This assumes compatible shapes.
#         """
#         # Router
#         self.router.load_from_hf_router(hf_block.gate)

#         # Experts (very model-specific; adapt mapping to your HF class if needed)
#         # For a simple ExpertFFN (fc1/fc2), map from HF expert MLP layers accordingly:
#         #   self.experts[e].fc1 <- hf_block.experts[e].w_up (or similar)
#         #   self.experts[e].fc2 <- hf_block.experts[e].w_down (or similar)
#         # NOTE: Mixtral uses SwiGLU; this simple FFN won't be a 1:1 mapping.
#         pass

#     def forward(self, hidden_states: torch.Tensor):
#         """
#         hidden_states: [B, T, D]
#         returns:
#           out: [B, T, D]
#           aux: dict with routing info (topk indices and probabilities)
#         """
#         B, T, D = hidden_states.shape
#         x = hidden_states.reshape(B * T, D)  # [N, D], N=B*T

#         # 1) Router logits and top-k selection
#         logits = self.router(x)                           # [N, E]
#         topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # both [N, k]
#         probs = F.softmax(topk_vals, dim=-1)              # [N, k], softmax only over top-k

#         # 2) Prepare an output buffer
#         y = torch.zeros_like(x)  # [N, D]

#         # 3) For each route (k), dispatch to that expert subset, run, and scatter-add
#         #    This is simple & readable; can be optimized later if needed.
#         for r in range(self.top_k):
#             route_expert = topk_idx[:, r]        # [N], long
#             route_prob   = probs[:, r]           # [N], float

#             # Loop over experts; gather tokens for expert `e`, run MLP, scatter-add
#             for e in range(self.num_experts):
#                 idx = (route_expert == e).nonzero(as_tuple=True)[0]  # 1-D indices into N
#                 if idx.numel() == 0:
#                     continue

#                 x_e = x.index_select(0, idx)                # [n_e, D]
#                 out_e = self.experts[e](x_e)                # [n_e, D]
#                 out_e = out_e * route_prob.index_select(0, idx).unsqueeze(1)  # weight

#                 # Accumulate per-token contributions over multiple routes
#                 y.index_add_(0, idx, out_e)

#         out = y.view(B, T, D)

#         aux = {
#             "topk_idx": topk_idx.view(B, T, self.top_k),
#             "router_probs": probs.view(B, T, self.top_k),
#         }
#         return out, aux
    
def test_sparse_moe_block():
    import os
    os.environ["TORCH_LOGS"] = "dynamic"
    os.environ["TORCHDYNAMO_EXTENDED_DEBUG_CPP"] = str(1)
    model_id = "mistralai/Mixtral-8x7B-v0.1"  
    cfg = AutoConfig.from_pretrained(model_id)

    # torch._dynamo.config.capture_dynamic_output_shape_ops = True
    # torch._dynamo.config.capture_scalar_outputs = True 

    # full = load_mixtral_fullmodel()

    print(torch._dynamo.list_backends())

    moe0 = MixtralSparseMoeBlock(cfg)
    # moe0 = swap_mixtral_moe_blocks(moe0)
    '''        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok'''
    
    # moe0 = TTMoEBlock(
    #     hidden_size=cfg.hidden_size,
    #     ffn_dim=cfg.intermediate_size,
    #     num_experts=cfg.num_local_experts,
    # )
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    moe0 = moe0.to(device="cpu", dtype=torch.bfloat16)
    
    # compiled = torch.compile(moe0, backend="eager", dynamic=True, fullgraph=True, )

    x = torch.randn(2, 4, cfg.hidden_size, dtype=torch.bfloat16)
    import importlib
    mod = importlib.import_module("torch.export._draft_export")
    from torch.export._draft_export import draft_export
    export = draft_export(moe0, (x,), )
    compiled = torch.compile(export, backend="eager", dynamic=True, fullgraph=True, )
    y, logits = compiled(x.to("xla"))
    
    print(y.shape)       # torch.Size([2, 4, cfg.hidden_size])
    print(logits.shape)  # torch.Size([2*4, cfg.num_local_experts])
    print("done")
    print(y)
    print(logits)