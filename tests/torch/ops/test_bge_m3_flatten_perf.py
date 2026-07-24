# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Single-process device-perf microbench for the BGE-m3 flatten-inputs regression
(tt-xla issue #5756).

PR #5700 (`mmanzoor/vllm-flatten-inputs`) makes the vLLM pooling runner reshape the
encoder's `[N, T, H]` activations to a flat `[N*T, H]` token stream at the model-call
boundary. For a BERT/RoBERTa encoder like bge-m3 the ONLY thing that changes is tensor
**rank**: the same projection / FFN matmuls and LayerNorms run rank-2 instead of rank-3.
The attention SDPA is provably identical between the two modes (activations are reshaped
back to `[num_users, T, heads, hd]` regardless), so it is deliberately omitted here — it
would be byte-identical in both variants and cancel in the diff.

This reproduces the rank-sensitive op set of an XLMRobertaLayer at bge-m3 dimensions
(H=1024, FFN=4096, gelu, post-LN) and runs it directly through the `tt` torch-xla backend
in a SINGLE process, so `tracy` can capture per-op device timings. (The vLLM path can't be
traced cleanly: multiprocessing runs the model in an EngineCore child tracy doesn't
capture, and the in-process engine is pathologically slow under tracy instrumentation.)

The weight matrices are passed as graph INPUTS (not nn.Parameters) so the tt pipeline does
NOT const-eval them — const-eval of the large weight tensors under tracy instrumentation is
pathologically slow (20+ min); with weights-as-inputs the compile is quick, like the
weightless June-25 topk microbench. The matmul kernels exercised are identical either way.

Profile each variant and diff the summed `DEVICE FW DURATION [ns]`:

    BENCH_FLAT=0 tracy -p -r --sync-host-device -n bge_batched \
        -m pytest -svv tests/torch/ops/test_bge_m3_flatten_perf.py
    BENCH_FLAT=1 tracy -p -r --sync-host-device -n bge_flattened \
        -m pytest -svv tests/torch/ops/test_bge_m3_flatten_perf.py

Env knobs:
    BENCH_FLAT    0 = rank-3 [N, T, H] (batched)  | 1 = rank-2 [N*T, H] (flattened)
    BENCH_BATCH   N   (default 32)
    BENCH_T       T   (default 64, = bge-m3 test max_model_len)
    BENCH_LAYERS  times to loop the layer op set (default 4, to average fixed overhead)
"""
import os

import pytest
import torch
import torch.nn.functional as F
from infra import Framework, run_op_test
from utils import Category

from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig

# --- bge-m3 (XLMRobertaModel) dimensions (override via env for tractable compile) ---
H = int(os.environ.get("BENCH_H", "1024"))  # hidden_size (bge-m3: 1024)
FFN = int(os.environ.get("BENCH_FFN", str(4 * H)))  # intermediate_size (bge-m3: 4096)


class _RankSensitiveEncoder(torch.nn.Module):
    """The rank-sensitive (non-SDPA) ops of an XLMRobertaLayer, bge-m3 post-LN, looped
    `num_layers` times. Weights arrive as inputs (see module docstring), so nothing is
    const-eval'd. q/k/v projections are kept live via a cheap elementwise combine that
    stands in for the (rank-normalized, identical) attention context.

    All ops act on the trailing H dim, so this is agnostic to whether `x` is rank-3
    [N,T,H] or rank-2 [N*T,H] — exactly the difference PR #5700 introduces.
    """

    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers

    def forward(self, x, wq, wk, wv, wo, w_up, w_down, g1, b1, g2, b2):
        for _ in range(self.num_layers):
            q = F.linear(x, wq)  # [*,H] @ [H,H]
            k = F.linear(x, wk)
            v = F.linear(x, wv)
            ctx = v + F.silu(q) * torch.sigmoid(k)  # elementwise, rank-agnostic
            attn = F.linear(ctx, wo)  # [*,H] @ [H,H]
            x = F.layer_norm(x + attn, (H,), g1, b1, eps=1e-5)
            inter = F.gelu(F.linear(x, w_up))  # [*,H] @ [H,4H]
            out = F.linear(inter, w_down)  # [*,4H] @ [4H,H]
            x = F.layer_norm(x + out, (H,), g2, b2, eps=1e-5)
        return x


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_bge_m3_flatten_perf():
    flat = os.environ.get("BENCH_FLAT", "0") == "1"
    batch = int(os.environ.get("BENCH_BATCH", "32"))
    seq = int(os.environ.get("BENCH_T", "64"))
    layers = int(os.environ.get("BENCH_LAYERS", "4"))
    dtype = torch.bfloat16

    if flat:
        shape = (batch * seq, H)  # flattened token stream
    else:
        shape = (batch, seq, H)  # batched [N, T, H]

    print(
        f"\n[bge-m3 flatten-perf] flat={flat} shape={shape} layers={layers} "
        f"(H={H}, FFN={FFN})"
    )

    def w(*s):
        return torch.randn(*s, dtype=dtype)

    model = _RankSensitiveEncoder(layers)
    inputs = [
        torch.randn(*shape, dtype=dtype),  # x
        w(H, H),  # wq  (F.linear weight is [out, in])
        w(H, H),  # wk
        w(H, H),  # wv
        w(H, H),  # wo
        w(FFN, H),  # w_up   H -> 4H
        w(H, FFN),  # w_down 4H -> H
        w(H),  # g1 (layernorm weight)
        w(H),  # b1
        w(H),  # g2
        w(H),  # b2
    ]

    # Deep bf16 matmul stack drifts vs fp32 CPU golden; relax PCC. Correctness isn't the
    # goal here — the device execution (which tracy profiles) is.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.90))

    run_op_test(
        model,
        inputs,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )
