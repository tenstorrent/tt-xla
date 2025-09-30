# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import copy
import tokenizers
import torch
import torch.nn as nn
import torch_xla
from infra.comparators.torch_comparator import TorchComparator
from transformers import AutoModelForCausalLM, AutoConfig

from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

# from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

import torch
import torch.nn as nn
import torch.nn.functional as nnF


def pcc(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int | None = None,
    eps: float = 1e-8,
    keepdim: bool = False,
) -> torch.Tensor:
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape=} vs {y.shape=}")

    x = x.to(torch.float32)
    y = y.to(torch.float32)

    if dim is None:
        x = x.reshape(-1)
        y = y.reshape(-1)
        dim = 0

    xm = x - x.mean(dim=dim, keepdim=True)
    ym = y - y.mean(dim=dim, keepdim=True)

    num = (xm * ym).sum(dim=dim, keepdim=keepdim)
    den = torch.sqrt(
        xm.pow(2).sum(dim=dim, keepdim=keepdim)
        * ym.pow(2).sum(dim=dim, keepdim=keepdim)
        + eps
    )

    r = num / den
    return r


def test_sparse_moe_block_mixtral():
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    cfg = AutoConfig.from_pretrained(model_id)
    # os.environ["PT_XLA_DEBUG_LEVEL"] = "2"

    moe0 = MixtralSparseMoeBlock(cfg)
    moe0.eval()
    moe0_cpu = copy.deepcopy(moe0.to(device="cpu", dtype=torch.float32))

    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    # torch._dynamo.config.capture_scalar_outputs = True
    moe0 = moe0.to(device="xla", dtype=torch.bfloat16)

    # mod = torch.compile(moe0, backend="tt", fullgraph=True, dynamic=False)
    mod = moe0

    x = torch.randn(2, 8, cfg.hidden_size, dtype=torch.bfloat16)
    with torch_xla.experimental.eager_mode_context(True):
        y, logits = mod(x.to("xla"))

    x_cpu = x.to(device="cpu", dtype=torch.float32)

    print(y.shape)  # torch.Size([2, 4, cfg.hidden_size])
    print(logits.shape)  # torch.Size([2*4, cfg.num_local_experts])
    print("done")
    print(y)
    print(logits)

    y_cpu, logits_cpu = moe0_cpu(x_cpu)
    print("CPU output:")
    print(y_cpu)
    print(logits_cpu)

    print(f'pcc of outputs: {pcc(y.to("cpu"), y_cpu)}')
    print(f'pcc of logits: {pcc(logits.to("cpu"), logits_cpu)}')


def test_sparse_moe_block_qwen3():
    # working with transformers 4.52.4. But not with latest one.
    model_id = "Qwen/Qwen3-30B-A3B"
    cfg = AutoConfig.from_pretrained(model_id)
    # os.environ["PT_XLA_DEBUG_LEVEL"] = "2"

    moe0 = Qwen3MoeSparseMoeBlock(cfg)
    moe0.eval()
    moe0_cpu = copy.deepcopy(moe0.to(device="cpu", dtype=torch.float32))

    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    # torch._dynamo.config.capture_scalar_outputs = True
    moe0 = moe0.to(device="xla", dtype=torch.bfloat16)

    mod = torch.compile(moe0, backend="tt", fullgraph=True, dynamic=False)

    x = torch.randn(2, 8, cfg.hidden_size, dtype=torch.bfloat16)
    y, logits = mod(x.to("xla"))

    x_cpu = x.to(device="cpu", dtype=torch.float32)

    print(y.shape)  # torch.Size([2, 4, cfg.hidden_size])
    print(logits.shape)  # torch.Size([2*4, cfg.num_local_experts])
    print("done")
    print(y)
    print(logits)

    y_cpu, logits_cpu = moe0_cpu(x_cpu)
    print("CPU output:")
    print(y_cpu)
    print(logits_cpu)

    print(f'pcc of outputs: {pcc(y.to("cpu"), y_cpu)}')
    print(f'pcc of logits: {pcc(logits.to("cpu"), logits_cpu)}')
