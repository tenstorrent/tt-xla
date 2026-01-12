# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test_with_random_inputs
from utils import Category
from loguru import logger


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name=" torch.nn.functional.scaled_dot_product_attention",
)
def test_SDPA_exp1():

    class SDPA(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,q,k,v,attn_bias ):
            
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias
            )
            
            return x

    model = SDPA()
    model = model.to(dtype=torch.bfloat16)
    
    q = (1, 12, 4096, 64)
    k = (1, 12, 4096, 64)
    v = (1, 12, 4096, 64)
    attn_bias = (1, 12, 4096, 4096)

    run_op_test_with_random_inputs(
        model, [q,k,v, attn_bias], dtype=torch.bfloat16, framework=Framework.TORCH
    )

@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name=" torch.nn.functional.scaled_dot_product_attention",
)
def test_SDPA_exp2():

    class SDPA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.num_heads = 12
            self.B = 1

        def forward(self,q,k,v,attn_bias,rel_w,rel_h ):
            
            rel_h = rel_h.view(
                self.B, self.num_heads, rel_h.size(1), rel_h.size(2), rel_h.size(3)
            )
            rel_w = rel_w.view(
                self.B, self.num_heads, rel_w.size(1), rel_w.size(2), rel_w.size(3)
            )
            attn_bias = (rel_h + rel_w).view(
                self.B, self.num_heads, rel_h.size(2), rel_h.size(3) * rel_w.size(4)
            )
            
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias
            )
            
            return x

    model = SDPA()
    model = model.to(dtype=torch.bfloat16)
    
    q = (1, 12, 4096, 64)
    k = (1, 12, 4096, 64)
    v = (1, 12, 4096, 64)
    attn_bias = (1, 12, 4096, 4096)
    rel_w = (12, 4096, 1, 64)
    rel_h = (12, 4096, 64, 1)

    run_op_test_with_random_inputs(
        model, [q,k,v, attn_bias,rel_w,rel_h], dtype=torch.bfloat16, framework=Framework.TORCH
    )


