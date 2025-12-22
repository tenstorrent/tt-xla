import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def run_sdpa():
    xr.set_device_type("TT")
    import torch.nn.functional as F
    
    class SDPA(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, query, key, value):
            return F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)

    query = torch.randn(1, 1, 4096, 512, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 4096, 512, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 4096, 512, dtype=torch.bfloat16)

    model = SDPA()
    model = model.to(torch.bfloat16)
    model = model.eval()
    model.compile(backend='tt')
    device = xm.xla_device()
    model = model.to(device)
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    model.eval()
    with torch.no_grad():
        tt_output = model(query, key, value)
    tt_output = tt_output.cpu()
    print(tt_output)

    import time
    start_time = time.time()
    with torch.no_grad():
        tt_output = model(query, key, value)
    tt_output = tt_output.cpu()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    run_sdpa()