import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

@torch.inference_mode()
def fill_cache():
    batch_size = 1
    num_GQA_heads = 8
    prefill_seqlen = 64
    max_cache_seqlen = 1024
    head_dim = 128

    class Prefill(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.key_states:torch.Tensor = torch.ones((batch_size, num_GQA_heads, prefill_seqlen, head_dim))

        def forward(self, cache_position, key_cache):
            key_cache.index_copy_(2, cache_position, self.key_states)
            return key_cache

    xr.set_device_type("TT")
    device = xm.xla_device()

    prefillModel:Prefill = Prefill()
    prefillModel.key_states = prefillModel.key_states.to(device)
    prefillModel = prefillModel.eval()

    prefillModel.compile(backend="tt")


    cache_positions = torch.arange(64).to(device)  # Match prefill_seqlen
    key_cache:torch.Tensor = torch.zeros((batch_size, num_GQA_heads, max_cache_seqlen, head_dim)).to(device)

    with torch.no_grad():
        output = prefillModel(cache_positions, key_cache)
        output = output.to("cpu")
        print(output)

    # print("Cache shape:", output.shape)

    # Verify the cache was filled correctly
    # print("First few cache values:", output[0, 0, :5, 0])  # Should be 1.0
    # print("Cache filling successful:", torch.all(output[0, 0, :64, :] == 1.0).item())


fill_cache()