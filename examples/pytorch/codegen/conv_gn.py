import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from torch import Tensor, nn

# Set up XLA runtime for TT backend
xr.set_device_type("TT")


class DetrMaskHeadSmallConv(nn.Module):
    def __init__(self, dim, context_dim):
        super().__init__()
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.lay5(x)
        x = self.gn5(x)
        return x
    
model = DetrMaskHeadSmallConv(dim=264,context_dim=256).to(torch.bfloat16)
model.eval()

x = torch.randn(100, 32, 200, 267 ,dtype=torch.bfloat16)
codegen_py(model, x, export_path="conv_gn_emitpy", export_tensors=False)