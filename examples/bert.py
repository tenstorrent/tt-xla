import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
import sys

from transformers import BertModel

from torch_xla.experimental import plugins
class TTPjrtPlugin(plugins.DevicePlugin):

  def library_path(self):
    return os.path.join(os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so")

plugins.register_plugin("TT", TTPjrtPlugin())

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch.export import export
from torch_xla.stablehlo import exported_program_to_stablehlo

def convert_ints_to_32(mod):
  for k, v in mod.state_dict().items():
    if v.dtype == torch.int64:
      mod.state_dict()[k] = v.to(torch.int32)
  for name, param in mod.named_parameters():
    if param.dtype == torch.int64:
      param.data = param.data.to(torch.int32)
  for name, buffer in mod.named_buffers():
    if buffer.dtype == torch.int64:
      buffer.data = buffer.data.to(torch.int32)


def bert_encoder():
    device = xm.xla_device()

    model = BertModel.from_pretrained("prajjwal1/bert-tiny").encoder
    model = model.to(device)
    input = torch.randn(1, 128, 128, device='cpu')
    input = input.to(device)
    out = model(input)
    print(out)


def bert():
  device = xm.xla_device()

  model = BertModel.from_pretrained("prajjwal1/bert-tiny").embeddings
  model = model.eval()
  convert_ints_to_32(model)
  model = model.to(device)
  # input = torch.randint(0, 10000, (1, 128), device='cpu') 
  input = torch.randint(0, 10000, (1, 128), device='cpu', dtype=torch.int32) 
  input = input.to(device)
  out = model(input)
  print(out)

os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
os.environ["TT_XLA_NUM_DEVICES"] = "2"

if __name__ == "__main__":
    bert()
