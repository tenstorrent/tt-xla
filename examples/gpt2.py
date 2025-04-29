import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

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


def gpt2():
    device = xm.xla_device()

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    breakpoint()
    model = model.transformer.h.to(device)
    inp = torch.randint(0, 10000, (1, 128), device="cpu")
    out = model.generate(inp, max_length=16)
    print(tokenizer.decode(out[0]))



os.environ["PJRT_DEVICE"] = "TT"
os.environ["XLA_STABLEHLO_COMPILE"] = "1"
os.environ["TT_XLA_NUM_DEVICES"] = "2"

if __name__ == "__main__":
    gpt2()
