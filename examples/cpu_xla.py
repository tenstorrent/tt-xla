import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_xla.core.xla_model as xm
import torch.nn as nn

def gpt2():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    inputs = tokenizer("This is a sample text from ", return_tensors="pt")
    device = xm.xla_device()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    out = model.generate(**inputs, max_length=16)
    print(tokenizer.decode(out[0]))

def llama():
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
    inputs = tokenizer("This is a sample text from ", return_tensors="pt")
    device = xm.xla_device()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    out = model.generate(**inputs, max_length=16)
    print(tokenizer.decode(out[0]))

def sanity():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    model = Basic()
    device = xm.xla_device()
    model = model.to(device)
    input = torch.randn(4, 4, device="cpu")
    input = input.to(device)
    out = model(input)
    print(out)

def control_flow():
    class ControlFlow(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            y = x - 1
            if y[0] < 7:
                return y/4
            else:
                return y*4

    model = ControlFlow()
    device = xm.xla_device()
    model = model.to(device)
    input = torch.ones(1, device="cpu")*3
    input = input.to(device)
    out = model(input)
    print(out)


def run_twice():
    class Basic(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            x = x + 7
            x = x * 7
            x = x + 1
            return x

    model = Basic()
    device = xm.xla_device()
    model = model.to(device)
    input = torch.ones(1, device="cpu")
    input = input.to(device)
    out = model(input)
    print(out)
    out = model(input)
    print(out)

if __name__ == "__main__":
    control_flow()
