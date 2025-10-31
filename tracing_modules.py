import os
os.environ["LOGGER_LEVEL"] = "INFO"
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from typing import Any, List
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def compute_pcc(x: torch.Tensor, y: torch.Tensor):
        x_flat, y_flat = x.flatten(), y.flatten()
        vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        denom = vx.norm() * vy.norm()

        return torch.tensor(float("nan")) if denom == 0 else torch.clamp((vx @ vy) / denom, -1, 1)

def compare_tt_with_golden(module: nn.Module, inp: Any):
    # torch forward
    with torch.no_grad():
        if isinstance(inp, tuple):
            out = module(*inp)
        else:
            out = module(inp)
    
    # tt forward
    module.compile(backend='tt')
    device = xm.xla_device()
    
    module = module.to(device)
    if isinstance(inp, tuple):
        inp = tuple(t.to(device) for t in inp)
    else:
        inp = inp.to(device)
    
    with torch.no_grad():
        if isinstance(inp, tuple):
            out = module(*inp)
        else:
            out = module(inp)

    pcc = compute_pcc(out.cpu(), out.cpu())
    return pcc

def get_model_and_inputs():
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float16
    )
    model = vae.decoder
    model = model.to(torch.bfloat16)
    orig_input = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
    return model, (orig_input,)

def get_topo_sorted_leaf_modules(module: nn.Module) -> List[nn.Module]:
    """
    Generate a topologically sorted list of leaf modules from a given module.
    Returns a list of modules.
    """
    leaf_modules = []
    
    def _collect_leaf_modules(parent_module, prefix=''):
        has_children = False
        for name, child in parent_module.named_children():
            has_children = True
            full_name = f"{prefix}.{name}" if prefix else name
            # Recursively traverse children
            _collect_leaf_modules(child, full_name)
        
        # If no children, this is a leaf module
        if not has_children:
            leaf_modules.append(parent_module)
    
    _collect_leaf_modules(module)
    return leaf_modules

def capture_submodule_inputs_and_order(model, example_inputs):
    """
    Captures the leaf submodules and their inputs in topological order using forward hooks.
    
    Args:
        model (nn.Module): The PyTorch model.
        example_inputs (tuple): Example inputs to pass through the model (e.g., (torch.randn(1, input_size),)).
    
    Returns:
        list of tuples: Each tuple contains (module, inputs), where inputs is a tuple of tensors.
    """
    leaf_modules = get_topo_sorted_leaf_modules(model)
    captured = []
    hooks = []
    
    def capture_hook(mod, inp):
        captured.append((mod, inp))
    
    for mod in leaf_modules:
        hook = mod.register_forward_hook(lambda m, i, o: capture_hook(m, i))
        hooks.append(hook)
    
    # Perform the forward pass to capture the order and inputs
    with torch.no_grad():
        model(*example_inputs)
    
    # Remove the hooks
    for h in hooks:
        h.remove()
    
    return captured

def execute_step_by_step(captured):
    """
    Performs step-by-step forward execution on each captured submodule with its respective inputs.

    Args:
        captured (list of tuples): List of (module, inputs) tuples.
    """
    for idx, (mod, inp) in enumerate(captured):
        if isinstance(inp, tuple):
            input_shapes = [t.shape if isinstance(t, torch.Tensor) else str(t) for t in inp]
        else:
            input_shapes = [inp.shape if isinstance(inp, torch.Tensor) else str(inp)]

        print(f"Step {idx + 1}: Executing {type(mod).__name__} with input shapes {input_shapes}")
        
        pcc = compare_tt_with_golden(mod, inp)
        print(f"PCC: {pcc}")

if __name__ == "__main__":
    xr.set_device_type("TT")

    model, example_inputs = get_model_and_inputs()
    captured_list = capture_submodule_inputs_and_order(model, example_inputs)
    
    for idx, (mod, inp) in enumerate(captured_list):
        if isinstance(inp, tuple):
            input_shapes = [t.shape if isinstance(t, torch.Tensor) else str(t) for t in inp]
        else:
            input_shapes = [inp.shape if isinstance(inp, torch.Tensor) else str(inp)]
        print(f"Tuple {idx + 1}: Module {type(mod).__name__}, Input shapes {input_shapes}")
    
    execute_step_by_step(captured_list)