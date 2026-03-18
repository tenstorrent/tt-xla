"""
Minimal script to run model(**inputs) on TT device with 4x8 mesh using gpt_oss loader.
Uses only third_party/tt_forge_models/gpt_oss/pytorch/loader.py and PyTorch/torch_xla.
"""
import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


def setup_spmd():
    """Initialize SPMD mode in torch_xla."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    print("XLA SPMD environment configured.")


def run_gpt_oss_120b_4x8():
    # 1. Use loader to get model and inputs (both on CPU)
    loader = ModelLoader(variant=ModelVariant.GPT_OSS_120B)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs()

    # 2. CPU execution (before device setup)
    with torch.no_grad():
        cpu_output = model(**inputs)
    cpu_logits = cpu_output.logits.float()

    # 3. Set TT device and enable SPMD
    xr.set_device_type("TT")
    setup_spmd()

    # 4. Create 4x8 mesh (32 devices for Galaxy)
    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)
    print(f"Created mesh {mesh_shape} with {num_devices} devices")

    # 5. Move model and inputs to TT device
    device = torch_xla.device()
    model = model.to(device)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # 6. Apply shard specs from loader
    shard_specs = loader.load_shard_spec(model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    # 7. Compile and run on TT
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        tt_output = compiled_model(**inputs)

    tt_logits = tt_output.logits.cpu().float()
    print(f"Output logits shape: {tt_logits.shape}")

    # 8. PCC (using model runner's evaluator)
    comparator = TorchComparisonEvaluator(
        ComparisonConfig(pcc=PccConfig(), assert_on_failure=False)
    )
    result = comparator.evaluate(tt_logits, cpu_logits)
    pcc = result.pcc
    print(f"PCC: {pcc:.6f}")
    return tt_output, pcc


if __name__ == "__main__":
    run_gpt_oss_120b_4x8()
