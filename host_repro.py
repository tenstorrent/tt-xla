import os
import numpy as np
import psutil
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM

# Choose what to test: "tensor" or "llama"
TEST_MODE = "tensor"


def setup_tt_environment():
    """Initialize TT device environment for multi-chip SPMD."""
    xr.set_device_type("TT")
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"

    # Additional setup for multichip
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["MESH_SHAPE"] = "8,4"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["DISABLE_NUMERIC_CC_TOKEN"] = "1"

    xr.use_spmd()


def create_mesh(num_devices):
    """Create a mesh for SPMD sharding."""
    device_ids = np.array(range(num_devices))
    mesh_shape = (8, 4)
    axis_names = ("data", "model")
    return xs.Mesh(
        device_ids=device_ids,
        mesh_shape=mesh_shape,
        axis_names=axis_names,
    )


def get_memory_gb():
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)


def get_available_ram_gb():
    """Get available system RAM in GB."""
    return psutil.virtual_memory().available / (1024 * 1024 * 1024)


def get_total_ram_gb():
    """Get total system RAM in GB."""
    return psutil.virtual_memory().total / (1024 * 1024 * 1024)


def get_model(model_name: str, device: torch.device):
    """Load a model and move it to device."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(torch.bfloat16)
    model.to(device)
    return model


def test_memory_allocation():
    setup_tt_environment()

    num_devices = xr.global_runtime_device_count()
    device = torch_xla.device()
    print(f"Devices: {num_devices}", flush=True)

    mesh = create_mesh(num_devices)
    print(f"Mesh: {mesh}", flush=True)

    print(f"\nTEST_MODE: {TEST_MODE}", flush=True)
    print(f"Total system RAM: {get_total_ram_gb():.2f} GB", flush=True)

    mem_before = get_memory_gb()
    avail_before = get_available_ram_gb()
    print(f"\nHost memory BEFORE: {mem_before:.2f} GB", flush=True)
    print(f"Available RAM BEFORE: {avail_before:.2f} GB", flush=True)

    if TEST_MODE == "tensor":
        tensor_size = 4096
        expected_size_gb = (tensor_size * tensor_size * 2) / (1024 * 1024 * 1024)  # bfloat16 = 2 bytes
        print(f"Expected tensor size: {expected_size_gb:.2f} GB", flush=True)

        A = torch.randn(tensor_size, tensor_size, dtype=torch.bfloat16)
        A = A.to(device)
        #xs.mark_sharding(A, mesh, (None, "model"))
        torch_xla.sync(wait=True)

    elif TEST_MODE == "llama":
        model_name = "meta-llama/Llama-3.2-1B"
        print(f"Loading model: {model_name}", flush=True)

        model = get_model(model_name, device)
        torch_xla.sync(wait=True)

    else:
        raise ValueError(f"Invalid TEST_MODE: {TEST_MODE}")

    mem_after = get_memory_gb()
    avail_after = get_available_ram_gb()
    print(f"\nHost memory AFTER: {mem_after:.2f} GB", flush=True)
    print(f"Available RAM AFTER: {avail_after:.2f} GB", flush=True)
    print(f"Process memory difference: {mem_after - mem_before:.2f} GB", flush=True)


if __name__ == "__main__":
    test_memory_allocation()
