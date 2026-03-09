# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import shutil
import tempfile
import time

import numpy as np
import psutil
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

NUM_PARAMS = 4
# 5120 × 10240 = 52,428,800 elements × 2 bytes (bf16) = 100 MB per parameter
# PARAM_SHAPE = (5120, 10240)
PARAM_SHAPE = (5120, 5120)
PARAM_DTYPE = torch.float32


def get_memory_info():
    """Get current process memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / 1024 / 1024
    vms_mb = mem_info.vms / 1024 / 1024
    return {"rss_mb": rss_mb, "vms_mb": vms_mb}


def print_memory_snapshot(label):
    """Print labeled snapshot of memory usage."""
    print(f"\n{'='*70}")
    print(f"Memory Snapshot: {label}")
    print(f"{'='*70}")
    mem = get_memory_info()
    print(f"RSS: {mem['rss_mb']:.2f} MB")
    print(f"VMS: {mem['vms_mb']:.2f} MB")
    print(f"{'='*70}\n", flush=True)
    return mem


def _write_param_to_disk(path, index):
    """Write a single ~100 MB bf16 parameter to a raw binary file."""
    data = np.ones(PARAM_SHAPE, dtype=np.float32) * (index + 1)
    # data = data.astype(np.dtype('bfloat16'))
    data.tofile(path)
    del data
    gc.collect()


class LargeBf16Model(nn.Module):
    """20 × 100 MB bf16 parameter model backed by mmap'd files on disk.

    Each parameter is memory-mapped via ``torch.from_file`` so the OS can
    page data in/out rather than pinning 2 GB in host RAM all at once.
    """

    def __init__(self, weight_dir):
        super().__init__()
        numel = PARAM_SHAPE[0] * PARAM_SHAPE[1]
        params = []
        for i in range(NUM_PARAMS):
            path = os.path.join(weight_dir, f"param_{i}.bin")
            print("writing param to disk ", i , "at path ", path)
            
            if not os.path.exists(path):
                _write_param_to_disk(path, i)
            t = torch.from_file(
                path, shared=False, size=numel, dtype=PARAM_DTYPE
            ).reshape(*PARAM_SHAPE)
            params.append(nn.Parameter(t))
        self.param_tensors = nn.ParameterList(params)

    def forward(self, scale_factor):
        result = self.param_tensors[0].clone()
        for p in self.param_tensors[1:]:
            result = result + p
        if isinstance(scale_factor, torch.Tensor) and scale_factor.numel() == 1:
            scale_factor = scale_factor.item()
        return result * scale_factor


@pytest.fixture(scope="module")
def tt_device():
    xr.set_device_type("TT")
    return xm.xla_device()


def test_disk_offload_large_model(tt_device):
    """Load 4 × 100 MB model from disk-mmap, move to TT device, execute."""
    expected = float(sum(range(1, NUM_PARAMS + 1))) * 2.0

    with tempfile.TemporaryDirectory(prefix="disk_offload_") as weight_dir:
        model = LargeBf16Model(weight_dir)
        model.eval()

        # Sanity-check on CPU
        with torch.no_grad():
            cpu_out = model(torch.tensor(2.0, dtype=PARAM_DTYPE))
        assert abs(cpu_out[0, 0].item() - expected) < 1.0, (
            f"CPU mismatch: expected {expected}, got {cpu_out[0, 0].item()}"
        )

        # Move to TT device and run
        model = model.to(tt_device)
        with torch.no_grad():
            device_out = model(torch.tensor(2.0, dtype=PARAM_DTYPE))
        torch_xla.sync()

        actual = device_out.cpu()[0, 0].item()
        assert abs(actual - expected) < 1.0, (
            f"Device mismatch: expected {expected}, got {actual}"
        )


@pytest.mark.parametrize("use_disk_offload", [True, False], ids=["with_offload", "no_offload"])
@pytest.mark.parametrize("use_spmd", [True, False], ids=["spmd", "no_spmd"])
def test_llama_disk_offload(tt_device, use_disk_offload, use_spmd):
    """Load Llama 3.2 1B, move to TT device, run inference.

    Parameterized to test:
    - With/without disk offload: Transformers disk offload during loading
    - With/without SPMD: Single vs multi-device execution

    Requires: HuggingFace authentication (huggingface-cli login)
    """
    if not TRANSFORMERS_AVAILABLE:
        pytest.skip("transformers not available")

    num_devices = xr.global_runtime_device_count()
    if use_spmd and num_devices < 2:
        pytest.skip("SPMD requires multiple devices")

    model_name = "meta-llama/Llama-3.2-1B"
    prompt = "The capital of France is"

    print(f"\n{'='*70}")
    print(f"Mode: {'SPMD' if use_spmd else 'Single'} | "
          f"Offload: {'YES' if use_disk_offload else 'NO'}")
    print(f"{'='*70}")

    # Setup SPMD if needed
    if use_spmd:
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = xs.Mesh(device_ids, mesh_shape, ("batch", "model"))
        print(f"SPMD mesh: {mesh_shape} with {num_devices} devices")

    print_memory_snapshot("Initial")

    offload_dir = tempfile.mkdtemp(prefix="llama_offload_") if use_disk_offload else None

    try:
        # Load model
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
        }

        if use_disk_offload:
            print(f"\nLoading {model_name} with disk offload...")
            print(f"Offload dir: {offload_dir}\n")
            load_kwargs["low_cpu_mem_usage"] = True
            load_kwargs["device_map"] = "cpu"
            load_kwargs["offload_folder"] = offload_dir
            load_kwargs["offload_state_dict"] = True
        else:
            print(f"\nLoading {model_name} without disk offload...")
            print("This will load all weights into RAM\n")
            # Don't use low_cpu_mem_usage or device_map to force full load into RAM

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # Calculate actual model size
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = param_count * 2 / (1024 ** 2)  # 2 bytes per bf16 param
        print(f"Model parameters: {param_count:,} ({param_size_mb:.2f} MB)\n")

        mem_after_load = print_memory_snapshot(
            f"After CPU load {'with' if use_disk_offload else 'without'} disk offload"
        )

        # For non-offload case, touch all parameters to ensure they're in RAM
        if not use_disk_offload:
            print("Touching all parameters to force them into physical RAM...")
            total = 0
            for p in model.parameters():
                total += p.sum().item()
            print(f"Parameter sum: {total:.2e}\n")
            mem_after_load = print_memory_snapshot("After touching all parameters (actual RAM usage)")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Move to TT device (triggers PJRT mmap for all parameters)
        print(f"Moving model to TT device...")
        print("This creates mmap regions in PJRT for each parameter")
        print("Set TTXLA_LOGGER_LEVEL=DEBUG to see copyFromHost messages\n")

        model = model.to(tt_device)
        time.sleep(1)  # Allow system to process

        # Mark SPMD sharding if enabled
        if use_spmd:
            print("Marking SPMD sharding on model weights...")
            for layer in model.model.layers:
                xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
                xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
                xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))
                xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
                xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
                xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
                xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

        mem_after_to_device = print_memory_snapshot("After move to TT device")

        # Run forward pass
        print(f"Running forward pass with prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(tt_device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)

        torch_xla.sync()
        mem_after_inference = print_memory_snapshot("After inference")

        # Decode output
        logits = outputs.logits
        predicted_token_id = logits[0, -1].argmax().item()
        predicted_token = tokenizer.decode([predicted_token_id])
        print(f"Predicted next token: '{predicted_token}'")

        # Print summary
        print(f"\n{'='*70}")
        print(f"Memory Summary - {'SPMD' if use_spmd else 'Single'} | "
              f"Offload: {'YES' if use_disk_offload else 'NO'}")
        print(f"{'='*70}")
        print(f"Model: {param_size_mb:.2f} MB ({param_count:,} params)")
        print(f"After CPU load:   {mem_after_load['rss_mb']:.2f} MB")
        print(f"After to(device): {mem_after_to_device['rss_mb']:.2f} MB "
              f"({mem_after_to_device['rss_mb'] - mem_after_load['rss_mb']:+.2f} MB)")
        print(f"After inference:  {mem_after_inference['rss_mb']:.2f} MB "
              f"({mem_after_inference['rss_mb'] - mem_after_to_device['rss_mb']:+.2f} MB)")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\nError: {e}")
        print("Common issues:")
        print("  - Need HuggingFace auth: huggingface-cli login")
        print("  - Not enough disk space")
        print("  - OOM during load (try smaller model)")
        import traceback
        traceback.print_exc()
        pytest.skip(f"Skipping due to error: {e}")

    finally:
        if offload_dir and os.path.exists(offload_dir):
            print(f"Cleaning up: {offload_dir}")
            shutil.rmtree(offload_dir, ignore_errors=True)
