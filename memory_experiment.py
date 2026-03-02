# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.core.xla_builder import create_placeholder_tensor
import torch_xla
import os
import psutil
import time

import tempfile
import mmap
import gc


def get_memory_info():
    """Get current process memory usage information."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    # RSS: Resident Set Size - actual physical memory used
    # VMS: Virtual Memory Size - total virtual memory
    rss_mb = mem_info.rss / 1024 / 1024
    vms_mb = mem_info.vms / 1024 / 1024

    return {
        'rss_mb': rss_mb,
        'vms_mb': vms_mb,
        'rss_human': f"{rss_mb:.2f} MB",
        'vms_human': f"{vms_mb:.2f} MB"
    }


def get_mmap_regions():
    """Get information about memory-mapped regions of the current process."""
    pid = os.getpid()
    maps_file = f"/proc/{pid}/maps"

    if not os.path.exists(maps_file):
        return "Cannot read /proc/*/maps (not on Linux?)"

    # Count anonymous mmap regions (which our implementation uses)
    anon_regions = []
    total_anon_size = 0

    with open(maps_file, 'r') as f:
        for line in f:
            # Anonymous mappings don't have a file path or are marked with certain patterns
            parts = line.strip().split()
            if len(parts) >= 2:
                addr_range = parts[0]
                start, end = addr_range.split('-')
                start_addr = int(start, 16)
                end_addr = int(end, 16)
                size = end_addr - start_addr

                # Check if this is an anonymous mapping
                # Anonymous mappings typically don't have a pathname or are stack/heap
                is_anon = len(parts) < 6 or parts[-1].startswith('[') or parts[-1] == ''

                if is_anon and size > 1024 * 1024:  # Only show regions > 1MB
                    anon_regions.append({
                        'start': start,
                        'end': end,
                        'size_mb': size / 1024 / 1024,
                        'perms': parts[1] if len(parts) > 1 else 'unknown'
                    })
                    total_anon_size += size

    return {
        'regions': anon_regions,
        'total_size_mb': total_anon_size / 1024 / 1024,
        'count': len(anon_regions)
    }


def print_memory_snapshot(label):
    """Print a labeled snapshot of current memory usage and mmap regions."""
    print(f"\n{'='*70}")
    print(f"Memory Snapshot: {label}")
    print(f"PID: {os.getpid()}")
    print(f"{'='*70}")

    mem = get_memory_info()
    print(f"RSS (Physical Memory): {mem['rss_human']}")
    print(f"VMS (Virtual Memory):  {mem['vms_human']}")

    mmap_info = get_mmap_regions()
    if isinstance(mmap_info, dict):
        print(f"\nAnonymous mmap regions (>1MB): {mmap_info['count']}")
        print(f"Total anonymous mmap size: {mmap_info['total_size_mb']:.2f} MB")

        if mmap_info['count'] > 0:
            print("\nLargest regions:")
            sorted_regions = sorted(mmap_info['regions'], key=lambda x: x['size_mb'], reverse=True)
            for i, region in enumerate(sorted_regions[:5]):  # Show top 5
                print(f"  {i+1}. {region['start']}-{region['end']}: {region['size_mb']:.2f} MB ({region['perms']})")
    else:
        print(f"mmap info: {mmap_info}")
        
    print(f"{'='*70}\n", flush=True)
    return mem


class SimpleAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single parameter tensor for elementwise addition
        self.add_tensor = nn.Parameter(torch.ones(16, 16, dtype=torch.bfloat16))

    def forward(self, x):
        return x + self.add_tensor


def test_simple_add_diskbacked():
    """Test simple elementwise addition on TT device."""
    # Set device type to TT
    xr.set_device_type("TT")

    # Instantiate model
    torch.manual_seed(42)
    model = SimpleAddModel()

    # Put it in inference mode
    model = model.eval()

    # Get TT device
    device = xm.xla_device()

    print("Moving model to device...", flush=True)
    # Move model to device
    model = model.to(device)

    print(f"Model successfully moved to device: {device}")

    # Create 16x16 input
    
    # inputs = torch.randn(16, 16, dtype=torch.float32)
    # inputs.numpy().tofile('16x16xf32.pt')
    
    # need to specify size / dtype as this gives you back a flat tensor
    inputs = torch.from_file('16x16xf32.pt', dtype=torch.float32, size=256).reshape(16, 16).to(torch.bfloat16)
    storage = inputs.untyped_storage()
    print(f"Storage size: {storage.size()}")
    print(f"Storage data_ptr: {storage.data_ptr()} | filepath: {storage.filename}")
    print(f"Is storage memory-mapped: {storage.is_shared()}")  # Should be True for shared=True
    # inputs.numpy().tofile('16x16xbf16.pt')
    print(inputs.shape)
    print("Moving inputs to device...", flush=True)
    inputs = inputs.to(device)

    # Perform elementwise addition
    output = model(inputs)

    # print(f"Output device: {output.device}")
    # print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


def test_simple_add():
    """Test simple elementwise addition on TT device."""
    # Set device type to TT
    xr.set_device_type("TT")

    # Instantiate model
    torch.manual_seed(42)
    model = SimpleAddModel()

    # Put it in inference mode
    model = model.eval()

    # Get TT device
    device = xm.xla_device()

    print("Moving model to device...", flush=True)
    # Move model to device
    model = model.to(device)

    print(f"Model successfully moved to device: {device}")

    # Create 16x16 input
    inputs = torch.randn(16, 16, dtype=torch.float32)
    
    # need to specify size / dtype as this gives you back a flat tensor
    print(inputs.shape)
    print("Moving inputs to device...", flush=True)
    inputs = inputs.to(device)

    # Perform elementwise addition
    output = model(inputs)

    # print(f"Output device: {output.device}")
    # print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")

class SimpleAddModelPlaceholder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Single parameter tensor for elementwise addition
        self.add_tensor = create_placeholder_tensor(shape=(16, 16), dtype=torch.bfloat16)

    def forward(self, x):
        return x + self.add_tensor

def test_create_placeholder_tensor():
    """Test creating a placeholder tensor on TT device."""
    # Set device type to TT
    xr.set_device_type("TT")

    # Create a placeholder tensor
    placeholder = create_placeholder_tensor(shape=(16, 16), dtype=torch.bfloat16)
    print(f"Placeholder tensor: {placeholder.shape}")
    device = xm.xla_device()
    placeholder = placeholder.to(device)
    
    print("placeholder device: ", placeholder.device)
    
    model = SimpleAddModelPlaceholder()
    model.compile(backend="tt")
    
    model = model.to(device)

    output = model(placeholder)
    torch_xla.sync()



def test_simple_spmd():
    """Test simple elementwise addition on TT device."""
    # Set device type to TT
    import os
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    xr.use_spmd()
    # Set up SPMD and create a 1x2 device mesh.
    import torch_xla.distributed.spmd as xs
    import numpy as np
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.spmd import Mesh
    num_devices = xr.global_runtime_device_count()

    # Create a 1x2 mesh using 2 TT logical devices
    devices = np.array(range(num_devices))
    mesh = Mesh(devices, (1, num_devices), ("batch", "model"))

    print(f"SPMD mesh created: {mesh}")


    # Instantiate model
    torch.manual_seed(42)
    model = SimpleAddModel()

    # Put it in inference mode
    model = model.eval()

    # Get TT device
    device = xm.xla_device()

    print("Moving model to device...", flush=True)
    # Move model to device
    model = model.to(device)

    print(f"Model successfully moved to device: {device}")

    # Create 16x16 input
    
    inputs = torch.randn(16, 16, dtype=torch.float32)
    # inputs.numpy().tofile('16x16xf32.pt')
    
    # need to specify size / dtype as this gives you back a flat tensor
    print(inputs.shape)
    print("Moving inputs to device...", flush=True)
    inputs = inputs.to(device)
    
    print("Marking inputs for sharding...", flush=True)
    inputs = xs.mark_sharding(inputs, mesh, (None, "model"))
    # Perform elementwise addition
    print("Running model...", flush=True)
    output = model(inputs)

    # print(f"Output device: {output.device}")
    # print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")


def test_large_tensor_mmap():
    """Test memory-mapped buffer behavior with a large tensor.

    This test creates a ~100MB tensor and monitors memory usage and mmap regions
    at different stages:
    1. Before creating the tensor
    2. After creating the tensor (in CPU memory)
    3. After moving to TT device (should be in mmap'd region)
    4. After computation
    """
    print("\n" + "="*70)
    print("Testing Large Tensor Memory Mapping")
    print("="*70)

    # Set device type to TT
    xr.set_device_type("TT")

    # Take initial memory snapshot
    print_memory_snapshot("Initial (before tensor creation)")

    # Create a large tensor: 100MB = 100 * 1024 * 1024 bytes
    # Using float32 (4 bytes per element): 100MB / 4 = 26,214,400 elements
    # Let's use a nice shape: 5120 x 5120 = 26,214,400 elements = exactly 100MB
    print("Creating 100MB tensor (5120x5120 float32)...")
    large_tensor = torch.ones(5120, 5120, dtype=torch.float32)
    tensor_size_mb = large_tensor.element_size() * large_tensor.numel() / 1024 / 1024
    print(f"Tensor created: shape={large_tensor.shape}, dtype={large_tensor.dtype}, "
          f"size={tensor_size_mb:.2f} MB")

    # Take snapshot after tensor creation
    mem_after_create = print_memory_snapshot("After tensor creation (CPU)")

    # Get TT device
    device = xm.xla_device()
    print(f"TT device: {device}")

    # Move tensor to device - this should trigger mmap in PJRT copyFromHost
    print(f"\nMoving {tensor_size_mb:.2f} MB tensor to TT device...")
    print("(This should create an mmap'd buffer in PJRT)")
    large_tensor_device = large_tensor.to(device)

    # Give the system a moment to process
    time.sleep(0.5)

    # Take snapshot after moving to device
    mem_after_to_device = print_memory_snapshot("After moving to device (should see mmap)")

    # Perform a simple operation
    print("Performing computation: output = tensor + 1.0")
    output = large_tensor_device + 1.0

    # Force synchronization to ensure computation is done
    print("Synchronizing...")
    torch_xla.sync()

    # Take final snapshot
    mem_after_compute = print_memory_snapshot("After computation")

    # Print summary
    print("\n" + "="*70)
    print("Memory Usage Summary")
    print("="*70)
    print(f"Initial RSS:        {mem_after_create['rss_mb']:.2f} MB")
    print(f"After to(device):   {mem_after_to_device['rss_mb']:.2f} MB "
          f"(+{mem_after_to_device['rss_mb'] - mem_after_create['rss_mb']:.2f} MB)")
    print(f"After computation:  {mem_after_compute['rss_mb']:.2f} MB "
          f"(+{mem_after_compute['rss_mb'] - mem_after_to_device['rss_mb']:.2f} MB)")
    print(f"\nExpected behavior:")
    print(f"  - RSS increase after to(device) should be close to {tensor_size_mb:.2f} MB")
    print(f"    (this is the mmap'd buffer that holds a copy of the tensor)")
    print(f"  - Check the 'Anonymous mmap regions' sections above to verify")
    print(f"    that a ~{tensor_size_mb:.2f} MB region was created")
    print("="*70 + "\n")

    return output


def test_multiple_large_tensors_mmap():
    """Test memory-mapped buffer behavior with multiple large tensors.

    This test creates 10x 100MB tensors and monitors memory usage as they are
    moved to device one by one. This stresses the mmap implementation and shows
    how memory grows with multiple buffers.

    The test performs element-wise sum of all tensors to verify computation works.
    """
    print("\n" + "="*70)
    print("Testing Multiple Large Tensors Memory Mapping")
    print("="*70)

    # Set device type to TT
    xr.set_device_type("TT")

    # Take initial memory snapshot
    print_memory_snapshot("Initial (before any tensors)")

    num_tensors = 10
    tensor_shape = (5120, 5120)  # 100MB per tensor (float32)
    tensor_size_mb = 5120 * 5120 * 4 / 1024 / 1024  # 4 bytes per float32

    print(f"\nCreating {num_tensors} tensors of {tensor_size_mb:.2f} MB each...")
    print(f"Total size: {num_tensors * tensor_size_mb:.2f} MB")

    # Create all tensors on CPU first
    cpu_tensors = []
    for i in range(num_tensors):
        print(f"\n[CPU] Creating tensor {i+1}/{num_tensors}...", flush=True)
        # Use different values for each tensor so we can verify the sum
        tensor = torch.full(tensor_shape, fill_value=float(i+1), dtype=torch.float32)
        cpu_tensors.append(tensor)

    # Take snapshot after CPU tensor creation
    mem_after_cpu = print_memory_snapshot(f"After creating {num_tensors} tensors on CPU")

    # Get TT device
    device = xm.xla_device()
    print(f"\nTT device: {device}")

    # Move tensors to device one by one, taking snapshots
    device_tensors = []
    for i, cpu_tensor in enumerate(cpu_tensors):
        print(f"\n{'='*70}")
        print(f"[DEVICE] Moving tensor {i+1}/{num_tensors} to device...")
        print(f"         This should create ~{tensor_size_mb:.2f} MB mmap region")
        print(f"{'='*70}", flush=True)

        device_tensor = cpu_tensor.to(device)
        device_tensors.append(device_tensor)

        # Give system a moment to process
        time.sleep(0.3)

        # Take snapshot after each tensor
        mem = print_memory_snapshot(f"After moving tensor {i+1}/{num_tensors} to device")

        # Show incremental memory increase
        if i == 0:
            delta = mem['rss_mb'] - mem_after_cpu['rss_mb']
        else:
            # Compare to previous iteration - we don't have that, so compare to CPU baseline
            delta = mem['rss_mb'] - mem_after_cpu['rss_mb']

        print(f"RSS increase from CPU baseline: {delta:.2f} MB "
              f"(expected ~{(i+1) * tensor_size_mb:.2f} MB)", flush=True)

    # Take final snapshot before computation
    mem_before_compute = print_memory_snapshot(f"All {num_tensors} tensors on device (before compute)")

    # Perform computation: sum all tensors element-wise
    print(f"\n{'='*70}")
    print("Performing computation: summing all tensors element-wise...")
    print(f"{'='*70}", flush=True)

    # Start with first tensor, then add each subsequent one
    result = device_tensors[0]
    for i in range(1, len(device_tensors)):
        print(f"Adding tensor {i+1}...", flush=True)
        result = result + device_tensors[i]

    # Force synchronization
    print("\nSynchronizing...", flush=True)
    torch_xla.sync()

    # Take final snapshot
    mem_after_compute = print_memory_snapshot("After computation and sync")

    # Verify the result
    # Sum should be: 1 + 2 + 3 + ... + 10 = 55
    # Each element should equal sum(1..10) = 55
    expected_sum = sum(range(1, num_tensors + 1))
    print(f"\n{'='*70}")
    print("Computation Verification")
    print(f"{'='*70}")
    print(f"Expected sum value (per element): {expected_sum}")

    # Move result back to CPU to verify
    result_cpu = result.cpu()
    actual_value = result_cpu[0, 0].item()
    print(f"Actual value (at [0,0]): {actual_value}")

    if abs(actual_value - expected_sum) < 0.01:
        print("✓ Computation result is CORRECT!")
    else:
        print(f"✗ Computation result is INCORRECT (expected {expected_sum}, got {actual_value})")

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("Memory Usage Summary")
    print(f"{'='*70}")
    print(f"Initial RSS:                  {mem_after_cpu['rss_mb']:.2f} MB")
    print(f"After all to(device):         {mem_before_compute['rss_mb']:.2f} MB "
          f"(+{mem_before_compute['rss_mb'] - mem_after_cpu['rss_mb']:.2f} MB)")
    print(f"After computation:            {mem_after_compute['rss_mb']:.2f} MB "
          f"(+{mem_after_compute['rss_mb'] - mem_before_compute['rss_mb']:.2f} MB)")
    print(f"\nExpected behavior:")
    print(f"  - Total mmap size: {num_tensors * tensor_size_mb:.2f} MB ({num_tensors} tensors × {tensor_size_mb:.2f} MB)")
    print(f"  - RSS increase should be close to {num_tensors * tensor_size_mb:.2f} MB")
    print(f"  - Each tensor move should show a new ~{tensor_size_mb:.2f} MB mmap region")
    print(f"  - Check PJRT logs for BufferInstance[UID=*] messages with mmap addresses")
    print(f"  - Correlate mmap addresses in PJRT logs with /proc/{os.getpid()}/maps regions above")
    print("="*70 + "\n")

    return result


def disk_offload_model(model, offload_dir=None):
    """Offload model parameters to disk using mmap.

    This function replaces all model parameters with mmap-backed tensors,
    allowing the OS to swap them to disk when memory pressure is high.

    Args:
        model: PyTorch model to offload
        offload_dir: Directory to store mmap files (default: temp directory)

    Returns:
        Modified model with mmap-backed parameters, and the offload directory
    """
    if offload_dir is None:
        offload_dir = tempfile.mkdtemp(prefix="disk_offload_")

    print(f"Disk offloading model to: {offload_dir}")

    # Track all mmap'd files so they persist
    mmap_files = []
    param_count = 0
    total_size_mb = 0

    # Iterate through all parameters and replace with mmap-backed versions
    for name, param in model.named_parameters():
        param_count += 1
        param_size_bytes = param.element_size() * param.numel()
        param_size_mb = param_size_bytes / 1024 / 1024

        total_size_mb += param_size_mb

        print(f"  [{param_count}] Offloading parameter '{name}': "
              f"shape={list(param.shape)}, size={param_size_mb:.2f} MB", flush=True)

        # Create a temporary file for this parameter
        file_path = os.path.join(offload_dir, f"param_{param_count}_{name.replace('.', '_')}.bin")

        # Write parameter data to file
        with open(file_path, 'wb') as f:
            # Write the tensor data
            f.write(param.data.cpu().numpy().tobytes())

        # Open the file for mmap
        file_handle = open(file_path, 'r+b')
        mmap_files.append(file_handle)  # Keep file handle alive

        # Create mmap
        mmapped = mmap.mmap(file_handle.fileno(), 0)

        # Create a tensor from the mmap'd memory
        # Use from_buffer to create a tensor that shares memory with mmap
        import numpy as np
        np_array = np.frombuffer(mmapped, dtype=np.float32).reshape(param.shape)

        # Convert to PyTorch tensor
        # Note: This creates a tensor that shares memory with the numpy array
        offloaded_tensor = torch.from_numpy(np_array)

        # Replace the parameter data with the mmap-backed tensor
        param.data = offloaded_tensor

        print(f"      -> Mapped to file: {file_path}", flush=True)

    print(f"\nTotal parameters offloaded: {param_count}")
    print(f"Total size offloaded: {total_size_mb:.2f} MB")

    # Advise the kernel these pages can be swapped aggressively
    # Note: This only works on the actual mmap object, not the tensors
    for mm in mmap_files:
        try:
            # Python's mmap doesn't directly expose madvise, but we can suggest
            # the OS to not keep these pages in memory
            pass  # We'll rely on OS default behavior
        except:
            pass

    # Store mmap files in the model so they don't get garbage collected
    model._mmap_files = mmap_files
    model._offload_dir = offload_dir

    return model, offload_dir


def cleanup_disk_offload(model, offload_dir):
    """Clean up disk-offloaded model files.

    Args:
        model: Model with disk offload
        offload_dir: Directory containing offload files
    """
    if hasattr(model, '_mmap_files'):
        for f in model._mmap_files:
            try:
                f.close()
            except:
                pass
        del model._mmap_files

    if offload_dir and os.path.exists(offload_dir):
        import shutil
        shutil.rmtree(offload_dir, ignore_errors=True)
        print(f"Cleaned up offload directory: {offload_dir}")


class LargeParameterModel(nn.Module):
    """Model with 10 large (100MB each) parameter tensors.

    This model is designed to stress-test memory management:
    - 10 parameters of 5120x5120 float32 each = 100MB per parameter
    - Total model size: ~1000MB (1GB)
    - Forward pass: sums all parameters and scales by input
    """

    def __init__(self):
        super().__init__()
        # Create 10 large parameter tensors, each 100MB
        # Shape: 5120 x 5120 float32 = 26,214,400 elements * 4 bytes = 100MB
        self.param_tensors = nn.ParameterList([
            nn.Parameter(torch.ones(5120, 5120, dtype=torch.float32) * (i + 1))
            for i in range(10)
        ])

    def forward(self, scale_factor):
        """Sum all parameters and scale by the input scalar.

        Args:
            scale_factor: A scalar tensor to multiply the sum by

        Returns:
            A 5120x5120 tensor that is the sum of all parameters scaled by input
        """
        # Start with the first parameter
        result = self.param_tensors[0].clone()

        # Sum all remaining parameters
        for i in range(1, len(self.param_tensors)):
            result = result + self.param_tensors[i]

        # Scale by input
        # If scale_factor is a tensor, need to extract scalar for broadcasting
        if isinstance(scale_factor, torch.Tensor):
            if scale_factor.numel() == 1:
                scale_factor = scale_factor.item()

        result = result * scale_factor

        return result

    def get_total_size_mb(self):
        """Calculate total model size in MB."""
        total_params = sum(p.numel() for p in self.parameters())
        total_bytes = total_params * 4  # float32 = 4 bytes
        return total_bytes / 1024 / 1024


def test_large_parameter_model_tt_device():
    """Test large parameter model on TT device with PJRT mmap implementation.

    This test:
    1. Creates a 1GB model (10x 100MB parameters)
    2. Moves it to TT device (triggers PJRT mmap for each parameter)
    3. Monitors memory at each stage
    4. Runs inference with a scalar input
    5. Verifies computation correctness

    This is the main test that exercises our PJRT BufferInstance mmap implementation!
    """
    print("\n" + "="*70)
    print("Testing Large Parameter Model on TT Device")
    print("(This tests our PJRT mmap implementation!)")
    print("="*70)

    # Set device type to TT
    xr.set_device_type("TT")

    # Take initial snapshot
    print_memory_snapshot("Initial (before model creation)")

    # Create the model
    print("\nCreating 1GB model (10x 100MB parameters)...")
    model = LargeParameterModel()
    model_size_mb = model.get_total_size_mb()
    print(f"Model created: {model_size_mb:.2f} MB total")
    print(f"Parameters: {sum(1 for _ in model.parameters())} tensors")

    # Take snapshot after model creation (on CPU)
    mem_after_cpu_creation = print_memory_snapshot("After model creation on CPU")

    # Get TT device
    device = xm.xla_device()
    print(f"\nTT device: {device}")

    # Move model to device - this will trigger copyFromHost in PJRT for each parameter
    print("\n" + "="*70)
    print(f"Moving model to TT device...")
    print(f"This will create {sum(1 for _ in model.parameters())} mmap regions in PJRT")
    print(f"Expected total mmap size: ~{model_size_mb:.2f} MB")
    print("="*70)
    print("\nCheck PJRT logs for BufferInstance[UID=*]::copyFromHost messages", flush=True)
    print("Each parameter will show its mmap address and size\n", flush=True)

    device_model = model.to(device)

    # Give system time to process
    time.sleep(1)

    # Take snapshot after moving to device
    mem_after_to_device = print_memory_snapshot("After moving model to TT device (mmap'd)")

    # Prepare input
    print("\n" + "="*70)
    print("Running inference with scale_factor=2.5...")
    print("="*70)

    scale_input = torch.tensor(2.5)

    with torch.no_grad():
        output = device_model(scale_input)

    # Force synchronization
    print("\nSynchronizing...", flush=True)
    torch_xla.sync()

    # Take snapshot after computation
    mem_after_compute = print_memory_snapshot("After computation and sync")

    # Move result back to CPU for verification
    print("\nMoving result back to CPU for verification...")
    result_cpu = output.cpu()

    # Verify output
    # Expected: sum(1..10) * 2.5 = 55 * 2.5 = 137.5
    expected_value = sum(range(1, 11)) * 2.5
    actual_value = result_cpu[0, 0].item()

    print(f"\n{'='*70}")
    print("Computation Verification")
    print(f"{'='*70}")
    print(f"Expected value (per element): {expected_value}")
    print(f"Actual value (at [0,0]): {actual_value}")

    if abs(actual_value - expected_value) < 0.1:
        print("✓ Computation result is CORRECT!")
    else:
        print(f"✗ Computation result is INCORRECT (expected {expected_value}, got {actual_value})")

    # Print comprehensive summary
    print(f"\n{'='*70}")
    print("Memory Usage Summary - TT Device with PJRT mmap")
    print(f"{'='*70}")
    print(f"Initial RSS (CPU):        {mem_after_cpu_creation['rss_mb']:.2f} MB")
    print(f"After to(device):         {mem_after_to_device['rss_mb']:.2f} MB "
          f"(+{mem_after_to_device['rss_mb'] - mem_after_cpu_creation['rss_mb']:.2f} MB)")
    print(f"After computation:        {mem_after_compute['rss_mb']:.2f} MB "
          f"(+{mem_after_compute['rss_mb'] - mem_after_to_device['rss_mb']:.2f} MB)")

    print(f"\nKey observations:")
    print(f"  - Model size: {model_size_mb:.2f} MB")
    print(f"  - Memory increase from to(device): {mem_after_to_device['rss_mb'] - mem_after_cpu_creation['rss_mb']:.2f} MB")
    print(f"    (Should be close to {model_size_mb:.2f} MB for mmap'd buffers)")
    print(f"  - Each parameter triggered BufferInstance::copyFromHost in PJRT")
    print(f"  - Each copyFromHost created an mmap'd buffer (check debug logs)")
    print(f"  - Anonymous mmap regions above should show ~{model_size_mb:.2f} MB total")
    print(f"\nHow to correlate with PJRT logs:")
    print(f"  1. Set TTXLA_LOGGER_LEVEL=DEBUG before running")
    print(f"  2. Look for 'BufferInstance[UID=*]::copyFromHost - Mmap created at address 0x...'")
    print(f"  3. Match those addresses with the anonymous mmap regions shown above")
    print(f"  4. Verify each parameter has its own mmap region")
    print("="*70 + "\n")

    return output


def test_manual_disk_offload():
    """Test manual disk offload using mmap with large parameter model.

    This test:
    1. Creates a 1GB model (10x 100MB parameters)
    2. Offloads it to disk using our custom mmap implementation
    3. Measures memory before/after disk offload
    4. Runs inference and measures memory
    5. Compares with in-memory baseline
    """
    print("\n" + "="*70)
    print("Testing Manual Disk Offload (mmap-backed)")
    print("="*70)

    # Take initial snapshot
    print_memory_snapshot("Initial (before model creation)")

    # Create the model normally first (in memory)
    print("\nCreating 1GB model (10x 100MB parameters)...")
    model = LargeParameterModel()
    model_size_mb = model.get_total_size_mb()
    print(f"Model created: {model_size_mb:.2f} MB total")

    # Take snapshot after model creation
    mem_after_creation = print_memory_snapshot("After model creation (all in RAM)")

    # Now offload to disk using mmap
    print("\n" + "="*70)
    print("Offloading model parameters to disk using mmap...")
    print("="*70)

    offload_dir = None
    try:
        model, offload_dir = disk_offload_model(model)

        # Force garbage collection to free old parameter memory
        gc.collect()
        time.sleep(1)

        mem_after_offload = print_memory_snapshot("After disk offload (mmap-backed)")

        # Check the offload directory size
        if offload_dir:
            dir_size = sum(
                os.path.getsize(os.path.join(offload_dir, f))
                for f in os.listdir(offload_dir)
            ) / 1024 / 1024
            print(f"\nDisk usage in offload directory: {dir_size:.2f} MB")

        # Run inference
        print("\n" + "="*70)
        print("Running inference with scale_factor=2.0...")
        print("(This will page in parameters from disk as needed)")
        print("="*70)

        scale_input = torch.tensor(2.0)

        with torch.no_grad():
            output = model(scale_input)

        mem_after_inference = print_memory_snapshot("After inference")

        # Verify output
        # Expected: sum(1..10) * 2.0 = 55 * 2.0 = 110.0
        expected_value = sum(range(1, 11)) * 2.0
        actual_value = output[0, 0].item()

        print(f"\n{'='*70}")
        print("Computation Verification")
        print(f"{'='*70}")
        print(f"Expected value (per element): {expected_value}")
        print(f"Actual value (at [0,0]): {actual_value}")

        if abs(actual_value - expected_value) < 0.01:
            print("✓ Computation result is CORRECT!")
        else:
            print(f"✗ Computation result is INCORRECT (expected {expected_value}, got {actual_value})")

        # Print summary
        print(f"\n{'='*70}")
        print("Memory Usage Summary - Manual Disk Offload")
        print(f"{'='*70}")
        print(f"Initial RSS:              {mem_after_creation['rss_mb']:.2f} MB")
        print(f"After disk offload:       {mem_after_offload['rss_mb']:.2f} MB "
              f"({mem_after_offload['rss_mb'] - mem_after_creation['rss_mb']:+.2f} MB)")
        print(f"After inference:          {mem_after_inference['rss_mb']:.2f} MB "
              f"({mem_after_inference['rss_mb'] - mem_after_offload['rss_mb']:+.2f} MB)")

        print(f"\nKey observations:")
        print(f"  - Model size: {model_size_mb:.2f} MB")
        print(f"  - Memory change from offload: {mem_after_offload['rss_mb'] - mem_after_creation['rss_mb']:+.2f} MB")
        if dir_size:
            print(f"  - Disk space used: {dir_size:.2f} MB")
        print(f"  - Memory used during inference: {mem_after_inference['rss_mb'] - mem_after_offload['rss_mb']:+.2f} MB")
        print(f"    (Parameters paged in from disk as needed)")
        print(f"\nHow it works:")
        print(f"  - Each parameter is written to a file in {offload_dir}")
        print(f"  - Files are mmap'd and parameters point to mmap'd memory")
        print(f"  - OS can swap mmap'd pages to disk under memory pressure")
        print(f"  - Similar to our PJRT BufferInstance mmap implementation!")
        print("="*70 + "\n")

        return output

    finally:
        # Cleanup
        if offload_dir:
            cleanup_disk_offload(model, offload_dir)


def test_large_parameter_model_cpu():
    """Test the large parameter model on CPU without any offloading.

    This is a baseline test to compare against both:
    1. Accelerate's disk offload
    2. Our PJRT mmap implementation
    """
    print("\n" + "="*70)
    print("Testing Large Parameter Model on CPU (Baseline)")
    print("="*70)

    # Take initial snapshot
    print_memory_snapshot("Initial (before model)")

    # Create model
    print("\nCreating 1GB model (10x 100MB parameters)...")
    model = LargeParameterModel()
    model_size_mb = model.get_total_size_mb()
    print(f"Model created: {model_size_mb:.2f} MB total")

    mem_after_creation = print_memory_snapshot("After model creation")

    # Run inference
    print("\nRunning inference with scale_factor=2.0...")
    scale_input = torch.tensor(2.0)

    with torch.no_grad():
        output = model(scale_input)

    mem_after_inference = print_memory_snapshot("After inference")

    # Verify output
    expected_value = sum(range(1, 11)) * 2.0
    actual_value = output[0, 0].item()

    print(f"\n{'='*70}")
    print("Computation Verification")
    print(f"{'='*70}")
    print(f"Expected value: {expected_value}, Actual: {actual_value}")

    if abs(actual_value - expected_value) < 0.01:
        print("✓ Computation CORRECT!")
    else:
        print("✗ Computation INCORRECT")

    print(f"\n{'='*70}")
    print("Memory Usage Summary - CPU Baseline")
    print(f"{'='*70}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"RSS after creation: {mem_after_creation['rss_mb']:.2f} MB")
    print(f"RSS after inference: {mem_after_inference['rss_mb']:.2f} MB")
    print(f"Inference overhead: {mem_after_inference['rss_mb'] - mem_after_creation['rss_mb']:.2f} MB")
    print("="*70 + "\n")

    return output