# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoImageProcessor, ResNetForImageClassification


# --------------------------------
# Test run
# --------------------------------
def resnet_dp():
    # Instantiate model.
    model_name = "microsoft/resnet-50"
    model: torch.nn.Module = ResNetForImageClassification.from_pretrained(model_name)
    model = model.to(torch.bfloat16)
    model = model.eval()

    # Instantiate image processor.
    image_processor = AutoImageProcessor.from_pretrained(
        model_name, dtype=torch.bfloat16
    )

    # Setup SPMD.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    # Create mesh.
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

    # Connect the device.
    device = xm.xla_device()

    # Generate dummy inputs with batch size of num devices.
    batch_size = num_devices
    images = torch.randn(batch_size, 3, 224, 224)
    inputs = image_processor(images=images, return_tensors="pt").pixel_values

    # Move inputs and model to device.
    inputs = inputs.to(device)
    model = model.to(device)

    # Mark sharding for inputs along batch dimension.
    xs.mark_sharding(inputs, mesh, ("batch", None, None, None))

    # Compile model
    compiled_model = torch.compile(model, backend="tt")

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output = compiled_model(inputs)

    print(output)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    resnet_dp()
