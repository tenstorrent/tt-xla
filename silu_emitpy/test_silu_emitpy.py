# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import ttnn
from loguru import logger

import silu_emitpy.utils as utils
from third_party.tt_forge_models.yolov9.pytorch.loader import ModelLoader, ModelVariant

_CONST_EVAL_CACHE = {}


def _main(input):
    ttnn_to_layout_0 = ttnn.to_layout(
        input[0],
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(input[0], False)
    ttnn_silu_0 = ttnn.silu(
        ttnn_to_layout_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    util_create_list_0 = [ttnn_silu_0]
    return util_create_list_0


def load_inputs_for__main():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_1 = [utils_load_tensor_0]
    return util_create_list_1


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float32) if x.dtype != torch.float32 else x
    y_float = y.to(torch.float32) if y.dtype != torch.float32 else y

    x_flat, y_flat = x_float.flatten(), y_float.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")
    else:
        return ((vx @ vy) / denom).item()



def test_silu_org():
    # tt run
    load_inputs_for__main_0 = load_inputs_for__main()
    tt_input_torch = ttnn.to_torch(load_inputs_for__main_0[0])
    _main_0 = _main(load_inputs_for__main_0)
    tt_output = _main_0[0]

    # cpu run
    cpu_input = torch.load('act_ip.pt', map_location="cpu")

    # Compare inputs
    logger.info("\n=== Input Comparison ===")
    logger.info("tt_input_torch: {}", tt_input_torch)
    logger.info("tt_input_torch.shape: {}", tt_input_torch.shape)
    logger.info("tt_input_torch.dtype: {}", tt_input_torch.dtype)

    logger.info("cpu_input: {}", cpu_input)
    logger.info("cpu_input.shape: {}", cpu_input.shape)
    logger.info("cpu_input.dtype: {}", cpu_input.dtype)

    input_allclose = torch.equal(tt_input_torch, cpu_input)
    logger.info(f"Input equal?: {input_allclose}")

    # cpu run
    loader = ModelLoader(ModelVariant.S)
    model = loader.load_model(dtype_override=torch.bfloat16)
    logger.info("model={}",model)

    with torch.no_grad():
        cpu_output = model(cpu_input)

    # Compare outputs
    logger.info("\n=== Output Comparison ===")
    tt_output_torch = ttnn.to_torch(tt_output)

    logger.info("tt_output_torch: {}", tt_output_torch)
    logger.info("tt_output_torch.shape: {}", tt_output_torch.shape)
    logger.info("tt_output_torch.dtype: {}", tt_output_torch.dtype)

    logger.info("cpu_output: {}", cpu_output)
    logger.info("cpu_output.shape: {}", cpu_output.shape)
    logger.info("cpu_output.dtype: {}", cpu_output.dtype)

    # Compute PCC
    pcc = compute_pcc(tt_output_torch, cpu_output)
    logger.info(f"Output PCC: {pcc}")
