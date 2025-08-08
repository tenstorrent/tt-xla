# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import time
import operator

from .decompositions import (
    CUSTOM_DECOMPOSITION_TABLE,
)
import os
import tempfile
import multiprocessing as mp
import re
import pickle
import faulthandler
import collections
from .passes import (
    bypass_redundant_getitem,
    bypass_dtype_promotion,
    bypass_redundant_cast,
    rectify_buffer_inplace_copy,
    run_shape_prop,
    constant_fold,
)

from torch.export.graph_signature import InputKind
from torch._dynamo import register_backend

from tt_torch.tools.utils import (
    CompilerConfig,
    CompileDepth,
    Op,
    OpCompilationStatus,
    calculate_atol,
    calculate_pcc,
)

import torch_xla
import torch_xla.core.xla_model as xm


def bypass_assert_tensor_metadata(gm):
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten._assert_tensor_metadata.default
        ):
            gm.graph.erase_node(node)
    return gm


def xla_pass_pipeline(gm, example_inputs, compiler_config):
    decompositions = torch._decomp.core_aten_decompositions()
    decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
    compiled_graph = (
        torch.export.export_for_training(gm, tuple(example_inputs), strict=False)
        .run_decompositions(decompositions)
        .module()
    )

    compiled_graph = bypass_dtype_promotion(compiled_graph, compiler_config)
    run_shape_prop(compiled_graph, example_inputs)
    compiled_graph = bypass_redundant_cast(compiled_graph)

    if compiler_config.enable_consteval:
        compiled_graph = constant_fold(compiled_graph)
    elif compiler_config.consteval_parameters:
        raise Exception("consteval_parameters is enabled but enable_consteval is not")

    compiled_graph = bypass_redundant_getitem(compiled_graph)
    compiled_graph = rectify_buffer_inplace_copy(compiled_graph)
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)
    program = torch.export.export(compiled_graph, tuple(example_inputs), strict=False)

    return program


class XLAExecutor:
    def __init__(self, program, compiler_config):
        self.program = program
        self.compiler_config = compiler_config
        self.arg_type_map_str = None

        self.inputs = []
        self.user_input_indices = []
        for idx, input_spec in enumerate(self.program._graph_signature.input_specs):
            if input_spec.kind == InputKind.USER_INPUT:
                self.inputs.append(None)
                self.user_input_indices.append(idx)
            else:
                self.inputs.append(self.program.state_dict[input_spec.target].to("xla"))

    def push_tensors_to_device(self, inputs, device):
        if hasattr(inputs, "to"):
            if device not in [inputs.device, inputs.device.type]:
                return inputs.to(device)
            else:
                return inputs
        elif isinstance(
            inputs, dict
        ):  # transformers input/output objects are subclasses of dict, however we still wish to return the same wrapper object
            return type(inputs)(
                **{k: self.push_tensors_to_device(v, device) for k, v in inputs.items()}
            )
        elif isinstance(inputs, collections.abc.Sequence):
            return type(inputs)(
                [self.push_tensors_to_device(i, device) for i in inputs]
            )
        elif hasattr(inputs, "key_cache") or hasattr(inputs, "value_cache"):
            if hasattr(inputs, "key_cache"):
                inputs.key_cache = self.push_tensors_to_device(inputs.key_cache, device)
            if hasattr(inputs, "value_cache"):
                inputs.value_cache = self.push_tensors_to_device(
                    inputs.value_cache, device
                )
            return inputs
        else:
            return inputs

    def generate_arg_type_map_str(self, output_object):
        hlo_input_ids, _ = torch_xla._XLAC._get_tensors_xla_device_data_node(
            output_object
        )

        # xm.get_stablehlo(output_object) gives a graph with just as many inputs as in hlo_input_ids

        hlo_input_positions = [id - min(hlo_input_ids) for id in hlo_input_ids]

        def get_kind_str(kind):
            if kind == InputKind.USER_INPUT:
                return "input"
            elif kind == InputKind.PARAMETER:
                return "parameter"
            else:
                return "constant"

        arg_types = []
        output_args = [o.arg for o in self.program.graph_signature.output_specs]
        for idx in range(len(hlo_input_positions)):
            if hlo_input_positions[idx] < len(self.program.graph_signature.input_specs):
                in_spec = self.program.graph_signature.input_specs[
                    hlo_input_positions[idx]
                ]

                # If an input is passed right through to the output, it will not be
                # captured as an argument
                if in_spec.arg in output_args:
                    continue

                arg_types.append(get_kind_str(in_spec.kind))
            else:
                arg_types.append("constant")

        self.arg_type_map_str = "main=" + ",".join(arg_types)

    def __call__(self, *args):
        args = self.push_tensors_to_device(args, "xla")
        inputs = self.inputs
        for idx in range(len(args)):
            inputs[self.user_input_indices[idx]] = args[idx]

        output = self.program.graph_module(*inputs)

        if self.compiler_config.arg_type_map_override:
            if self.arg_type_map_str is None:
                self.generate_arg_type_map_str(output)
            if os.environ.get("ARG_TYPE_MAP_OVERRIDE") != self.arg_type_map_str:
                os.environ["ARG_TYPE_MAP_OVERRIDE"] = self.arg_type_map_str

        xm.mark_step()
        if self.compiler_config.push_outputs_to_cpu:
            return self.push_tensors_to_device(output, "cpu")
        return output

    def __del__(self):
        # Remove the arg type map override environment variable
        os.environ.pop("ARG_TYPE_MAP_OVERRIDE", None)


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options: CompilerConfig = None):
    compiler_config = options
    if compiler_config is None:
        compiler_config = CompilerConfig()

    program = xla_pass_pipeline(gm, example_inputs, compiler_config)
    return XLAExecutor(program, compiler_config)
