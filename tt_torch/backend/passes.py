# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import gc
from torch.fx.experimental import const_fold


def bypass_assert_tensor_metadata(gm):
    """
    Bypass assert_tensor_metadata nodes.
    This is a noop node that is used to assert the tensor metadata.
    This is used to remove these assertion ops as we may be perparing a GraphModule
    which was originally compiled on CPU. Thus, these assertions will assert
    that its input tensor is on CPU. When running the graph on an XLA device later,
    it would fail the assertion.
    """
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten._assert_tensor_metadata.default
        ):
            gm.graph.erase_node(node)
    return gm


def run_shape_prop(gm, example_inputs):
    """
    Propagates shape information for each node through the graph bassed on
    The inputs in `example_inputs`. This will also populate the meta data
    for each node, which is usefuly for debugging.

    Runs quicly as only metadata is propagated, no compute is performed.
    """
    shape_prop = torch.fx.passes.shape_prop.ShapeProp(gm)
    if shape_prop.fake_mode is not None:
        fake_args = [
            shape_prop.fake_mode.from_tensor(act, static_shapes=True)
            if isinstance(act, torch.Tensor)
            else act
            for act in example_inputs
        ]
    else:
        fake_args = example_inputs
    shape_prop.run(*fake_args)


def bypass_redundant_getitem(gm):
    """
    Replaces `getitem` calls with a direct reference to the tensor being retrieved.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            if isinstance(node.args[0], tuple):
                idx = node.args[1]
                if isinstance(idx, int):
                    node.replace_all_uses_with(node.args[0][idx])
    return gm


def bypass_redundant_cast(gm):
    """
    Removes data type casting operations which are applied to tensors
    which are already of the desired dtype.
    """
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "name")
            and "prims::convert_element_type" in node.target.name()
        ):
            if node.args[1] == node.args[0].meta["tensor_meta"].dtype:
                node.replace_all_uses_with(node.args[0])

    return gm


def bypass_dtype_promotion(gm, compiler_config):
    """
    Removes casting of nodes to float32 unless they were explicitly cast by the user.
    Pytorch insists on casting params to float32, even though the user may have specified a different dtype,
    and forcing certain decomposition (i.e. adaptive_avg_pool2d) to be in float32
    """
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "name")
            and "prims::convert_element_type" in node.target.name()
        ):
            if (
                node.meta["original_aten"]._name != "aten::_to_copy"
                and node.args[1] == torch.float32
            ):
                node.replace_all_uses_with(node.args[0])

    return gm


def rectify_buffer_inplace_copy(gm):
    """
    Transformers static cache uses index_copy_, which produces illegal inplace copy_ nodes
    Remove these illegal nodes, and replicate the inplace copy semantics using op fusion and
    buffer semantics in the backend.
    """

    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            # Detect inplace copy with buffer destination
            destination_node = node.args[0]
            if destination_node.op != "get_attr":
                continue
            gm.graph.erase_node(node)
    return gm


def constant_fold(gm):
    """
    Splits constant subgraphs and folds constants. This is required for two reasons:
    1. Creation operations will have the target device be "cpu", however we ultimately
       intend to run the graph on an XLA device. So, when their outputs (which are on "cpu")
       are operated against a tensor on an XLA device, the program will fail to execute.
       By folding them, the output tensors are computed at compile time and are added to
       the state_dict of the GraphModule, which can be pushed to device along with the
       model parameters/buffers/constants. Remedyies for this in the future would include
       either forcing the `device` attribute of creation ops to be "xla", or compling
       a model which is already on an XLA device rather than a CPU device.
          - However, if `constant_fold` is run on a model which is already on an XLA device,
            the compute involved with constant-folding would be performed on the XLA device
            as well, which can be problematic if any of those computations cannot be
            compiled through tt-mlir.
    2. Some ops which are constant-foldable cannot be consumed by tt-mlir, so eliminating
       them here is useful. Remedies for this in the future would include handling them with
       cpu-fallback in tt-mlir.
    """
    gm = const_fold.split_const_subgraphs(gm)
    gc.collect()
    gm.run_folding()

    gm.graph.eliminate_dead_code()
    return gm
