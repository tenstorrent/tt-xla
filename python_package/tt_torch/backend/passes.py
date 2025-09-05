# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import gc
from torch.fx.experimental import const_fold
from torch.export.graph_signature import InputKind


def insert_argument_type_markers(
    gm: torch.fx.GraphModule, graph_signature
) -> torch.fx.GraphModule:

    input_nodes = gm.graph.find_nodes(op="get_attr") + gm.graph.find_nodes(
        op="placeholder"
    )
    input_signature = graph_signature.input_specs
    get_attr_target_type_dict = {}
    placeholder_target_type_dict = {}
    for in_spec in input_signature:
        type_str = None
        if in_spec.kind == InputKind.USER_INPUT:
            type_str = "input"
        elif in_spec.kind == InputKind.PARAMETER:
            type_str = "parameter"
        elif in_spec.kind == InputKind.BUFFER:
            # Buffers like static cache should be treated as inputs, not constants
            type_str = "input"
        # InputKind.CONSTANT_TENSOR should still be marked as constant
        else:
            type_str = "constant"

        if in_spec.target is not None:
            get_attr_target_type_dict[in_spec.target] = type_str
        else:
            placeholder_target_type_dict[in_spec.arg.name] = type_str

    for input_node in input_nodes:
        users = list(input_node.users.keys())
        if len(users) == 0:
            continue

        argument_type = None
        if input_node.target in get_attr_target_type_dict:
            argument_type = get_attr_target_type_dict[input_node.target]
        elif input_node.name in placeholder_target_type_dict:
            argument_type = placeholder_target_type_dict[input_node.name]
        else:
            continue

        with gm.graph.inserting_after(input_node):
            new_input = gm.graph.create_node(
                "call_function",
                torch.ops.tt.mark_argument_attributes,
                args=(input_node,),
                kwargs={"argument_type": argument_type, "name": input_node.name},
            )

        for user in users:
            user.replace_input_with(input_node, new_input)

    return gm


def bypass_assert_tensor_metadata(gm):
    """
    Bypass assert_tensor_metadata nodes.
    This is a noop node that is used to assert the tensor metadata.
    This is used to remove these assertion ops as we may be preparing a GraphModule
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
    Propagates shape information for each node through the graph based on
    The inputs in `example_inputs`. This will also populate the meta data
    for each node, which is useful for debugging.

    Runs quickly as only metadata is propagated, no compute is performed.
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
            if "tensor_meta" not in node.args[0].meta:
                continue
            if node.args[1] == node.args[0].meta["tensor_meta"].dtype:
                node.replace_all_uses_with(node.args[0])

    return gm


def bypass_dtype_promotion(gm):
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
