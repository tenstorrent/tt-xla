# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import gc
from torch.fx.experimental import const_fold


def run_shape_prop(gm, example_inputs):
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
    for node in gm.graph.nodes:
        if node.op == "call_function" and "getitem" in node.name:
            if isinstance(node.args[0], tuple):
                idx = node.args[1]
                if isinstance(idx, int):
                    node.replace_all_uses_with(node.args[0][idx])
    return gm


def bypass_redundant_cast(gm):
    # Removes cast nodes that cast to already existing dtype
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
    # Removes casting of nodes to float32 unless they were explicitly cast by the user.
    # Pytorch insists on casting params to float32, even though the user may have specified a different dtype,
    # and forcing certain decomposition (i.e. adaptive_avg_pool2d) to be in float32
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
    gm = const_fold.split_const_subgraphs(gm)
    gc.collect()
    gm.run_folding()

    gm.graph.eliminate_dead_code()
    return gm
