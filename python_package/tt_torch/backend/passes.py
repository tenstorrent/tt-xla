# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.export.graph_signature import InputKind, OutputKind
from tt_torch import composite_ops


def handle_composite_ops(gm: torch.fx.GraphModule) -> None:
    """
    Replaces torch ops with composite ops if we have a proper replacement.

    Handles two types of nodes:
    1. call_function nodes: Functional ops like torch.nn.functional.gelu
       - node.target is a function reference
       - Replaced by changing node.target to composite function

    2. call_module nodes: nn.Module instances like nn.LayerNorm
       - node.target is a string like "layer_norm"
       - Replaced by creating new call_function node with get_attr for parameters
    """
    nodes_to_replace = []

    for node in gm.graph.nodes:
        if node.op == "call_function":
            if node.target in composite_ops.function_replacements:
                nodes_to_replace.append(("function", node, None))

        elif node.op == "call_module":
            module = gm.get_submodule(node.target)
            module_type = type(module)

            if module_type in composite_ops.module_replacements:
                nodes_to_replace.append(("module", node, module))

    for replacement_type, node, module in nodes_to_replace:
        if replacement_type == "function":
            _replace_function_node(gm, node)
        else:  # replacement_type == "module"
            _replace_module_node(gm, node, module)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()


def _replace_function_node(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """
    Replace a call_function node with its composite equivalent.

    Simple replacement: just change the target function reference.
    Args and kwargs remain the same.
    """
    node.target = composite_ops.function_replacements[node.target]


def _replace_module_node(
    gm: torch.fx.GraphModule, node: torch.fx.Node, module: torch.nn.Module
) -> None:
    """
    Replace a call_module node with a call_function node to the composite equivalent.

    Strategy:
    1. Extract module parameters and configuration
    2. Create get_attr nodes for parameters (weight, bias)
    3. Create new call_function node with composite function
    4. Replace all uses and remove old node

    The parameters remain in the module's state dict and are accessed via get_attr.
    Later, torch.export will lift these get_attr nodes to placeholders.
    """
    module_type = type(module)
    composite_fn = composite_ops.module_replacements[module_type]

    # Dispatch to type-specific handler
    if module_type == torch.nn.LayerNorm:
        _replace_layer_norm_node(gm, node, module, composite_fn)
    else:
        raise ValueError(
            f"Module type {module_type} is in module_replacements but has no handler. "
            f"Please add a _replace_{module_type.__name__.lower()}_node function."
        )


def _replace_layer_norm_node(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    module: torch.nn.LayerNorm,
    composite_fn,
) -> None:
    """
    Replace nn.LayerNorm call_module node with composite_layer_norm call_function.

    Transformation:
        BEFORE: %out = call_module[target=layer_norm](args=(%x,))
        AFTER:  %weight = get_attr[target=layer_norm.weight]
                %bias = get_attr[target=layer_norm.bias]
                %out = call_function[target=composite_layer_norm](
                    args=(%x,),
                    kwargs={normalized_shape: (768,), weight: %weight,
                            bias: %bias, eps: 1e-5}
                )

    Args:
        gm: GraphModule containing the node
        node: call_module node to replace
        module: nn.LayerNorm instance
        composite_fn: composite_layer_norm function
    """
    # Extract module configuration
    normalized_shape = module.normalized_shape
    eps = module.eps
    has_weight = module.weight is not None and module.elementwise_affine
    has_bias = module.bias is not None and module.elementwise_affine

    # Get input tensor (first argument to the module call)
    input_tensor = node.args[0]

    # Build kwargs for the composite function
    kwargs = {"normalized_shape": normalized_shape, "eps": eps}

    # Create get_attr nodes for weight and bias if they exist
    with gm.graph.inserting_before(node):
        if has_weight:
            weight_node = gm.graph.get_attr(f"{node.target}.weight")
            kwargs["weight"] = weight_node
        else:
            kwargs["weight"] = None

        if has_bias:
            bias_node = gm.graph.get_attr(f"{node.target}.bias")
            kwargs["bias"] = bias_node
        else:
            kwargs["bias"] = None

        new_node = gm.graph.call_function(
            composite_fn, args=(input_tensor,), kwargs=kwargs
        )

    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


def insert_argument_type_markers(
    gm: torch.fx.GraphModule, graph_signature
) -> torch.fx.GraphModule:

    input_nodes = gm.graph.find_nodes(op="get_attr") + gm.graph.find_nodes(
        op="placeholder"
    )
    input_signature = graph_signature.input_specs
    output_signature = graph_signature.output_specs

    # Keep track of buffers which are mutated as we do not want these arguments to be hoisted into a consteval graph.
    mutated_buffer_targets = set()
    for out_spec in output_signature:
        if out_spec.kind == OutputKind.BUFFER_MUTATION:
            mutated_buffer_targets.add(out_spec.target)

    get_attr_target_type_dict = {}
    placeholder_target_type_dict = {}
    for in_spec in input_signature:
        type_str = None
        if in_spec.kind == InputKind.USER_INPUT:
            type_str = "input"
        # We do not model these argument types in tt-mlir. To avoid graph transformations that would
        # impact how these inputs are handled (i.e. consteval), we will mark them as "input".
        elif in_spec.kind in [InputKind.TOKEN, InputKind.CUSTOM_OBJ]:
            type_str = "input"
        elif in_spec.kind == InputKind.PARAMETER:
            type_str = "parameter"
        elif in_spec.kind == InputKind.CONSTANT_TENSOR:
            type_str = "constant"
        # If a buffer is mutated, we do not want to hoist the argument into a consteval graph.
        # This is because the argument will be mutated in place, and we do not want to used the cached
        # version of the input from the first iteration of the graph. If it is not mutated then we can
        # mark it as a constant.
        elif in_spec.kind == InputKind.BUFFER:
            if in_spec.target in mutated_buffer_targets:
                type_str = "input"
            else:
                type_str = "constant"
        else:
            assert False, f"Unexpected input kind: {in_spec.kind}"

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
            # Replacing the input to an in-place copy_ op with a `tt.mark_argument_attributes` result
            # causes XLA to handle the copying into an input tensor incorrectly. So, we do not
            # replace the destination tensor with the `tt.mark_argument_attributes` result.
            if (
                user.target == torch.ops.aten.copy_.default
                and user.args[0] == input_node
            ):
                continue
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


def run_shape_prop(gm, example_inputs):
    """
    Propagates shape and dtype information through the graph.
    """
    shape_prop = torch.fx.passes.shape_prop.ShapeProp(gm)
    if shape_prop.fake_mode is not None:
        fake_args = [
            (
                shape_prop.fake_mode.from_tensor(act, static_shapes=True)
                if isinstance(act, torch.Tensor)
                else act
            )
            for act in example_inputs
        ]
    else:
        fake_args = example_inputs
    shape_prop.run(*fake_args)


def bypass_dtype_promotion_and_redundant_cast(gm, example_inputs):
    """
    Removes casting of nodes to float32 unless they were explicitly set by the user.
    Pytorch insists on casting nodes to float32 during decompositions, even though the
    user may have specified a different dtype.
    Also removes redundant casts.
    """
    removed_non_redundant_casts = False
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and hasattr(node.target, "name")
            and "prims::convert_element_type" in node.target.name()
        ):
            is_unwanted_dtype_promotion = (
                node.meta["original_aten"]._name != "aten::_to_copy"
                and node.args[1] == torch.float32
            )
            is_redundant_cast = (
                "tensor_meta" in node.args[0].meta
                and node.args[0].meta["tensor_meta"].dtype == node.args[1]
            )

            if is_unwanted_dtype_promotion or is_redundant_cast:
                node.replace_all_uses_with(node.args[0])
                removed_non_redundant_casts |= is_unwanted_dtype_promotion

    gm.graph.eliminate_dead_code()
    gm.graph.lint()

    if removed_non_redundant_casts:
        # if non redundant nodes were removed, re-propagate shape and dtype and re-run pass to remove redundant casts
        run_shape_prop(gm, example_inputs)
        gm = bypass_dtype_promotion_and_redundant_cast(gm, example_inputs)

    return gm
