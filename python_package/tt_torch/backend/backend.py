# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import torch
import torch_xla
from torch._dynamo import register_backend
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind

from .decompositions import CUSTOM_DECOMPOSITION_TABLE
from .metadata_propagation import MetadataDispatchMode, extract_nodes_info
from .passes import (
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    handle_composite_ops,
    insert_argument_type_markers,
    run_shape_prop,
)


def insert_argument_type_markers_v2(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:

    global graph_signature
    input_nodes = gm.graph.find_nodes(op="get_attr") + gm.graph.find_nodes(
        op="placeholder"
    )

    # Keep track of buffers which are mutated as we do not want these arguments to be hoisted into a consteval graph.
    mutated_buffer_targets = set()
    for buffer_mutation in graph_signature.buffers_to_mutate:
        mutated_buffer_targets.add(buffer_mutation)

    get_attr_target_type_dict = {}
    placeholder_target_type_dict = {}

    for input_name in graph_signature.user_inputs:
        placeholder_target_type_dict[input_name] = "input"

    for arg_name in graph_signature.inputs_to_parameters.keys():
        placeholder_target_type_dict[arg_name] = "parameter"

    for buffer_name in graph_signature.buffers:
        if buffer_name in mutated_buffer_targets:
            get_attr_target_type_dict[buffer_name] = "input"
        else:
            get_attr_target_type_dict[buffer_name] = "constant"

    if hasattr(graph_signature, "tokens") and graph_signature.tokens:
        for token_name in graph_signature.tokens:
            placeholder_target_type_dict[token_name] = "input"

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


import torch_xla.core.dynamo_bridge as bridge


def tt_torch_helper(model, fake_tensor_inputs):
    import torch_xla.core.dynamo_bridge as bridge
    from functorch.compile import make_boxed_func

    compiled_graph = None
    node_info = None

    def fwd(*args):
        nonlocal model
        nonlocal compiled_graph
        nonlocal node_info
        if compiled_graph is None:
            model = insert_argument_type_markers_v2(model)
            model = torch_pass_pipeline(model, args)
            node_info = extract_nodes_info(model)
            # with MetadataDispatchMode(node_info):
            compiled_graph = bridge.extract_compiled_graph(model, args)
            del model
        return compiled_graph(*args)

    return fwd


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
) -> Tuple[torch.fx.GraphModule, torch.export.ExportGraphSignature, list[str]]:

    # Currently, handle_composite_ops causes regressions on multi-chip TP models:
    # https://github.com/tenstorrent/tt-xla/issues/1616.
    # TODO: Fix composite ops to support multi-chip models before uncommenting this.
    # handle_composite_ops(gm)

    # decompositions = torch._decomp.core_aten_decompositions()
    # decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
    #
    # We use `export_for_training` here as we plan to use this flow to compile training graphs.
    # In addition to that, the functionality in `export_for_training` will become the default
    # functionality in torch.export in a future PyTorch release:
    # https://docs.pytorch.org/docs/stable/export.html#export-for-training-and-inference
    print(
        f"len(example_inputs) - before bypass_dtype_promotion_and_redundant_cast: {len(example_inputs)}"
    )
    compiled_graph = bypass_dtype_promotion_and_redundant_cast(gm, example_inputs)
    print(
        f"len(example_inputs) - after bypass_dtype_promotion_and_redundant_cast: {len(example_inputs)}"
    )
    compiled_graph = bypass_redundant_getitem(compiled_graph)
    print(
        f"len(example_inputs) - after bypass_redundant_getitem: {len(example_inputs)}"
    )
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)
    print(
        f"len(example_inputs) - after bypass_assert_tensor_metadata: {len(example_inputs)}"
    )

    # Recompile the GraphModule to ensure the modifications made by the above
    # passes are reflected during execution.
    compiled_graph.recompile()

    # Extract metadata from FX nodes in order to inject them into locs
    # node_info = extract_nodes_info(compiled_graph)

    return compiled_graph


from torch._dynamo.backends.torchxla import openxla


class XLAExecutor:
    """
    This class is used to execute a compiled program on an XLA device.
    It is responsible for:
    1. Executing the GraphModule
    2. Signalling to torch-xla to cut the graph at the model output.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        signature: torch.export.ExportGraphSignature,
        node_info: list[str],
    ):
        self.module = module
        self.signature = signature
        self.node_info = node_info
        # Inject metadata if xla debug is enabled and node_info is not empty
        # We need xla debug to be enabled in order for torch-xla to inject metadata
        self.inject_metadata = os.environ.get("XLA_HLO_DEBUG", "0") == "1" and node_info

        # Collect all devices this model will use. This device list is used to
        # signal to torch xla which devices are involved in computing the output
        # tensors, so that we may cut the graph on the output tensors correctly.
        self.devices = set()
        for _, tensor in module.state_dict().items():
            self.devices.add(tensor.device.type)
        self.devices = list(self.devices)

    def __call__(self, *args):

        if self.inject_metadata:
            # MetadataDispatchMode intercepts tensor operations via TorchDispatchMode and
            # attaches FX metadata (module hierarchy, file, line) to XLA tensors.
            with MetadataDispatchMode(self.node_info):
                output = self.module(*args)
        else:
            output = self.module(*args)

        gm_has_functional_output_kind: bool = True

        for el in self.signature.output_specs:
            if el.kind is not OutputKind.USER_OUTPUT:
                gm_has_functional_output_kind = False
                break

        if gm_has_functional_output_kind:
            # This tells torch-xla to cut the graph at only what is required to
            # compute all tensors in the `output` list.
            torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        else:
            # Some graphs have side effects not included in graph output.
            # In these cases we must call sync() to force materialization of non-user-output
            # tensors, eg. inplace static cache updates as OutputKind.USER_INPUT_MUTATION.
            # This causes buffer mutations to show up as graph outputs in MLIR.
            torch_xla.sync()

        return output


from torch._functorch.aot_autograd import aot_export_module
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils import _pytree as pytree


def build_flat_args_from_sig(
    orig_gm, gm, sig, user_args, *, token_value=None, fake_mode=None
):
    # Maps for module state (FQNs must match those in sig)
    param_map = dict(orig_gm.named_parameters(remove_duplicate=False))
    buf_map = dict(orig_gm.named_buffers(remove_duplicate=False))

    # Flatten user inputs using standard pytree flatten
    user_flat, user_spec = pytree.tree_flatten(user_args)
    # Optional sanity: structure should match what export saw
    # (don't hard-fail; just a helpful check)
    try:
        roundtrip = pytree.tree_unflatten(user_flat, sig.in_spec)
    except Exception:
        roundtrip = None
    # You can assert if you want strictness:
    # assert roundtrip is not None, "User inputs don't match signature's in_spec shape"

    user_i = 0
    token_names = set(getattr(sig, "input_tokens", []))

    flat = []
    for n in gm.graph.nodes:
        if n.op != "placeholder":
            continue
        name = n.target  # GraphInputName

        print(f"Processing placeholder: {name}")

        if name in sig.inputs_to_parameters:
            fqn = sig.inputs_to_parameters[name]
            print(f"  It's a parameter with fqn: {fqn}")
            t = param_map[fqn].data  # plain Tensor (not nn.Parameter)
            print(f"    Parameter tensor shape: {t.shape}")

        elif name in sig.inputs_to_buffers:
            fqn = sig.inputs_to_buffers[name]
            print(f"  It's a buffer with fqn: {fqn}")
            t = buf_map[fqn]
            print(f"    Buffer tensor shape: {t.shape}")

        elif name in sig.user_inputs:
            t = user_flat[user_i]
            print(f"  It's a user input at index: {user_i}")
            user_i += 1
            print(f"    User input tensor shape: {t.shape}")

        elif name in token_names:
            # token sentinel; many backends accept None
            assert False, "Tokens not supported yet in tt backend"
            t = token_value if token_value is not None else None

        else:
            assert False, f"Unrecognized placeholder '{name}'"
            # Likely a dynamic dim / symint placeholder
            meta_val = n.meta.get("val") if isinstance(n.meta, dict) else None
            if isinstance(meta_val, (int, bool)):
                t = int(meta_val)
            else:
                raise RuntimeError(f"Unrecognized placeholder '{name}' (meta={n.meta})")

        # # If you're in FakeTensorMode and need fake tensors, wrap here:
        # if fake_mode is not None and isinstance(t, torch.Tensor):
        #     if not getattr(t, "_is_fake", False):  # or use torch._subclasses.fake_tensor.is_fake if available
        #         t = fake_mode.from_real_tensor(t)

        flat.append(t)

    # Final checks
    num_ph = sum(1 for m in gm.graph.nodes if m.op == "placeholder")
    assert num_ph == len(flat), (num_ph, len(flat))
    # Quick validation (will raise if any index is wrong):
    # TODO: THIS FAILS IN ALEXNET
    # run_shape_prop(gm, tuple(flat))

    return tuple(flat)


graph_signature = None


class TTBackendCompiler:
    def __init__(self, **kwargs):
        from torch._dynamo.backends.common import aot_autograd

        self.decompositions = torch._decomp.core_aten_decompositions()
        self.decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
        self.aot_autograd = aot_autograd(
            fw_compiler=tt_torch_helper,
            # decompositions=self.decompositions,
        )
        self.compiled_graph = None

    def __call__(self, orig_gm: torch.fx.GraphModule, example_inputs, **kwargs):

        # program = torch.export.export_for_training(
        #     gm, tuple(example_inputs), strict=False
        # ).run_decompositions(self.decompositions)
        # # gm = insert_argument_type_markers(program.module(), program.graph_signature)
        # gm = program.module()
        from torch._functorch.aot_autograd import aot_export_module

        gm, graph_sig = aot_export_module(
            orig_gm,
            example_inputs,
            decompositions=self.decompositions,
            trace_joint=False,
        )
        global graph_signature
        graph_signature = graph_sig
        # gm = insert_argument_type_markers_v2(gm, graph_sig)

        # print(f"gm.code: {gm.code}")

        # example_inputs = tuple(dict(orig_gm.named_parameters(remove_duplicate=False)).values()) + tuple(
        #     dict(orig_gm.named_buffers(remove_duplicate=False)).values()
        # ) + tuple(example_inputs)
        # print(f"example_inputs length: {len(example_inputs)}")

        # flat_args = build_flat_args_from_sig(orig_gm, gm, graph_sig, example_inputs)
        # flat_args = example_inputs
        # return tt_torch_helper(model, flat_args)

        # if self.compiled_graph is None:
        #     model = torch_pass_pipeline(gm, flat_args)
        #     node_info = extract_nodes_info(model)
        #     # with MetadataDispatchMode(node_info):
        #     self.compiled_graph = bridge.extract_compiled_graph(model, flat_args)
        #     del model
        # return self.compiled_graph(*flat_args)

        # print(f"flat_args length: {len(flat_args)}")
        return self.aot_autograd(orig_gm, example_inputs, **kwargs)


def create_compiler_fn() -> TTBackendCompiler:
    return TTBackendCompiler()


assert callable(create_compiler_fn())

register_backend(name="tt", compiler_fn=create_compiler_fn())

# def xla_backend():
#     """TT backend for torch.compile."""
#     from torch._dynamo.backends.common import aot_autograd
#     decompositions = torch._decomp.core_aten_decompositions()
#     decompositions.update(CUSTOM_DECOMPOSITION_TABLE)
#     return aot_autograd(
#         fw_compiler=tt_torch_helper,
#         decompositions=decompositions,
#     )
#     # module, graph_signature, node_info = torch_pass_pipeline(gm, example_inputs)
#     # return XLAExecutor(module, graph_signature, node_info)
#
# register_backend(name="tt", compiler_fn=xla_backend)
