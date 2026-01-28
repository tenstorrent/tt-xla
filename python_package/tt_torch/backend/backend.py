# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import torch
import torch.export
import torch_xla
import torch_xla.core.dynamo_bridge as bridge
from functorch.compile import make_boxed_func
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch._subclasses.functional_tensor import mb_unwrap_functional_tensor
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from ttxla_tools.logging import logger

from .decompositions import populate_decompositions
from .metadata_propagation import MetadataDispatchMode, extract_nodes_info
from .passes import (
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    handle_composite_ops,
    insert_argument_type_markers,
    run_fusion_passes,
)


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
) -> Tuple[torch.fx.GraphModule, torch.export.ExportGraphSignature, list[str]]:

    # Run fusion passes to detect and fuse multi-op patterns
    # This runs before composite_ops to allow fused patterns to be wrapped as composites
    enable_fusion_passes = options is None or options.get(
        "tt_enable_torch_fx_fusion_pass", True
    )
    if enable_fusion_passes:
        gm = run_fusion_passes(gm)

    # This is a temporary option to disable / enable composite ops
    # that will be removed once composite ops are more stable.
    # default to True if options are not given or if tt_enable_composite_ops is not present

    enable_composite_ops = options is None or options.get(
        "tt_enable_composite_ops", True
    )
    if enable_composite_ops:
        handle_composite_ops(gm)

    decompositions = populate_decompositions()

    program = torch.export.export(
        gm,
        tuple(example_inputs),
        strict=False,
    )
    program = program.run_decompositions(decompositions)

    compiled_graph = program.module()
    compiled_graph = insert_argument_type_markers(
        compiled_graph, program.graph_signature
    )
    compiled_graph = bypass_dtype_promotion_and_redundant_cast(
        compiled_graph, example_inputs
    )
    compiled_graph = bypass_redundant_getitem(compiled_graph)
    compiled_graph = bypass_assert_tensor_metadata(compiled_graph)

    # Recompile the GraphModule to ensure the modifications made by the above
    # passes are reflected during execution.
    compiled_graph.recompile()

    # Extract metadata from FX nodes in order to inject them into locs
    node_info = extract_nodes_info(compiled_graph)

    return compiled_graph, program.graph_signature, node_info


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
        legacy_compile_enabled: bool,
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

        # Whether to enable the legacy compile flow.
        # The following group of fields will only be used if the experimental flow is enabled.
        self.legacy_compile_enabled = legacy_compile_enabled
        self.params_and_consts = None
        self.compiled_graph = None

    # Extract the param and consts from the exported program.
    def _build_params_and_consts(self, ep: ExportedProgram) -> Tuple[torch.Tensor]:
        sig = ep.graph_signature

        # Export keeps a state dict for lifted params/buffers and a const dict for lifted constants.
        state = ep.state_dict
        constants = ep.constants

        # Map from placeholder name -> tensor.
        total_args = tuple()
        encountered_user_input = False
        for spec in sig.input_specs:
            # Kinds: CUSTOM_OBJ and TOKEN haven't been tested.
            # USER_INPUT will not exist in state_dict, it is passed in from the outside.
            if spec.kind == InputKind.USER_INPUT:
                encountered_user_input = True
                continue

            assert (
                not encountered_user_input
            ), "We expect user inputs to be last in the list of inputs."

            assert spec.target is not None, f"Spec target is None for spec {spec}"
            if spec.kind == InputKind.CONSTANT_TENSOR:
                arg = constants[spec.target]
            else:
                arg = state[spec.target]  # Handles: PARAMETER, BUFFER
            if arg.device.type != "xla":
                if spec.kind != InputKind.CONSTANT_TENSOR:
                    logger.warning(
                        f"Found an argument on non-XLA device which was not a lifted constant: {spec.target}. "
                        "Passing a non-XLA tensor to TT compile was likely not intended. Force moving the argument to XLA."
                    )
                arg = arg.to(
                    torch.device("xla")
                )  # Maybe it makes sense to modify the ep to avoid multiple moves of constants?
            total_args += (arg,)

        return total_args

    def _call_experimental_compile(self, *args):
        if self.compiled_graph is None:
            # To use the `optimized_mod` from `torch_xla` we need to have all of the arguments (user input, params, constants)
            # inlined in the function signature (torch calls this "lifting" the arguments). Exporting does this.
            program = torch.export.export(self.module, tuple(args), strict=False)

            # Collect the params and constants from the exported program.
            self.params_and_consts = self._build_params_and_consts(program)

            # Use `torch_xla` function to replace the graph module with the `optimized_mod`.
            # This helps us avoid tracing the graph on the subsequent model execution. On the next
            # invocation of forward - `optimized_mod` will just look up in its cache and execute the graph
            # without any tracing.
            self.compiled_graph = bridge.extract_compiled_graph(
                program.graph_module, self.params_and_consts + args
            )

        full_args = self.params_and_consts + args
        return self.compiled_graph(*full_args)

    def __call__(self, *args):
        if not self.legacy_compile_enabled:
            return self._call_experimental_compile(*args)

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


def _build_executor(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
) -> XLAExecutor:
    module, graph_signature, node_info = torch_pass_pipeline(
        gm, example_inputs, options
    )
    legacy_compile_default = False
    legacy_compile_enabled = legacy_compile_default
    if options:
        if "tt_experimental_compile" in options:
            print(
                'Warning: Experimental compile is now the default. As such, the "tt_experimental_compile" flag is deprecated.'
                'Honoring the flag, but please use "tt_legacy_compile" flag or no flag in the future.'
            )
            legacy_compile_enabled = not bool(options["tt_experimental_compile"])
        if "tt_legacy_compile" in options:
            legacy_compile_enabled = bool(options["tt_legacy_compile"])
    return XLAExecutor(module, graph_signature, node_info, legacy_compile_enabled)


def _has_fake_tensors(args: tuple) -> bool:
    for arg in args:
        if isinstance(arg, torch.Tensor) and is_fake(arg):
            return True
    return False


def _move_inputs_to_xla(args: tuple) -> tuple:
    device = torch.device("xla")
    moved_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg = mb_unwrap_functional_tensor(arg)
            if arg.device.type != "xla":
                arg = arg.to(device)
            moved_args.append(arg)
        else:
            moved_args.append(arg)
    return tuple(moved_args)


def _tt_aot_autograd_backend(gm, example_inputs, options=None):
    """
    AOTAutograd backend for TT/XLA compilation.

    The key challenge is that AOTAutograd traces the forward with FunctionalTensor
    wrappers, but XLA doesn't support FunctionalTensor. To work around this,
    we use FakeTensorMode to create fake tensors that have the same metadata as
    the XLA tensors but don't actually dispatch to XLA during tracing.

    This approach works uniformly for both inference and training.
    """
    decompositions = populate_decompositions()

    def fw_compiler(model, inputs):
        compiled_executor = None

        def fwd(*args):
            nonlocal compiled_executor
            if _has_fake_tensors(args):
                return model(*args)
            args = _move_inputs_to_xla(args)
            if compiled_executor is None:
                compiled_executor = _build_executor(model, args, options)
            return compiled_executor(*args)

        return make_boxed_func(fwd)

    # Use FakeTensorMode to create fake inputs that don't dispatch to XLA
    # This allows AOTAutograd's functionalization tracing to work without
    # hitting XLA's FunctionalTensor incompatibility
    with FakeTensorMode(allow_non_fake_inputs=True) as fake_mode:
        fake_example_inputs = []
        for inp in example_inputs:
            if isinstance(inp, torch.Tensor):
                # Create a fake tensor with the same properties but on meta device
                # This prevents any actual computation during tracing
                fake_inp = fake_mode.from_tensor(inp, static_shapes=True)
                fake_example_inputs.append(fake_inp)
            else:
                fake_example_inputs.append(inp)
        fake_example_inputs = tuple(fake_example_inputs)

        return aot_autograd(fw_compiler=fw_compiler, decompositions=decompositions)(
            gm, fake_example_inputs
        )


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=None):
    """TT backend for torch.compile."""
    return _tt_aot_autograd_backend(gm, example_inputs, options=options)
