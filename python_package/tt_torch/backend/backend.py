# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Iterable, Tuple

import torch
import torch.export
import torch_xla
import torch_xla.core.dynamo_bridge as bridge
from torch._dynamo import register_backend
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from functorch.compile import aot_module_simplified

from .decompositions import populate_decompositions
from .metadata_propagation import MetadataDispatchMode, extract_nodes_info
from .passes import (
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    handle_composite_ops,
    insert_argument_type_markers,
    replace_cpu_device_with_xla,
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
        options: dict[str, bool] | None = None,
    ):
        self.module = module
        self.signature = signature
        self.node_info = node_info
        self.options = options
        # Inject metadata if xla debug is enabled and node_info is not empty
        # We need xla debug to be enabled in order for torch-xla to inject metadata
        self.inject_metadata = os.environ.get("XLA_HLO_DEBUG", "0") == "1" and node_info

        # Whether to enable the legacy compile flow.
        self.legacy_compile_enabled = legacy_compile_enabled
        self.params_and_consts = None
        self.compiled_graph = None
        self.devices = None

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
                    print(
                        f"Found an argument on non-XLA device which was not a lifted constant: {spec.target}.\n"
                        "Passing a non-XLA tensor to TT compile was likely not intended. Force moving the argument to XLA."
                    )
                arg = arg.to(
                    torch.device("xla")
                )  # Maybe it makes sense to modify the ep to avoid multiple moves of constants?
            total_args += (arg,)

        return total_args

    def _compile_fx_graph(self, gm, args, run_decompositions=True):
        """Compile a graph module for XLA execution."""
        if run_decompositions:
            compiled_graph, _, _ = torch_pass_pipeline(gm, args, self.options)
        else:
            # Backward graph: run fusion and composite ops only, skip decompositions
            if self.options is None or self.options.get("tt_enable_torch_fx_fusion_pass", True):
                gm = run_fusion_passes(gm)
            if self.options is None or self.options.get("tt_enable_composite_ops", True):
                handle_composite_ops(gm)
            gm.recompile()
            compiled_graph = gm

        return bridge.extract_compiled_graph(compiled_graph, args)

    def _make_lazy_compiler(self, run_decompositions):
        """Create a lazy compiler that defers compilation until real tensors arrive."""
        def compiler(gm, example_inputs):
            compiled_fn = None

            def wrapper(*args):
                nonlocal compiled_fn
                if compiled_fn is None:
                    compiled_fn = self._compile_fx_graph(gm, args, run_decompositions)
                return compiled_fn(*args)

            return wrapper
        return compiler

    def _call_experimental_compile(self, *args):
        if self.compiled_graph is None:
            # Use AOT Autograd to handle forward/backward separation.
            # Forward: full pipeline with decompositions
            # Backward: fusion + composite ops only (decompositions break backward ops)
            self.compiled_graph = aot_module_simplified(
                self.module,
                args,
                fw_compiler=self._make_lazy_compiler(run_decompositions=True),
                bw_compiler=self._make_lazy_compiler(run_decompositions=False),
            )

        return self.compiled_graph(*args)

    def _flatten_tensors(self, x):
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, Iterable):
            out = []
            for e in x:
                out.extend(self._flatten_tensors(e))
            return out
        return []

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
            flat_out = self._flatten_tensors(output)
            if self.devices is None:
                devs = set()
                for t in flat_out:
                    if isinstance(t, torch.Tensor) and hasattr(t, "device"):
                        if t.device.type == "xla":
                            devs.add(str(t.device))
                self.devices = list(devs)
            torch_xla._XLAC._xla_sync_multi(list(output), self.devices, wait=False)
        else:
            # Some graphs have side effects not included in graph output.
            # In these cases we must call sync() to force materialization of non-user-output
            # tensors, eg. inplace static cache updates as OutputKind.USER_INPUT_MUTATION.
            # This causes buffer mutations to show up as graph outputs in MLIR.
            torch_xla.sync()

        return output


@register_backend(name="tt")
def xla_backend(gm, example_inputs, options=None):
    """TT backend for torch.compile."""
    legacy_compile_enabled = False
    if options:
        if "tt_experimental_compile" in options:
            print(
                'Warning: Experimental compile is now the default. As such, the "tt_experimental_compile" flag is deprecated.'
                'Honoring the flag, but please use "tt_legacy_compile" flag or no flag in the future.'
            )
            legacy_compile_enabled = not bool(options["tt_experimental_compile"])
        if "tt_legacy_compile" in options:
            legacy_compile_enabled = bool(options["tt_legacy_compile"])

    if not legacy_compile_enabled:
        # For experimental compile (AOT Autograd), pass the raw graph module.
        # The pass pipeline will run inside the AOT compiler with real tensors.
        from torch.export.graph_signature import ExportGraphSignature
        dummy_signature = ExportGraphSignature(input_specs=[], output_specs=[])
        return XLAExecutor(
            gm,
            dummy_signature,
            [],
            legacy_compile_enabled=False,
            options=options,
        )

    # Legacy compile: run pass pipeline upfront
    compiled_graph, sub_sig, sub_node_info = torch_pass_pipeline(
        gm, example_inputs, options
    )
    return XLAExecutor(
        compiled_graph,
        sub_sig,
        sub_node_info,
        legacy_compile_enabled=True,
    )
