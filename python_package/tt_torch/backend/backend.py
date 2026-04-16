# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import torch
import torch.export
import torch_xla
import torch_xla.core.dynamo_bridge as bridge
import torch_xla.runtime as xr
from functorch.compile import make_boxed_func
from torch._decomp import get_decompositions as get_aten_decompositions
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd, fake_tensor_unsupported
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.passes.tools_common import legalize_graph
from torch_xla.distributed.spmd import ShardingType
from ttxla_tools.logging import logger

from ..utils import is_torch_2_10_or_newer
from .decompositions import populate_decompositions
from .metadata_propagation import (
    MetadataDispatchMode,
    MetadataInterpreter,
    extract_nodes_info,
)
from .passes import (
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    handle_composite_ops,
    insert_argument_type_markers,
    rewrite_adaptive_avgpool_to_mean,
    run_fusion_passes,
)


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
) -> Tuple[
    torch.fx.GraphModule,
    torch.export.ExportGraphSignature,
    dict[str, str],
    ExportedProgram,
]:

    # Run fusion passes to detect and fuse multi-op patterns
    # This runs before composite_ops to allow fused patterns to be wrapped as composites
    enable_fusion_passes = options is None or options.get(
        "tt_enable_torch_fx_fusion_pass", True
    )
    if enable_fusion_passes:
        run_fusion_passes(gm)

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

    # Get unlifted module BEFORE modifying program.graph_module so that the
    # unlifting process sees the original graph structure.
    compiled_graph = program.module()

    # When torch.compile traces a model, it flattens the module hierarchy and
    # mangles parameter names (e.g., "model.layers.0.weight" becomes something
    # like "L__self___model_layers___0___weight"). Dynamo stores a reverse
    # mapping in GraphModule.meta so we can recover the original names. We pass
    # this to insert_argument_type_markers so that MLIR argument names (ttir.name)
    # match the original model's state_dict keys.
    flat_name_to_original_fqn = compiled_graph.meta.get(
        "dynamo_flat_name_to_original_fqn", {}
    )
    compiled_graph = insert_argument_type_markers(
        compiled_graph, program.graph_signature, flat_name_to_original_fqn
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

    return compiled_graph, program.graph_signature, node_info, program


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
        node_info: dict[str, str],
        legacy_compile_enabled: bool,
        exported_program: ExportedProgram | None = None,
    ):
        self.module = module
        self.signature = signature
        self.node_info = node_info
        self.exported_program = exported_program
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

        # In multi-chip configurations with PyTorch 2.10+, torch.export.export()
        # creates pending TransferFromDevice ops for every sharded XLA parameter.
        # The experimental compile path calls extract_compiled_graph which
        # internally runs torch_xla.sync(), executing ALL pending ops at once
        # and flooding DRAM with replicated copies of each sharded param -> OOM.
        # The legacy path uses _xla_sync_multi(outputs) which only syncs the
        # output tensors, keeping sharded params as-is inside the computation.
        if (
            is_torch_2_10_or_newer()
            and not legacy_compile_enabled
            and xr.global_runtime_device_count() > 1
        ):
            logger.info(
                "Multi-chip detected on torch >= 2.10 (device_count={}), using "
                "legacy compile to avoid ReplicateShardedData flood in "
                "experimental path.",
                xr.global_runtime_device_count(),
            )
            legacy_compile_enabled = True

        if not legacy_compile_enabled and any(
            spec.kind == OutputKind.USER_INPUT_MUTATION
            for spec in self.signature.output_specs
        ):
            logger.info(
                "User-input mutation outputs detected, using legacy compile to "
                "preserve torch.compile input alias semantics."
            )
            legacy_compile_enabled = True

        # Whether to enable the legacy compile flow.
        # The following group of fields will only be used if the experimental flow is enabled.
        self.legacy_compile_enabled = legacy_compile_enabled
        self.params_and_consts = None
        self.compiled_graph = None
        self._lifted_graph_prepared = False

        # Pre-compute names of mutated buffers (e.g., KV-cache).
        # In the legacy compile path, the FX graph mutates module buffers
        # in-place via copy_ ops, but they are NOT included in the graph's
        # return value. We must sync them explicitly to avoid duplicating
        # the backbone computation in a separate XLA graph.
        self._mutated_buffer_names = []
        if self.legacy_compile_enabled:
            for spec in self.signature.output_specs:
                if spec.kind == OutputKind.BUFFER_MUTATION:
                    self._mutated_buffer_names.append(spec.target)
                elif spec.kind == OutputKind.USER_INPUT_MUTATION:
                    self._mutated_buffer_names.append(spec.target)
            if self._mutated_buffer_names:
                logger.info(
                    "Will sync {} mutated buffers alongside outputs "
                    "to prevent graph duplication.",
                    len(self._mutated_buffer_names),
                )

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

    def _apply_passes_to_lifted_graph(self):
        """Apply FX passes to the lifted graph module from the ExportedProgram.

        The lifted graph has params as placeholder nodes (instead of get_attr).
        insert_argument_type_markers handles both representations thanks to
        the dual-dict lookup (get_attr_target_type_dict + placeholder_target_type_dict).
        """
        if self._lifted_graph_prepared:
            return
        gm = self.exported_program.graph_module
        fqn = gm.meta.get("dynamo_flat_name_to_original_fqn", {})
        insert_argument_type_markers(gm, self.signature, fqn)
        bypass_dtype_promotion_and_redundant_cast(gm, [])
        bypass_redundant_getitem(gm)
        bypass_assert_tensor_metadata(gm)
        gm.recompile()
        self._lifted_graph_prepared = True

    def _call_experimental_compile(self, *args):
        # Move any CPU tensors in args to XLA. Some model attributes (e.g.
        # detection grid tensors in YOLOP) are not registered buffers, so
        # AOTAutograd lifts them as graph inputs but they remain on CPU.
        # Passing CPU tensors to extract_compiled_graph causes graph breaks.
        moved_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.device.type != "xla":
                logger.warning(
                    f"Found an argument on non-XLA device: {arg}. "
                    "Passing a non-XLA tensor to TT compile was likely not intended. Force moving the argument to XLA."
                )
                arg = arg.to(torch.device("xla"))
            moved_args.append(arg)
        args = tuple(moved_args)
        if self.compiled_graph is None:
            if self.exported_program is not None:
                # Reuse the ExportedProgram from torch_pass_pipeline. A second
                # torch.export.export() call would trigger TransferFromDevice for
                # every sharded XLA parameter, compiling and executing a separate
                # ReplicateShardedData graph for each one and exhausting DRAM.
                program = self.exported_program
                self._apply_passes_to_lifted_graph()
                legalize_graph(program.graph_module)
                self.params_and_consts = self._build_params_and_consts(program)

                self.compiled_graph = bridge.extract_compiled_graph(
                    program.graph_module, self.params_and_consts + args
                )
            else:
                # Fallback: re-export when no pre-exported program is available.
                program = torch.export.export(self.module, tuple(args), strict=False)
                legalize_graph(program.graph_module)
                self.params_and_consts = self._build_params_and_consts(program)

                self.compiled_graph = bridge.extract_compiled_graph(
                    program.graph_module, self.params_and_consts + args
                )

        full_args = self.params_and_consts + args

        return self.compiled_graph(*full_args)

    def __call__(self, *args):
        if not self.legacy_compile_enabled:
            return self._call_experimental_compile(*args)

        if self.inject_metadata:
            # Use MetadataInterpreter + MetadataDispatchMode to correctly track metadata
            # even when FX nodes decompose into multiple aten operations at dispatch time.
            # MetadataInterpreter sets a context variable for each FX node, and
            # MetadataDispatchMode reads it to attach the correct metadata to each dispatch.
            with MetadataDispatchMode():
                interp = MetadataInterpreter(self.module, self.node_info)
                output = interp.run(*args)
        else:
            output = self.module(*args)
        # Sync output tensors AND mutated buffers (e.g., KV-cache).
        # The FX graph mutates buffers in-place via copy_ ops as side effects,
        # but they are NOT part of the return value. Without syncing them here,
        # their lazy ops form a separate XLA graph that re-computes the entire
        # backbone.
        output_tensors = [o for o in output if isinstance(o, torch.Tensor)]
        if self._mutated_buffer_names:
            for buf in self.module.buffers():
                output_tensors.append(buf)
        devices = self.devices
        if not devices and output_tensors:
            devices = list({t.device.type for t in output_tensors})
        torch_xla._XLAC._xla_sync_multi(output_tensors, devices, wait=False)

        return output


def fw_compiler(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
):
    # Dump the FX graph to see copy_ ops and output structure
    logger.debug("=== FX GraphModule from Dynamo ===")
    for node in gm.graph.nodes:
        if node.op == "output" or "copy" in str(node.target):
            logger.debug(
                "  {} op={} target={} args={}",
                node.name,
                node.op,
                node.target,
                [str(a) for a in node.args[:5]],
            )
    logger.debug("=== END FX Graph dump ===")

    module, graph_signature, node_info, exported_program = torch_pass_pipeline(
        gm, example_inputs, options
    )

    legacy_compile = False
    if options:
        if "tt_experimental_compile" in options:
            print(
                'Warning: Experimental compile is now the default. As such, the "tt_experimental_compile" flag is deprecated.'
                'Honoring the flag, but please use "tt_legacy_compile" flag or no flag in the future.'
            )
            legacy_compile = not bool(options["tt_experimental_compile"])
        if "tt_legacy_compile" in options:
            legacy_compile = bool(options["tt_legacy_compile"])

    return XLAExecutor(
        module, graph_signature, node_info, legacy_compile, exported_program
    )


def aot_backend(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
):
    """AOTAutograd backend: run decompositions and trace through aot_autograd with _fw_compiler."""
    # Rewrite AdaptiveAvgPool1d/2d(1) to torch.mean before AOTAutograd tracing.
    # There is a Torch/TorchXLA bug where fakified XLA tensors fault in AdaptiveAveragePool, because of an as_strided_ call
    # THIS IS A HACK https://github.com/tenstorrent/tt-xla/issues/3549
    gm = rewrite_adaptive_avgpool_to_mean(gm)

    # There is a well known bug in our stack that stablehlo.batch_norm_training doesn't shard properly in multichip scenarios.
    # TorchXLA uses stablehlo.batch_norm_training for it's implementation of torch layernorm,
    # but that gets decomposed by decompositions inside torch_pass_pipeline anyway so we don't observe it.
    # In multichip scenarios, for reasons unknown, now the layernorm to batchnorm(specifically _native_batch_norm_legit.no_stats)
    # conversion happens before the fx module ever reaches us.
    # So we manually decompose batch norm early inside aot_autograd to avoid the bug. This could be a perf pitfall.
    # THIS IS A HACK https://github.com/tenstorrent/tt-xla/issues/3533
    aot_decompositions = get_aten_decompositions(
        [
            torch.ops.aten._native_batch_norm_legit.no_stats,
        ]
    )

    @fake_tensor_unsupported  # see https://github.com/tenstorrent/tt-xla/issues/3572
    def fw_compiler_boxed(gm, example_inputs):
        return make_boxed_func(fw_compiler(gm, example_inputs, options))

    return aot_autograd(
        fw_compiler=fw_compiler_boxed, decompositions=aot_decompositions
    )(gm, example_inputs)


@register_backend(name="tt")
def tt_backend(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None = None,
):
    use_aot_autograd = (
        bool(options.get("tt_use_aot_autograd", False)) if options else False
    )
    if use_aot_autograd:
        return aot_backend(gm, example_inputs, options)
    else:
        return fw_compiler(gm, example_inputs, options)
