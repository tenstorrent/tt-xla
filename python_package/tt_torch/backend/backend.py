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


def _patch_dynamo_bridge():
    """Patch partition_fx_graph_for_cpu_fallback to fix fused_0.xla_args bug.

    The bug: InputCollector stops at the 'output' node in the partitioned graph,
    so fused submodules whose call_module node appears after the 'output' node
    (or when InputCollector raises before reaching them) never have xla_args set.

    The fix: after InputCollector.run(), walk the partitioned graph and for any
    fused submodule missing xla_args, reconstruct the args from the placeholder
    values so extract_internal can proceed.
    """
    import torch_xla._dynamo.dynamo_bridge as db
    import torch as _torch

    _orig = db.partition_fx_graph_for_cpu_fallback

    def _patched(xla_model, xla_args, all_xla_args, all_xla_args_tensor_only):
        # Run the original implementation
        try:
            return _orig(xla_model, xla_args, all_xla_args, all_xla_args_tensor_only)
        except AttributeError as e:
            if "xla_args" not in str(e):
                raise
            # Fall through to recovery logic below

        # Recovery: the original implementation failed because InputCollector
        # did not set xla_args on some fused submodule. Re-run it but manually
        # set xla_args for any fused submodule that was missed.
        cloned_args = [
            _torch.clone(a) if isinstance(a, _torch.Tensor) else a
            for a in all_xla_args
        ]

        collector = db.UnsupportedNodesCollector(xla_model)
        collector.run(*xla_args)
        unsupported_nodes = collector.get_unsupported_nodes()

        db._clear_pending_irs_on_args(all_xla_args_tensor_only, cloned_args)
        import torch_xla as _txla
        _txla._XLAC._clear_pending_irs(str(_txla.device()))

        import operator
        class _XlaSupport(_torch.fx.passes.operator_support.OperatorSupport):
            def is_node_supported(self, submodules, node):
                return node.op in ["call_function", "call_module", "call_method"] and (
                    node not in unsupported_nodes or node.target == operator.getitem
                )

        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        partitioner = CapabilityBasedPartitioner(
            xla_model, _XlaSupport(), allows_single_node_partition=True
        )
        partitions = partitioner.propose_partitions()

        from torch_xla._dynamo.dynamo_bridge import topo_sort
        for p in partitions:
            p.nodes = topo_sort(p.nodes)

        partitioned_graph = partitioner.fuse_partitions(partitions)

        # Build a placeholder→value map so we can populate xla_args manually
        ph_vals = {}
        for i, node in enumerate(
            n for n in partitioned_graph.graph.nodes if n.op == "placeholder"
        ):
            ph_vals[node] = xla_args[i] if i < len(xla_args) else None

        # Run InputCollector; ignore exceptions (it may fail on in-place ops)
        try:
            db.InputCollector(partitioned_graph).run(*xla_args)
        except Exception:
            pass

        # For any fused module still missing xla_args, populate from ph_vals
        for node in partitioned_graph.graph.nodes:
            if node.op == "call_module" and "fused_" in node.name:
                submod = getattr(partitioned_graph, node.name, None)
                if submod is not None and not hasattr(submod, "xla_args"):
                    submod.xla_args = tuple(
                        ph_vals.get(a, a) if isinstance(a, _torch.fx.Node) else a
                        for a in node.args
                    )

        db._clear_pending_irs_on_args(all_xla_args_tensor_only, cloned_args)

        # Compile each fused submodule
        for node in list(partitioned_graph.graph.nodes):
            if node.op == "call_module" and "fused_" in node.name:
                fused_module = getattr(partitioned_graph, node.name)
                partitioned_graph.delete_submodule(node.target)
                with partitioned_graph.graph.inserting_after(node):
                    new_node = partitioned_graph.graph.call_function(
                        db.extract_internal(fused_module), node.args, None
                    )
                    node.replace_all_uses_with(new_node)
                partitioned_graph.graph.erase_node(node)

        partitioned_graph.recompile()
        return partitioned_graph

    db.partition_fx_graph_for_cpu_fallback = _patched
    _patch_dynamo_bridge._applied = True


_patch_dynamo_bridge._applied = False
_patch_dynamo_bridge()

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
) -> Tuple[torch.fx.GraphModule, torch.export.ExportGraphSignature, dict[str, str]]:

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
        node_info: dict[str, str],
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
            # To use the `optimized_mod` from `torch_xla` we need to have all of the arguments (user input, params, constants)
            # inlined in the function signature (torch calls this "lifting" the arguments). Exporting does this.
            program = torch.export.export(self.module, tuple(args), strict=False)

            # we observe that nodes in the fx graph can have inconsistent prev/next pointers.
            # specifically, after invoking `torch.export.export` as part of torch_pass_pipeline,
            # we observed in one case that a "placeholder=target['c_lifted_tensor_1']" node has it's successor set to "get_attr=target['_tensor_constant0']"
            # a node which doesn't appear at all when interating over the fx graph directly(and whose successor is the real successor of the node as per the fx graph)
            # Calling legalize_graph rebuilds the graph in topological order(from usage information), and fixes up the prev/next pointers in the process - which fixes our issue.
            # All this is a problem because DynamoBridge Partitioner can get confused by wrong next nodes and partition the graph in a way which fails to execute.
            legalize_graph(program.graph_module)

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
            # Use MetadataInterpreter + MetadataDispatchMode to correctly track metadata
            # even when FX nodes decompose into multiple aten operations at dispatch time.
            # MetadataInterpreter sets a context variable for each FX node, and
            # MetadataDispatchMode reads it to attach the correct metadata to each dispatch.
            with MetadataDispatchMode():
                interp = MetadataInterpreter(self.module, self.node_info)
                output = interp.run(*args)
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

            # Two hacks to make AOTAutograd with legacy compile work:
            # 1) Filter out non-tensor outputs (e.g. None values from aot_autograd
            # backward graphs where some inputs don't require gradients).
            output_tensors = [o for o in output if isinstance(o, torch.Tensor)]
            # 2) When AOTAutograd is used, the forward graph module has no
            # state_dict (parameters are lifted as inputs), so self.devices
            # may be empty. Derive devices from the output tensors instead.
            devices = self.devices
            if not devices and output_tensors:
                devices = list({t.device.type for t in output_tensors})
            torch_xla._XLAC._xla_sync_multi(output_tensors, devices, wait=False)
        else:
            # Some graphs have side effects not included in graph output.
            # In these cases we must call sync() to force materialization of non-user-output
            # tensors, eg. inplace static cache updates as OutputKind.USER_INPUT_MUTATION.
            # This causes buffer mutations to show up as graph outputs in MLIR.
            torch_xla.sync()

        return output


def fw_compiler(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
):
    module, graph_signature, node_info = torch_pass_pipeline(
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

    return XLAExecutor(module, graph_signature, node_info, legacy_compile)


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
