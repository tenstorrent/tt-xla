# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple

import torch
import torch_xla
import torch_xla.core.dynamo_bridge as bridge
import torch_xla.runtime as xr
from functorch.compile import make_boxed_func
from torch._decomp import get_decompositions as get_aten_decompositions
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import detect_fake_mode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.tools_common import legalize_graph
from torch_xla.distributed.spmd import ShardingType
from ttxla_tools.logging import logger

from .decompositions import populate_decompositions
from .metadata_propagation import (
    MetadataDispatchMode,
    MetadataInterpreter,
    extract_nodes_info,
)
from .passes import (
    _build_normalized_fqn_lookup,
    _demangle_name,
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    handle_composite_ops,
    insert_argument_type_markers,
    rewrite_adaptive_avgpool_to_mean,
    run_fusion_passes,
)


def _classify_inputs_for_aot(
    gm: torch.fx.GraphModule,
    param_names: list[str],
    buffer_names: list[str],
    flat_name_to_original_fqn: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Classify inputs for the AOTAutograd path using captured module info.

    AOTAutograd lifts all parameters and buffers as function arguments
    (placeholders) in the forward graph.  The ordering convention is:
    parameters first (in named_parameters() order), then buffers
    (in named_buffers() order), then user inputs.
    """
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    num_params = len(param_names)
    num_buffers = len(buffer_names)

    input_type_map: dict[str, str] = {}
    name_map: dict[str, str] = {}

    for i, node in enumerate(placeholders):
        if i < num_params:
            input_type_map[node.name] = "parameter"
            mangled = param_names[i]
            name_map[node.name] = flat_name_to_original_fqn.get(mangled, mangled)
        elif i < num_params + num_buffers:
            # AOTAutograd functionalises mutations, so buffers in the forward
            # graph are never mutated in-place — safe to mark as constant.
            input_type_map[node.name] = "constant"
            mangled = buffer_names[i - num_params]
            name_map[node.name] = flat_name_to_original_fqn.get(mangled, mangled)
        else:
            input_type_map[node.name] = "input"
            name_map[node.name] = node.name

    return input_type_map, name_map


def _classify_from_module_state(
    gm: torch.fx.GraphModule,
    flat_name_to_original_fqn: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], bool]:
    """Classify inputs from module state for the non-AOTAutograd path.

    In this path the module still has its parameters and buffers as
    attributes (get_attr nodes). Placeholders are user inputs.
    """
    normalized_fqn_lookup = _build_normalized_fqn_lookup(flat_name_to_original_fqn)
    params = dict(gm.named_parameters())
    buffers = dict(gm.named_buffers())

    # Detect mutated buffers: look for copy_ ops whose destination is a buffer.
    mutated_buffers: set[str] = set()
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.copy_.default
            and node.args
        ):
            dest = node.args[0]
            if dest.op == "get_attr" and dest.target in buffers:
                mutated_buffers.add(dest.target)

    has_output_mutations = bool(mutated_buffers)

    input_type_map: dict[str, str] = {}
    name_map: dict[str, str] = {}

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            if node.target in params:
                input_type_map[node.target] = "parameter"
            elif node.target in buffers:
                if node.target in mutated_buffers:
                    input_type_map[node.target] = "input"
                else:
                    input_type_map[node.target] = "constant"
            name_map[node.target] = _demangle_name(node.target, normalized_fqn_lookup)
        elif node.op == "placeholder":
            input_type_map[node.name] = "input"
            name_map[node.name] = _demangle_name(node.name, normalized_fqn_lookup)

    return input_type_map, name_map, has_output_mutations


def _lift_get_attr_to_placeholders(
    gm: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, tuple[torch.Tensor, ...]]:
    """Lift get_attr (parameter/buffer) nodes to placeholder inputs.

    Modifies *gm* in-place.  New placeholders are prepended before existing
    ones so the calling convention becomes ``(params..., user_inputs...)``.

    Returns the tuple of real tensors that correspond to the new placeholders,
    in the same order.
    """
    get_attr_nodes = gm.graph.find_nodes(op="get_attr")
    if not get_attr_nodes:
        return gm, ()

    # Find insertion point: before the first existing placeholder.
    first_placeholder = next((n for n in gm.graph.nodes if n.op == "placeholder"), None)

    tensors: list[torch.Tensor] = []
    for ga_node in get_attr_nodes:
        tensor = getattr(gm, ga_node.target)
        if tensor.device.type != "xla":
            logger.warning(
                f"Lifting non-XLA tensor to placeholder: {ga_node.target}. "
                "Force moving to XLA."
            )
            tensor = tensor.to(torch.device("xla"))
        tensors.append(tensor)

        with gm.graph.inserting_before(first_placeholder):
            new_ph = gm.graph.placeholder(f"lifted_{ga_node.target}")

        ga_node.replace_all_uses_with(new_ph)
        gm.graph.erase_node(ga_node)

    gm.recompile()
    return gm, tuple(tensors)


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
    input_classification: tuple[dict[str, str], dict[str, str], bool] | None = None,
) -> Tuple[torch.fx.GraphModule, bool, dict[str, str]]:
    """Run the torch FX pass pipeline.

    Args:
        gm: The graph module to transform.
        example_inputs: Example inputs for the graph module.
        options: Backend options.
        input_classification: Pre-built (input_type_map, name_map, has_output_mutations)
            tuple. When provided (AOTAutograd path), decompositions are skipped
            (assumed to have been applied by AOTAutograd already). When None
            (non-AOT path), make_fx is used to apply decompositions and
            inputs are classified from the module's parameter/buffer state.

    Returns:
        (compiled_graph, has_output_mutations, node_info)
    """

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

    if input_classification is not None:
        # AOTAutograd path: decompositions already applied, classification provided.
        compiled_graph = gm
        input_type_map, name_map, has_output_mutations = input_classification
    else:
        # Non-AOTAutograd path: apply decompositions via make_fx and classify
        # inputs from the module's own parameter/buffer state.
        decompositions = populate_decompositions()

        # When torch.compile traces a model, it flattens the module hierarchy
        # and mangles parameter names (e.g., "model.layers.0.weight" becomes
        # something like "L__self___model_layers___0___weight"). Dynamo stores
        # a reverse mapping in GraphModule.meta so we can recover the original
        # names for MLIR argument names (ttir.name).
        # Grab this before make_fx, as it creates a fresh GraphModule.
        flat_name_to_original_fqn = gm.meta.get("dynamo_flat_name_to_original_fqn", {})

        # Re-trace the graph module with decompositions applied.
        # Dynamo provides real tensors as example_inputs, so tracing_mode="real"
        # is safe and avoids FakeTensor complications with parameters.
        compiled_graph = make_fx(
            gm, decomposition_table=decompositions, tracing_mode="real"
        )(*example_inputs)

        input_type_map, name_map, has_output_mutations = _classify_from_module_state(
            compiled_graph, flat_name_to_original_fqn
        )

    compiled_graph = insert_argument_type_markers(
        compiled_graph, input_type_map, name_map
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

    return compiled_graph, has_output_mutations, node_info


def _mark_unsharded_args_replicated(args: Tuple[torch.Tensor]) -> None:
    """Mark unsharded XLA tensors as REPLICATED when running in SPMD mode.

    In SPMD mode, torch_xla's InputCollector propagates sharding annotations
    during graph capture and marks unsharded tensors as '<replicated>'. At
    runtime, freshly created tensors (e.g. input_ids, cache_position) carry no
    sharding annotation and return '' from _get_xla_sharding_spec. Explicitly
    marking them REPLICATED keeps their sharding spec consistent with what was
    recorded at capture time and prevents spurious retracing in dynamo_bridge.
    """
    if not xr.is_spmd():
        return
    replicated = torch_xla._XLAC.OpSharding([], [], [], ShardingType.REPLICATED)
    for arg in args:
        if isinstance(arg, torch.Tensor) and not torch_xla._XLAC._get_xla_sharding_spec(
            arg
        ):
            torch_xla._XLAC._xla_mark_sharding(arg, replicated)


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
        has_output_mutations: bool,
        node_info: dict[str, str],
        legacy_compile_enabled: bool,
    ):
        self.module = module
        self.has_output_mutations = has_output_mutations
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
            # bridge.extract_compiled_graph needs all inputs (params + user)
            # as placeholders in the graph and as a flat tensor tuple.
            # Lift any remaining get_attr nodes (params/buffers) to
            # placeholders so they become part of the flat input list.
            self.module, self.params_and_consts = _lift_get_attr_to_placeholders(
                self.module
            )

            # Rebuild the graph in topological order to fix up prev/next
            # pointers that can confuse the DynamoBridge Partitioner.
            legalize_graph(self.module)

            # Use `torch_xla` function to replace the graph module with the `optimized_mod`.
            # This helps us avoid tracing the graph on the subsequent model execution. On the next
            # invocation of forward - `optimized_mod` will just look up in its cache and execute the graph
            # without any tracing.
            self.compiled_graph = bridge.extract_compiled_graph(
                self.module, self.params_and_consts + args
            )

        full_args = self.params_and_consts + args
        # Ensure unsharded tensors are marked REPLICATED in SPMD mode so their
        # sharding spec matches what was recorded at graph capture time. This is a temporary workaround
        # until a change is made in torch-xla to automatically mark unsharded tensors as REPLICATED at runtime in SPMD mode.
        _mark_unsharded_args_replicated(full_args)
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

        if not self.has_output_mutations:
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
    input_classification: tuple[dict[str, str], dict[str, str], bool] | None = None,
):
    module, has_output_mutations, node_info = torch_pass_pipeline(
        gm, example_inputs, options, input_classification
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

    return XLAExecutor(module, has_output_mutations, node_info, legacy_compile)


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

    # Capture parameter/buffer names and the FQN mapping from the original
    # module *before* AOTAutograd lifts them into flat placeholder inputs.
    # AOTAutograd orders the lifted inputs as: params, buffers, user inputs.
    param_names = [name for name, _ in gm.named_parameters()]
    buffer_names = [name for name, _ in gm.named_buffers()]
    flat_name_to_original_fqn = gm.meta.get("dynamo_flat_name_to_original_fqn", {})

    # Merge all decompositions so that AOTAutograd applies them during
    # tracing — this eliminates the need for torch.export in
    # torch_pass_pipeline for the AOT path.
    aot_decompositions = populate_decompositions()
    # There is a well known bug in our stack that stablehlo.batch_norm_training doesn't shard properly in multichip scenarios.
    # TorchXLA uses stablehlo.batch_norm_training for it's implementation of torch layernorm,
    # but that gets decomposed by decompositions inside torch_pass_pipeline anyway so we don't observe it.
    # In multichip scenarios, for reasons unknown, now the layernorm to batchnorm(specifically _native_batch_norm_legit.no_stats)
    # conversion happens before the fx module ever reaches us.
    # So we manually decompose batch norm early inside aot_autograd to avoid the bug. This could be a perf pitfall.
    # THIS IS A HACK https://github.com/tenstorrent/tt-xla/issues/3533
    aot_decompositions.update(
        get_aten_decompositions(
            [
                torch.ops.aten._native_batch_norm_legit.no_stats,
            ]
        )
    )

    def fw_compiler_boxed(fw_gm, fw_example_inputs):
        # Build input classification from the captured module info.
        # AOTAutograd has already applied decompositions, so torch_pass_pipeline
        # will skip torch.export entirely when classification is provided.
        input_type_map, name_map = _classify_inputs_for_aot(
            fw_gm, param_names, buffer_names, flat_name_to_original_fqn
        )
        # AOTAutograd functionalises mutations — the forward graph has only
        # user outputs (no buffer mutation outputs).
        classification = (input_type_map, name_map, False)
        return make_boxed_func(
            fw_compiler(fw_gm, fw_example_inputs, options, classification)
        )

    # Dynamo creates FakeTensorMode(allow_non_fake_inputs=False) and stores it in
    # TracingContext.  aot_module_simplified retrieves it via detect_fake_mode and
    # uses it for all graph tracing.  Real tensor constants in the graph (e.g. empty
    # initialiser tensors in models like Qwen3) then hit "Please convert all Tensors
    # to FakeTensors first".  Temporarily enable allow_non_fake_inputs so those
    # constants are handled as FakeTensors instead of raising.
    _active_fake_mode = detect_fake_mode([])
    _saved_allow = None
    if _active_fake_mode is not None and not _active_fake_mode.allow_non_fake_inputs:
        _saved_allow = False
        _active_fake_mode.allow_non_fake_inputs = True
    try:
        return aot_autograd(
            fw_compiler=fw_compiler_boxed, decompositions=aot_decompositions
        )(gm, example_inputs)
    finally:
        if _saved_allow is not None:
            _active_fake_mode.allow_non_fake_inputs = _saved_allow


@register_backend(name="tt")
def tt_backend(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None = None,
):
    use_aot_autograd = (
        bool(options.get("tt_use_aot_autograd", True)) if options else True
    )
    if use_aot_autograd:
        return aot_backend(gm, example_inputs, options)
    else:
        return fw_compiler(gm, example_inputs, options)
