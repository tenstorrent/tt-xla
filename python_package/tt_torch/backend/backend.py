# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import re
from typing import Tuple

import torch
import torch._functorch.config as _functorch_config
import torch.export
import torch_xla
import torch_xla.core.dynamo_bridge as bridge
import torch_xla.runtime as xr
from functorch.compile import make_boxed_func, min_cut_rematerialization_partition
from torch._decomp import get_decompositions as get_aten_decompositions
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import detect_fake_mode
from torch._subclasses.fake_tensor import unset_fake_temporarily

# Force the min-cut partitioner to never save aten.view/permute/transpose/etc.
# A saved view of a parameter is a parameter-sized device copy for nothing -
# the parameter is already a backward input and the view is free to recompute.
_functorch_config.recompute_views = True
from torch.export import ExportedProgram
from torch.export.graph_signature import (
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TensorArgument,
)
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
    _normalize_fx_name,
    bypass_assert_tensor_metadata,
    bypass_dtype_promotion_and_redundant_cast,
    bypass_redundant_getitem,
    fold_view_bmm_view_to_einsum,
    handle_composite_ops,
    insert_argument_type_markers,
    rewrite_adaptive_avgpool_to_mean,
    rewrite_interpolate_to_matmul,
    run_fusion_passes,
)


def _relocate_lifted_constants_to_xla(gm: torch.fx.GraphModule) -> None:
    """Move the CPU constant buffers AOTAutograd lifted onto ``gm`` to XLA.

    AOTAutograd interns Python-literal tensors created mid-trace (e.g. a
    ``math.sqrt(hidden_size)`` scale) as registered buffers named
    ``_tensor_constant<N>``, left on CPU and surfacing as ``get_attr`` nodes.
    torch-xla's ``UnsupportedNodesCollector`` runs each node with real args,
    sees the CPU result, and ``CapabilityBasedPartitioner`` then splits the
    graph at that boundary — a host roundtrip that also blocks downstream
    fusion (e.g. RMSNorm in TTIRFusing).

    AOTAutograd keeps a ``FakeTensorMode`` on the dispatch stack through the
    fw_compiler callback, so a bare ``.to("xla")`` here would yield a
    FakeTensor; ``unset_fake_temporarily()`` pops that mode so the transfer
    produces a real XLA tensor. Rebinding via ``_buffers[name] = ...`` keeps
    nn.Module's ``__setattr__`` out of the loop.
    """
    xla_device = torch.device("xla")
    with unset_fake_temporarily():
        for mod in gm.modules():
            for name, value in list(mod._buffers.items()):
                if isinstance(value, torch.Tensor) and value.device.type != "xla":
                    logger.info(
                        f"[aot_backend] moving CPU constant buffer "
                        f"'{type(mod).__name__}.{name}' "
                        f"(dtype={value.dtype}, shape={tuple(value.shape)}) to XLA"
                    )
                    mod._buffers[name] = value.to(xla_device)


def _fetch_attr(mod: torch.nn.Module, target: str):
    """Resolve a (possibly dotted) get_attr target to its value, or None."""
    obj = mod
    for atom in target.split("."):
        if not hasattr(obj, atom):
            return None
        obj = getattr(obj, atom)
    return obj


def _constant_get_attr_specs(gm: torch.fx.GraphModule) -> list[InputSpec]:
    """Return input specs for every tensor-valued get_attr baked into ``gm``.

    AOTAutograd's make_fx tracer bakes constant tensors into the GraphModule as
    ``get_attr`` nodes rather than placeholders.  Those don't get a signature
    entry from the placeholder loop, so without an entry
    insert_argument_type_markers can't tag them and tt-mlir defaults the
    StableHLO argument to ``ttcore.argument_type = input`` — which blocks
    consteval folding and downstream fusions (e.g. RMSNorm in TTIRFusing).

    The tracer uses *several* name schemes for these get_attrs:
      * ``_tensor_constant<N>`` — a plain tensor literal promoted during tracing
        (e.g. gemma's ``torch.tensor(math.sqrt(hidden_size))``);
      * ``_param_constant<N>`` — an nn.Parameter not found in named_parameters()
        (PythonKeyTracer.create_arg);
      * a reused module FQN — when the constant is already a module attribute.
    Rather than enumerate names (brittle, and not provably exhaustive), classify
    by *what the node holds*.  This is safe because a tensor get_attr is always
    an immutable baked value: a user input is a placeholder (never a get_attr),
    and AOTAutograd functionalises every in-place mutation out-of-place and
    writes genuine input/buffer mutations back to *placeholders* via copy_
    epilogues — so a get_attr is never a mutation target.  We therefore tag an
    nn.Parameter value ``parameter`` and any other tensor ``constant`` (both
    consteval-eligible).  The non-tensor get_attrs the tracer can emit
    (``_torchbind_obj``, ``_tree_spec_constant``, ``_opaque_obj``) have
    non-Tensor values and are correctly skipped.
    """
    specs = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        value = _fetch_attr(gm, str(node.target))
        if not isinstance(value, torch.Tensor):
            continue
        kind = (
            InputKind.PARAMETER
            if isinstance(value, torch.nn.Parameter)
            else InputKind.CONSTANT_TENSOR
        )
        specs.append(
            InputSpec(
                kind=kind,
                arg=TensorArgument(name=node.name),
                target=str(node.target),
            )
        )
    return specs


def _build_aot_graph_signature(
    gm: torch.fx.GraphModule,
    param_names: list[str],
    buffer_names: list[str],
    flat_name_to_original_fqn: dict[str, str],
) -> ExportGraphSignature:
    """Build an ExportGraphSignature for the AOTAutograd forward graph.

    AOTAutograd lifts all parameters and buffers as placeholder arguments.
    The ordering convention is: params first, then buffers, then user inputs.

    Mutation detection uses AOTAutograd's own ViewAndMutationMeta
    (TracingContext.fw_metadata) — the same functionalization-based
    analysis that torch.export relies on.  When a buffer is mutated,
    we emit a BUFFER_MUTATION output spec so that
    insert_argument_type_markers classifies it as "input" (not
    hoisted into consteval).
    """
    tracing_ctx = torch._guards.TracingContext.try_get()
    assert tracing_ctx is not None, (
        "_build_aot_graph_signature must be called from within AOTAutograd's "
        "compiler callback (TracingContext not found)"
    )
    fw_metadata = tracing_ctx.fw_metadata
    assert fw_metadata is not None, (
        "TracingContext.fw_metadata is None — AOTAutograd should have set "
        "it before calling the compiler"
    )

    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    num_params = len(param_names)
    num_buffers = len(buffer_names)

    # Build a normalized lookup from flat_name_to_original_fqn so we can
    # match param/buffer names even when the mangling differs (e.g.
    # named_parameters() includes "_modules" segments that Dynamo omits).
    normalized_fqn_lookup = _build_normalized_fqn_lookup(flat_name_to_original_fqn)

    def _resolve_fqn(mangled: str) -> str:
        """Resolve a mangled param/buffer name to its clean FQN."""
        # Direct lookup first.
        result = flat_name_to_original_fqn.get(mangled)
        if result is not None:
            return result
        # Normalized lookup (collapses underscores, strips prefixes).
        normalized = _normalize_fx_name(mangled)
        result = normalized_fqn_lookup.get(normalized)
        if result is not None:
            return result
        # named_parameters() includes "_modules" in the path for ModuleList
        # children, but Dynamo's flat_name_to_original_fqn keys omit it.
        # Strip "_modules" segments and retry.
        stripped = re.sub(r"_modules_", "_", normalized)
        result = normalized_fqn_lookup.get(stripped)
        if result is not None:
            return result
        return mangled

    input_specs = []
    mutated_buffer_targets = set()

    for i, node in enumerate(placeholders):
        if i < num_params:
            mangled = param_names[i]
            target = _resolve_fqn(mangled)
            input_specs.append(
                InputSpec(
                    kind=InputKind.PARAMETER,
                    arg=TensorArgument(name=node.name),
                    target=target,
                )
            )
        elif i < num_params + num_buffers:
            mangled = buffer_names[i - num_params]
            target = _resolve_fqn(mangled)
            is_mutated = fw_metadata.input_info[i].mutates_data
            input_specs.append(
                InputSpec(
                    kind=InputKind.BUFFER,
                    arg=TensorArgument(name=node.name),
                    target=target,
                    persistent=True,
                )
            )
            if is_mutated:
                mutated_buffer_targets.add(target)
        else:
            input_specs.append(
                InputSpec(
                    kind=InputKind.USER_INPUT,
                    arg=TensorArgument(name=node.name),
                    target=None,
                )
            )

    # AOTAutograd bakes constant tensors into the GraphModule as get_attr nodes
    # rather than placeholders; register them as CONSTANT_TENSOR specs so the
    # marker pass tags them ``constant``. See _constant_get_attr_specs.
    input_specs.extend(_constant_get_attr_specs(gm))

    # AOTAutograd functionalises all mutations, so the forward graph itself
    # is pure functional. But we need BUFFER_MUTATION output specs so that
    # insert_argument_type_markers can detect which buffers are mutated.
    output_specs = []
    for target in mutated_buffer_targets:
        output_specs.append(
            OutputSpec(
                kind=OutputKind.BUFFER_MUTATION,
                arg=TensorArgument(name=""),
                target=target,
            )
        )
    output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
    if output_nodes:
        for out_arg in output_nodes[0].args[0]:
            if isinstance(out_arg, torch.fx.Node):
                output_specs.append(
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=out_arg.name),
                        target=None,
                    )
                )

    return ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)


def _build_aot_backward_graph_signature(
    gm: torch.fx.GraphModule,
    forward_signature: ExportGraphSignature,
) -> ExportGraphSignature:
    """Build an ExportGraphSignature for the AOTAutograd backward graph.

    The backward graph has a completely different input convention from the
    forward.  The min-cut partitioner feeds it the tensors saved off the
    forward pass — saved primals (forward inputs the backward still needs) and
    saved intermediates (forward activations stashed rather than recomputed) —
    followed by the tangents (the cotangents flowing in from downstream).

    The key fact we exploit: a *saved primal keeps the exact placeholder name
    it had in the forward graph* (``primals_K``), whereas a saved intermediate
    takes its forward producer's node name and a tangent is ``tangents_K``.
    So every backward placeholder whose name matches a forward input inherits
    the forward's classification — a weight saved into the backward stays a
    ``parameter`` and a non-mutated buffer stays a ``constant`` — which lets
    consteval hoist weight-only preprocessing in the backward exactly as it
    does in the forward, and is consistent with how the forward marks those
    same tensors.  Everything with no forward match (saved intermediates and
    tangents) is a USER_INPUT (``input``): it varies per iteration and must
    never be folded into consteval.

    Contrast _build_aot_graph_signature, which keys off the forward's
    params-then-buffers-then-user-inputs *ordering*.  That ordering does not
    hold in the backward (params/buffers/intermediates/tangents are
    interleaved by what the partitioner happened to save), so we match by name
    against the already-built forward signature instead.
    """
    fw_input_spec_by_name = {
        spec.arg.name: spec for spec in forward_signature.input_specs
    }

    input_specs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        fw_spec = fw_input_spec_by_name.get(node.name)
        if fw_spec is not None:
            # Saved primal: inherit the forward's kind and target so a weight
            # stays "parameter", a non-mutated buffer stays "constant", a user
            # input stays "input", and the backward arg even keeps the forward's
            # demangled FQN name.
            input_specs.append(
                InputSpec(
                    kind=fw_spec.kind,
                    arg=TensorArgument(name=node.name),
                    target=fw_spec.target,
                    persistent=fw_spec.persistent,
                )
            )
        else:
            # Saved intermediate or tangent: varies per iteration -> "input".
            input_specs.append(
                InputSpec(
                    kind=InputKind.USER_INPUT,
                    arg=TensorArgument(name=node.name),
                    target=None,
                )
            )

    # Constant tensors AOTAutograd interned during tracing can also be
    # rematerialised into the backward graph as get_attr nodes.  They remain
    # genuine constants regardless of which graph they land in, so tag them the
    # same way the forward path does. See _constant_get_attr_specs.
    input_specs.extend(_constant_get_attr_specs(gm))

    # Carry over the forward's BUFFER_MUTATION output specs so that a mutated
    # buffer saved into the backward is still classified "input" (not
    # consteval'd) by insert_argument_type_markers.  Targets that don't appear
    # among the backward inputs are simply unused and harmless.
    output_specs = [
        spec
        for spec in forward_signature.output_specs
        if spec.kind == OutputKind.BUFFER_MUTATION
    ]
    output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
    if output_nodes:
        for out_arg in output_nodes[0].args[0]:
            if isinstance(out_arg, torch.fx.Node):
                output_specs.append(
                    OutputSpec(
                        kind=OutputKind.USER_OUTPUT,
                        arg=TensorArgument(name=out_arg.name),
                        target=None,
                    )
                )

    return ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)


def _extract_params_and_consts(ep: ExportedProgram) -> Tuple[torch.Tensor, ...]:
    """Extract params and consts from an ExportedProgram as a flat tensor tuple.

    The returned tensors are in input_specs order (params, buffers, constants)
    and exclude USER_INPUT entries.  All tensors are moved to XLA if needed.
    """
    sig = ep.graph_signature
    state = ep.state_dict
    constants = ep.constants

    total_args = []
    encountered_user_input = False
    for spec in sig.input_specs:
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
            arg = arg.to(torch.device("xla"))
        total_args.append(arg)

    return tuple(total_args)


# This function runs a series of passes on a torch GraphModule.
# The passes here may be necessary (depending on the model) to
# convert a GraphModule into a form which tt-mlir can compile/execute.
def torch_pass_pipeline(
    gm: torch.fx.GraphModule,
    example_inputs: Tuple[torch.Tensor],
    options: dict[str, bool] | None,
    aot_graph_signature: ExportGraphSignature | None = None,
) -> Tuple[
    torch.fx.GraphModule,
    torch.export.ExportGraphSignature,
    dict[str, str],
    Tuple[torch.Tensor, ...],
]:
    if aot_graph_signature is not None:
        # AOTAutograd path: decompositions already applied by AOTAutograd.
        # All params/buffers are already lifted as placeholders in args.
        compiled_graph = gm
        graph_signature = aot_graph_signature
        params_and_consts = ()

        # Relocate the CPU constant buffers AOTAutograd lifted onto the graph
        # to XLA, so torch-xla's partitioner doesn't fracture the graph at the
        # resulting host op. See _relocate_lifted_constants_to_xla.
        _relocate_lifted_constants_to_xla(compiled_graph)

        # AOTAutograd's matmul lowering reshapes (..., M, K) @ (..., K, N) to
        # (B*H, M, K) @ (B*H, K, N) via a view->bmm->view sandwich.  Under TP
        # with the head dim sharded, the collapsed bmm form destroys head
        # sharding and forces SPMD to insert f32 all_gathers on the 4-D
        # Q/K/V.  Re-expand to a rank-preserving einsum here, after AOTA
        # has produced the sandwich.
        fold_view_bmm_view_to_einsum(compiled_graph)

        # Build a mapping from placeholder arg names (e.g. "primals_0") to
        # clean FQNs (e.g. "layers.0.weight") so that _demangle_name in
        # insert_argument_type_markers can resolve AOT placeholder names.
        flat_name_to_original_fqn = {
            spec.arg.name: spec.target
            for spec in aot_graph_signature.input_specs
            if spec.target is not None
        }
    else:
        # fx fusion and composite ops are pre applied in case of AOTAutograd, but not without it.
        # so we need to run them here if AOTAutograd is not used.

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

        # Non-AOTAutograd path: use torch.export for decompositions.
        decompositions = populate_decompositions()

        program = torch.export.export(
            gm,
            tuple(example_inputs),
            strict=False,
        )
        program = program.run_decompositions(decompositions)

        compiled_graph = program.graph_module
        graph_signature = program.graph_signature

        # Nodes in the fx graph can have inconsistent prev/next pointers after
        # torch.export.export.  legalize_graph rebuilds the graph in topological
        # order from usage information, fixing up the pointers.  This must happen
        # right after export, before any downstream consumer (e.g. DynamoBridge
        # Partitioner) can be confused by wrong next-node links.
        legalize_graph(compiled_graph)

        # compiled_graph above is the *flattened* exported graph_module: its
        # params, buffers and lifted constants are placeholder inputs, not
        # attributes (contrast program.module(), which retains them). They must
        # therefore be re-supplied ahead of the user inputs on every call, in
        # both the legacy and experimental flows. Extract them here, in
        # input_specs order, so XLAExecutor can prepend them to the user *args.
        params_and_consts = _extract_params_and_consts(program)

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
        compiled_graph, graph_signature, flat_name_to_original_fqn
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

    return compiled_graph, graph_signature, node_info, params_and_consts


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
        params_and_consts: Tuple[torch.Tensor, ...] = (),
    ):
        self.module = module
        self.signature = signature
        self.node_info = node_info
        # Inject metadata if xla debug is enabled and node_info is not empty
        # We need xla debug to be enabled in order for torch-xla to inject metadata
        self.inject_metadata = os.environ.get("XLA_HLO_DEBUG", "0") == "1" and node_info

        # Precompute whether there are non-USER_OUTPUT output specs.
        self.has_output_mutations = any(
            spec.kind is not OutputKind.USER_OUTPUT for spec in signature.output_specs
        )

        # Collect all devices this model will use. This device list is used to
        # signal to torch xla which devices are involved in computing the output
        # tensors, so that we may cut the graph on the output tensors correctly.
        self.devices = set()
        for _, tensor in module.state_dict().items():
            self.devices.add(tensor.device.type)
        self.devices = list(self.devices)

        # Whether to enable the legacy compile flow.
        self.legacy_compile_enabled = legacy_compile_enabled
        # Lifted params/buffers/constants to prepend to the user inputs on each
        # call. Empty in the AOT path, where AOTAutograd already lifts them into
        # the incoming *args. Used by both the legacy and experimental flows.
        self.params_and_consts = params_and_consts
        # Experimental flow only: cached torch-xla compiled graph.
        self.compiled_graph = None

    def _call_experimental_compile(self, full_args):
        if self.compiled_graph is None:
            # Use `torch_xla` function to replace the graph module with the `optimized_mod`.
            # This helps us avoid tracing the graph on the subsequent model execution. On the next
            # invocation of forward - `optimized_mod` will just look up in its cache and execute the graph
            # without any tracing.
            self.compiled_graph = bridge.extract_compiled_graph(self.module, full_args)

        return self.compiled_graph(*full_args)

    def __call__(self, *args):
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

        # self.module is the flattened forward graph with params/buffers/consts
        # lifted to placeholders, so they must be supplied ahead of the user
        # inputs. In the AOT path params_and_consts is empty because AOTAutograd
        # already prepends them to the incoming *args. This is the call signature
        # for both the legacy and experimental flows.
        full_args = self.params_and_consts + args

        if not self.legacy_compile_enabled:
            return self._call_experimental_compile(full_args)

        if self.inject_metadata:
            # Use MetadataInterpreter + MetadataDispatchMode to correctly track metadata
            # even when FX nodes decompose into multiple aten operations at dispatch time.
            # MetadataInterpreter sets a context variable for each FX node, and
            # MetadataDispatchMode reads it to attach the correct metadata to each dispatch.
            with MetadataDispatchMode():
                interp = MetadataInterpreter(self.module, self.node_info)
                output = interp.run(*full_args)
        else:
            output = self.module(*full_args)

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
    aot_graph_signature: ExportGraphSignature | None = None,
):
    module, graph_signature, node_info, params_and_consts = torch_pass_pipeline(
        gm, example_inputs, options, aot_graph_signature
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
        module, graph_signature, node_info, legacy_compile, params_and_consts
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

    # Rewrite F.interpolate(bilinear/nearest) to matmul-based implementation.
    # AOTAutograd seems to force decompose F.interpolate no matter what, and that op doesn't reach our decomposition table.
    # The standard decomposition works, but the rest of the stack is overfit on the old form and the new behavior would introduce a perf regression.
    # You guessed it, THIS IS A HACK (again) https://github.com/tenstorrent/tt-xla/issues/3912.
    gm = rewrite_interpolate_to_matmul(gm)

    # FX fusion is not robust to changes AOTAutograd applies, and that causes composite handling to break.
    # So let's just handle all that before AOTAutograd.
    enable_fusion_passes = options is None or options.get(
        "tt_enable_torch_fx_fusion_pass", True
    )
    if enable_fusion_passes:
        run_fusion_passes(gm)

    enable_composite_ops = options is None or options.get(
        "tt_enable_composite_ops", True
    )
    if enable_composite_ops:
        handle_composite_ops(gm)

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

    # bw_compiler matches the backward's saved primals back to the forward's
    # parameter/buffer/input classification by placeholder name, so the forward
    # signature must be available when the (lazily-compiled) backward runs.
    # AOTAutograd always compiles the forward before the matching backward, so
    # stashing it here when fw_compiler_boxed runs is sufficient.
    forward_signature_holder = {}

    def fw_compiler_boxed(fw_gm, fw_example_inputs):
        # Build a synthetic ExportGraphSignature from AOTAutograd's
        # TracingContext.  This uses fw_metadata for authoritative
        # mutation detection — the same analysis torch.export uses.
        signature = _build_aot_graph_signature(
            fw_gm, param_names, buffer_names, flat_name_to_original_fqn
        )
        forward_signature_holder["signature"] = signature
        return make_boxed_func(
            fw_compiler(fw_gm, fw_example_inputs, options, signature)
        )

    def bw_compiler_boxed(bw_gm, bw_example_inputs):
        # The backward graph reuses the forward compiler, but its inputs are
        # saved primals / saved intermediates / tangents rather than the
        # forward's params/buffers/user-inputs.  Drive fw_compiler with a
        # backward-specific signature that inherits the forward's marking for
        # saved primals (so a saved weight stays "parameter") and marks
        # intermediates/tangents as "input".  See
        # _build_aot_backward_graph_signature.
        forward_signature = forward_signature_holder.get("signature")
        assert forward_signature is not None, (
            "bw_compiler invoked before fw_compiler — cannot map the backward's "
            "saved primals back to the forward's parameter/buffer classification"
        )
        signature = _build_aot_backward_graph_signature(bw_gm, forward_signature)
        return make_boxed_func(
            fw_compiler(bw_gm, bw_example_inputs, options, signature)
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
            fw_compiler=fw_compiler_boxed,
            bw_compiler=bw_compiler_boxed,
            decompositions=aot_decompositions,
            partition_fn=min_cut_rematerialization_partition,
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
