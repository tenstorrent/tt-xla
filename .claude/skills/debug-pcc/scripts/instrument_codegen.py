#!/usr/bin/env python3

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Codegen Instrumentation Tool

Compares intermediate tensor outputs between a "vanilla" baseline and an
optimized version of generated TTNN Python code to pinpoint which operation
introduces numerical divergence.

Two vanillification modes are supported (``--mode``):

  **memory** (default)
    Baseline = DRAM interleaved, no program configs.
    Use when PCC regresses due to L1/sharded memory layout decisions.

  **dtype**
    Baseline = all bfp_bf8 weights (bfp_bf4 typecast targets promoted).
    Use when PCC regresses after enabling mixed-precision weight dtypes
    (e.g. some layers on bfp_bf4 via a mixed_precision_configs JSON).

USAGE:
------

1. All-in-one — memory mode (default, recommended):

   python scripts/instrument_codegen.py auto modules/main.py

2. All-in-one — weight dtype mode:

   python scripts/instrument_codegen.py auto modules/main.py --mode dtype

   This will:
   a) Create a vanilla version (bfp_bf4 typecasts replaced with bfp_bf8)
   b) Run the vanilla version and dump golden tensors
   c) Run the optimized version and compare against golden tensors

3. With CPU golden comparison (adds per-op device-vs-CPU PCC):

   python scripts/instrument_codegen.py auto modules/main.py --mode dtype --compare-cpu

4. Standalone CPU golden comparison (no vanilla needed):

   python scripts/instrument_codegen.py compare-cpu modules/main.py

5. Isolation testing (run each op with golden inputs to isolate errors):

   python scripts/instrument_codegen.py auto-isolate modules/main.py --mode dtype

   This will:
   a) Create a vanilla version
   b) Run vanilla and dump golden tensors AND per-op inputs
   c) Run optimized with golden inputs injected per-op, returning golden
      outputs downstream so each op is tested in isolation

6. Step-by-step:

   # Create vanilla baseline (add --mode dtype for weight-dtype mode)
   python scripts/instrument_codegen.py vanillify modules/main.py [--mode dtype]

   # Run vanilla and dump golden intermediate tensors
   python scripts/instrument_codegen.py dump modules/main_vanilla.py \\
       --tensor-dir golden_tensors

   # Run optimized and compare each op against golden
   python scripts/instrument_codegen.py compare modules/main.py \\
       --tensor-dir golden_tensors

   # Or: dump with inputs for isolation testing
   python scripts/instrument_codegen.py dump-inputs modules/main_vanilla.py \\
       --tensor-dir golden_tensors

   # Run optimized with golden input injection
   python scripts/instrument_codegen.py isolate modules/main.py \\
       --tensor-dir golden_tensors

WHAT VANILLIFY DOES:
--------------------

In **memory** mode, transforms the optimizer's output into a safe baseline by:
  - Converting ALL memory_config arguments to DRAM interleaved
  - Replacing matmul program_config with None (auto kernel selection)
  - Preserving sharded memory configs for ops with hardware constraints
    (e.g. inputs to paged_update_cache must remain sharded)
  - Keeping to_memory_config calls as DRAM-to-DRAM no-ops to preserve
    tensor lifetime semantics (important when a source tensor feeds
    multiple consumers)

In **dtype** mode, transforms the optimizer's output by:
  - Replacing every ``ttnn.DataType.BFLOAT4_B`` with ``ttnn.DataType.BFLOAT8_B``
    in typecast calls, promoting all bfp_bf4 weights to bfp_bf8
  - Leaving memory configs and program configs unchanged
  - This isolates PCC regressions caused by lower weight precision from
    those caused by memory layout/sharding decisions

PREREQUISITES:
--------------

  - TT_MLIR_HOME environment variable must be set
  - The tt-xla virtual environment must be activated (source venv/activate)
  - A generated main.py with input tensors must exist. To produce them,
    include the following keys in the compile options dict passed to
    ``torch_xla.set_custom_compile_options()``:

      "backend": "codegen_py"      — emit standalone Python instead of flatbuffer
      "export_tensors": True       — serialise model inputs as .tensorbin files

    These are added alongside the other compile options (optimization_level,
    export_path, etc.) already present in the benchmark harness. See
    ``tests/benchmark/benchmarks/llm_benchmark.py`` for the full options dict.

    After running the model, the codegen backend produces:

      modules/main.py              — generated TTNN Python code
      modules/utils.py             — codegen runtime utilities
      modules/tensors/*.tensorbin  — serialised input tensors
"""

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

import torch

# Sentinel value used to replace all MemoryConfig(...) expressions.
DRAM_INTERLEAVED = (
    "ttnn.MemoryConfig("
    "ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)"
)

# TTNN ops to intercept for PCC comparison.
# Includes data movement ops (to_memory_config, to_layout) to catch
# resharding-induced corruption.
COMPUTE_OPS = {
    "abs",
    "add",
    "concat",
    "cos",
    "div",
    "embedding",
    "exp",
    "gelu",
    "layer_norm",
    "linear",
    "matmul",
    "max",
    "mean",
    "min",
    "multiply",
    "neg",
    "pad",
    "permute",
    "reciprocal",
    "relu",
    "repeat",
    "reshape",
    "rms_norm",
    "rsqrt",
    "sigmoid",
    "silu",
    "slice",
    "sin",
    "softmax",
    "sqrt",
    "subtract",
    "sum",
    "tanh",
    "to_layout",
    "to_memory_config",
    "typecast",
    "where",
}

# Namespaced TTNN ops to intercept (dot-separated path under ttnn).
NAMESPACED_COMPUTE_OPS = {
    "experimental.rotary_embedding",
    "transformer.concatenate_heads",
    "transformer.scaled_dot_product_attention",
    "transformer.scaled_dot_product_attention_decode",
    "transformer.split_query_key_value_and_split_heads",
}

# PCC thresholds for classification.
PCC_OK_THRESHOLD = 0.99
PCC_BAD_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Vanillify helpers
# ---------------------------------------------------------------------------


def find_matching_paren(text, start):
    """Return the index of the closing paren matching the open paren at *start*.

    Uses a simple depth counter. Returns -1 if no match is found.
    """
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _replace_nested_expression(source, needle, replacement):
    """Replace every occurrence of *needle*(...) in *source* with *replacement*.

    Handles arbitrarily nested parentheses inside the expression.
    """
    result_parts = []
    i = 0
    while i < len(source):
        idx = source.find(needle, i)
        if idx == -1:
            result_parts.append(source[i:])
            break
        result_parts.append(source[i:idx])
        open_paren = idx + len(needle) - 1
        close_paren = find_matching_paren(source, open_paren)
        if close_paren == -1:
            # Malformed — keep as-is and advance past the needle.
            result_parts.append(source[idx : idx + len(needle)])
            i = idx + len(needle)
        else:
            result_parts.append(replacement)
            i = close_paren + 1
    return "".join(result_parts)


def replace_all_memconfigs(source):
    """Replace every ``ttnn.MemoryConfig(...)`` with DRAM interleaved."""
    return _replace_nested_expression(source, "ttnn.MemoryConfig(", DRAM_INTERLEAVED)


def replace_matmul_program_configs(source):
    """Replace ``program_config=ttnn.MatmulMultiCore...(...)`` with ``None``.

    If the program config contains a non-None ``fused_activation``, it is
    extracted and moved to the matmul's ``activation=`` kwarg so that the
    fused op is not silently dropped.
    """
    needle = "program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig("
    result_parts = []
    i = 0
    while i < len(source):
        idx = source.find(needle, i)
        if idx == -1:
            result_parts.append(source[i:])
            break
        result_parts.append(source[i:idx])
        open_paren = idx + len(needle) - 1
        close_paren = find_matching_paren(source, open_paren)
        if close_paren == -1:
            result_parts.append(source[idx : idx + len(needle)])
            i = idx + len(needle)
            continue

        # Extract the full program_config expression to check for fused_activation
        pc_text = source[idx : close_paren + 1]

        # Look for fused_activation that is not None.  The value may contain
        # nested parentheses (e.g. ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU))
        # so we extract it using balanced-paren matching instead of a simple
        # regex character class.
        activation_str = None
        fa_needle = "fused_activation=ttnn."
        fa_idx = pc_text.find(fa_needle)
        if fa_idx != -1:
            # Find the opening paren of the activation value
            fa_start = fa_idx + len("fused_activation=")
            fa_open = pc_text.find("(", fa_start)
            if fa_open != -1:
                fa_close = find_matching_paren(pc_text, fa_open)
                if fa_close != -1:
                    fa_expr = pc_text[fa_start : fa_close + 1]
                    # Extract the op name (e.g. GELU, SILU, RELU) and convert
                    # to the lowercase string form that activation= expects.
                    op_match = re.search(
                        r"UnaryOpType\.(\w+)", fa_expr
                    )
                    if op_match:
                        activation_str = f'"{op_match.group(1).lower()}"'

        result_parts.append("program_config=None")
        i = close_paren + 1

        # If there was a non-None fused_activation, patch the activation= kwarg
        # that follows the program_config in the same matmul call.
        if activation_str:
            act_needle = "activation=None"
            act_idx = source.find(act_needle, i)
            if act_idx != -1 and act_idx - i < 200:  # must be nearby
                result_parts.append(source[i:act_idx])
                result_parts.append(f"activation={activation_str}")
                i = act_idx + len(act_needle)

    return "".join(result_parts)


def restore_paged_cache_inputs(original, vanillified):
    """Restore original memory configs for ops that feed ``paged_update_cache``.

    ``paged_update_cache`` requires its input tensor (the second positional
    argument) to be sharded — this is a hardware constraint, not an optimizer
    decision. This function identifies the variable names passed as that
    argument and restores their assignment statements from the original source
    so the sharded memory config is preserved.
    """
    vars_to_restore = set()
    for m in re.finditer(
        r"paged_update_cache\(\s*\n\s*\S+,\s*\n\s*(\w+),", vanillified
    ):
        vars_to_restore.add(m.group(1))

    if not vars_to_restore:
        return vanillified

    result = vanillified
    for var_name in vars_to_restore:
        pattern = re.compile(
            rf"^(    {re.escape(var_name)} = ttnn\.\w+\()",
            re.MULTILINE,
        )
        orig_match = pattern.search(original)
        van_match = pattern.search(result)
        if orig_match and van_match:
            orig_end = find_matching_paren(original, orig_match.end() - 1)
            van_end = find_matching_paren(result, van_match.end() - 1)
            if orig_end != -1 and van_end != -1:
                orig_stmt = original[orig_match.start() : orig_end + 1]
                result = result[: van_match.start()] + orig_stmt + result[van_end + 1 :]

    return result


def vanillify(source):
    """Transform optimized codegen into a vanilla DRAM-interleaved baseline.

    The resulting code has the same graph structure as the optimized version
    but uses only DRAM interleaved memory configs and lets TTNN auto-select
    matmul program configs.
    """
    result = replace_all_memconfigs(source)
    result = replace_matmul_program_configs(result)
    result = restore_paged_cache_inputs(source, result)
    return result


# ---------------------------------------------------------------------------
# Weight-dtype vanillify helpers
# ---------------------------------------------------------------------------


def replace_bfp4_typecasts(source):
    """Replace ``BFLOAT4_B`` typecast targets with ``BFLOAT8_B``.

    In the generated Python code, weight dtype conversions appear as calls like::

        var = ttnn.typecast(input, ttnn.DataType.BFLOAT4_B)

    Replacing ``BFLOAT4_B`` with ``BFLOAT8_B`` promotes all bfp_bf4 weights to
    bfp_bf8, creating a baseline that uses higher-precision weights everywhere.
    """
    return source.replace("ttnn.DataType.BFLOAT4_B", "ttnn.DataType.BFLOAT8_B")


def vanillify_weight_dtype(source):
    """Transform codegen with bfp_bf4 weights into a bfp_bf8-only baseline.

    Replaces all ``BFLOAT4_B`` typecast operations with ``BFLOAT8_B``, keeping
    memory configs and program configs unchanged. This isolates PCC regressions
    caused by lower-precision weight conversions from those caused by memory
    layout decisions.
    """
    return replace_bfp4_typecasts(source)


# ---------------------------------------------------------------------------
# PCC computation
# ---------------------------------------------------------------------------


def _get_mesh_device():
    """Try to obtain the active MeshDevice from the codegen DeviceGetter."""
    try:
        import utils as codegen_utils

        return codegen_utils.DeviceGetter._instance
    except Exception:
        return None


def _ttnn_to_torch_safe(tensor, mesh_device=None):
    """Convert a ttnn tensor to torch, handling multi-device (mesh) tensors.

    For tensors distributed across a MeshDevice, concatenate shards along
    the last dimension using ConcatMeshToTensor so every element is included
    in the PCC comparison.

    *mesh_device* should be the ``MeshDevice`` that produced the tensor.
    It is required by newer versions of ``ConcatMeshToTensor``.
    """
    import ttnn

    # Check if tensor is on a multi-device mesh — use composer directly
    # to avoid the slow failed-attempt + TT_FATAL log spam.
    try:
        device = mesh_device or tensor.device()
    except Exception:
        device = None

    if device is not None:
        try:
            num_devices = device.get_num_devices()
        except Exception:
            num_devices = 1
        if num_devices > 1:
            composer = ttnn.ConcatMeshToTensor(device, dim=-1)
            return ttnn.to_torch(tensor, mesh_composer=composer)

    # Single-device or host tensor — direct conversion.
    try:
        return ttnn.to_torch(tensor)
    except (RuntimeError, TypeError):
        device = device or _get_mesh_device()
        if device is None:
            raise
        composer = ttnn.ConcatMeshToTensor(device, dim=-1)
        return ttnn.to_torch(tensor, mesh_composer=composer)


def compute_pcc(tensor_a, tensor_b):
    """Compute Pearson Correlation Coefficient between two flat tensors.

    Returns ``nan`` for size mismatches or scalar tensors, ``1.0`` for
    identical constant tensors, and ``0.0`` when only one tensor is constant.
    """
    a = tensor_a.float().flatten()
    b = tensor_b.float().flatten()

    if a.numel() != b.numel():
        return float("nan")
    if a.numel() <= 1:
        return float("nan") if a.numel() == 0 else 1.0

    a_mean, b_mean = a.mean(), b.mean()
    a_std, b_std = a.std(), b.std()
    if a_std == 0 and b_std == 0:
        return 1.0
    if a_std == 0 or b_std == 0:
        return 0.0

    a_centered = a - a_mean
    b_centered = b - b_mean
    pcc = (a_centered * b_centered).sum() / (
        torch.sqrt((a_centered**2).sum()) * torch.sqrt((b_centered**2).sum())
    )
    return pcc.item()


# ---------------------------------------------------------------------------
# Tensor tracking
# ---------------------------------------------------------------------------


class TensorTracker:
    """Records or compares intermediate tensors produced by TTNN ops.

    In *dump* mode, saves each intercepted tensor to ``tensor_dir`` as a
    ``.tensorbin`` file (TTNN's native FlatBuffer serialisation format).

    In *compare* mode, loads the corresponding golden tensor from
    ``tensor_dir`` and reports the PCC for each op.

    When *compare_cpu* is ``True`` (any mode), the tracker also computes
    a CPU golden output for each op using the op's built-in
    ``golden_function`` and reports PCC between device and CPU outputs.
    """

    def __init__(self, mode, tensor_dir, compare_cpu=False,
                 activation_only=False):
        self.mode = mode
        self.tensor_dir = Path(tensor_dir).resolve()
        self.op_counter = {}
        self.results = []
        self.cpu_results = []
        self.first_bad_op = None
        self.first_bad_cpu_op = None
        self.compare_cpu = compare_cpu
        self.activation_only = activation_only
        # For isolate mode: map op_id -> golden output host tensor
        self._golden_outputs = {}

        if mode in ("dump", "dump_inputs"):
            self.tensor_dir.mkdir(parents=True, exist_ok=True)
            if mode == "dump_inputs":
                (self.tensor_dir / "inputs").mkdir(exist_ok=True)

    def _next_op_id(self, op_name):
        """Return a unique identifier like ``matmul_0``, ``matmul_1``, etc."""
        count = self.op_counter.get(op_name, 0)
        self.op_counter[op_name] = count + 1
        return f"{op_name}_{count}"

    def _tensor_path(self, op_id):
        return self.tensor_dir / f"{op_id}.tensorbin"

    def _input_path(self, op_id, arg_idx=None, kw_name=None):
        """Return path for a saved input tensor."""
        inputs_dir = self.tensor_dir / "inputs"
        if arg_idx is not None:
            return inputs_dir / f"{op_id}_arg{arg_idx}.tensorbin"
        if kw_name is not None:
            return inputs_dir / f"{op_id}_kw_{kw_name}.tensorbin"
        return inputs_dir

    def _save_inputs(self, op_id, args, kwargs):
        """Save TTNN tensor inputs for later isolation testing.

        When ``activation_only`` is set, only saves arg[0] (the activation
        input for matmul/linear ops), skipping arg[1] (weight) so that
        during isolation the optimized weight is used instead.
        """
        import ttnn

        for i, arg in enumerate(args):
            if self.activation_only and i > 0:
                break
            if isinstance(arg, ttnn.Tensor):
                try:
                    host = ttnn.from_device(arg)
                    ttnn.dump_tensor(
                        str(self._input_path(op_id, arg_idx=i)), host
                    )
                except Exception as e:
                    print(f"  [{op_id}] WARNING: could not save arg{i}: {e}")
            elif isinstance(arg, (list, tuple)):
                for j, elem in enumerate(arg):
                    if isinstance(elem, ttnn.Tensor):
                        try:
                            host = ttnn.from_device(elem)
                            path = (
                                self.tensor_dir / "inputs"
                                / f"{op_id}_arg{i}_elem{j}.tensorbin"
                            )
                            ttnn.dump_tensor(str(path), host)
                        except Exception as e:
                            print(
                                f"  [{op_id}] WARNING: could not save "
                                f"arg{i}[{j}]: {e}"
                            )

        for key, val in kwargs.items():
            if isinstance(val, ttnn.Tensor):
                try:
                    host = ttnn.from_device(val)
                    ttnn.dump_tensor(
                        str(self._input_path(op_id, kw_name=key)), host
                    )
                except Exception as e:
                    print(f"  [{op_id}] WARNING: could not save kw_{key}: {e}")

    def load_golden_inputs(self, op_id, args, kwargs, device, memory_config=None):
        """Load golden inputs and substitute them for actual tensor args.

        Non-tensor args/kwargs are kept from the optimized run (these carry
        the memory_config, program_config, etc. that we want to test).

        Returns (new_args, new_kwargs).
        """
        import ttnn

        new_args = list(args)
        for i, arg in enumerate(args):
            if self.activation_only and i > 0:
                break
            if isinstance(arg, ttnn.Tensor):
                path = self._input_path(op_id, arg_idx=i)
                if path.exists():
                    golden = ttnn.load_tensor(str(path))
                    if golden.layout != arg.layout:
                        golden = ttnn.to_layout(golden, arg.layout)
                    if golden.dtype != arg.dtype:
                        golden = ttnn.to_dtype(golden, arg.dtype)
                    mc = memory_config
                    if mc is None:
                        try:
                            mc = arg.memory_config()
                        except Exception:
                            mc = ttnn.DRAM_MEMORY_CONFIG
                    golden = ttnn.to_device(golden, device, mc)
                    new_args[i] = golden
            elif isinstance(arg, (list, tuple)):
                elems = list(arg)
                for j, elem in enumerate(elems):
                    if isinstance(elem, ttnn.Tensor):
                        path = (
                            self.tensor_dir / "inputs"
                            / f"{op_id}_arg{i}_elem{j}.tensorbin"
                        )
                        if path.exists():
                            golden = ttnn.load_tensor(str(path))
                            if golden.layout != elem.layout:
                                golden = ttnn.to_layout(golden, elem.layout)
                            if golden.dtype != elem.dtype:
                                golden = ttnn.to_dtype(golden, elem.dtype)
                            mc = memory_config
                            if mc is None:
                                try:
                                    mc = elem.memory_config()
                                except Exception:
                                    mc = ttnn.DRAM_MEMORY_CONFIG
                            golden = ttnn.to_device(golden, device, mc)
                            elems[j] = golden
                new_args[i] = type(arg)(elems)

        new_kwargs = dict(kwargs)
        for key, val in kwargs.items():
            if isinstance(val, ttnn.Tensor):
                path = self._input_path(op_id, kw_name=key)
                if path.exists():
                    golden = ttnn.load_tensor(str(path))
                    if golden.layout != val.layout:
                        golden = ttnn.to_layout(golden, val.layout)
                    if golden.dtype != val.dtype:
                        golden = ttnn.to_dtype(golden, val.dtype)
                    mc = memory_config
                    if mc is None:
                        try:
                            mc = val.memory_config()
                        except Exception:
                            mc = ttnn.DRAM_MEMORY_CONFIG
                    golden = ttnn.to_device(golden, device, mc)
                    new_kwargs[key] = golden

        return tuple(new_args), new_kwargs

    def load_golden_output(self, op_id, device, layout=None, dtype=None,
                           memory_config=None):
        """Load a golden output tensor and place it on device.

        Returns the device tensor, or ``None`` if no golden exists.
        """
        import ttnn

        path = self._tensor_path(op_id)
        if not path.exists():
            return None
        golden = ttnn.load_tensor(str(path))
        if layout is not None and golden.layout != layout:
            golden = ttnn.to_layout(golden, layout)
        if dtype is not None and golden.dtype != dtype:
            golden = ttnn.to_dtype(golden, dtype)
        mc = memory_config or ttnn.DRAM_MEMORY_CONFIG
        return ttnn.to_device(golden, device, mc)

    @staticmethod
    def _classify_pcc(pcc):
        """Return (status, marker) for a PCC value."""
        if pcc >= PCC_OK_THRESHOLD:
            return "OK", ""
        elif pcc < PCC_BAD_THRESHOLD:
            return "BAD", " <<<"
        return "WARN", ""

    def track(self, op_name, result_tensor, original_op=None, args=None,
              kwargs=None):
        """Save or compare a single op result.

        *original_op*, *args*, and *kwargs* are only needed when
        ``compare_cpu`` is enabled.  *original_op* is the unwrapped TTNN
        operation object whose ``.golden_function`` will be called.
        """
        import ttnn

        op_id = self._next_op_id(op_name)

        # Capture the device reference before from_device (needed for
        # ConcatMeshToTensor when converting multi-device tensors to torch).
        try:
            mesh_device = result_tensor.device()
        except Exception:
            mesh_device = None

        try:
            host_tensor = ttnn.from_device(result_tensor)
        except Exception as e:
            print(f"  [{op_id}] WARNING: could not read back from device: {e}")
            self.results.append((op_id, None))
            if self.compare_cpu:
                self.cpu_results.append((op_id, None))
            return

        # --- CPU golden comparison (runs in any mode) ---
        cpu_pcc_str = ""
        if self.compare_cpu and original_op is not None and args is not None:
            cpu_pcc = self._compute_cpu_golden_pcc(
                op_id, original_op, args, kwargs or {}, host_tensor, mesh_device
            )
            if cpu_pcc is not None:
                cpu_status, cpu_marker = self._classify_pcc(cpu_pcc)
                cpu_pcc_str = f"  cpu={cpu_pcc:.6f} {cpu_status}{cpu_marker}"
                self.cpu_results.append((op_id, cpu_pcc))
                if cpu_status == "BAD" and self.first_bad_cpu_op is None:
                    self.first_bad_cpu_op = op_id
            else:
                self.cpu_results.append((op_id, None))

        # --- Standard dump / compare ---
        if self.mode == "dump":
            ttnn.dump_tensor(str(self._tensor_path(op_id)), host_tensor)
            print(
                f"  [{op_id}] saved "
                f"shape={list(host_tensor.shape)} dtype={host_tensor.dtype}"
                f"{cpu_pcc_str}"
            )
            self.results.append((op_id, None))

        elif self.mode == "compare":
            golden_path = self._tensor_path(op_id)
            if not golden_path.exists():
                print(f"  [{op_id}] WARNING: no golden tensor found, skipping")
                self.results.append((op_id, None))
                return

            golden = ttnn.load_tensor(str(golden_path))
            pcc = compute_pcc(
                _ttnn_to_torch_safe(host_tensor, mesh_device),
                _ttnn_to_torch_safe(golden, mesh_device),
            )

            status, marker = self._classify_pcc(pcc)
            print(f"  [{op_id}] PCC={pcc:.6f} {status}{marker}{cpu_pcc_str}")
            self.results.append((op_id, pcc))

            if status == "BAD" and self.first_bad_op is None:
                self.first_bad_op = op_id

        elif self.mode == "compare_cpu":
            # CPU-only mode — just report CPU golden PCC.
            print(
                f"  [{op_id}] "
                f"shape={list(host_tensor.shape)} dtype={host_tensor.dtype}"
                f"{cpu_pcc_str}"
            )
            self.results.append((op_id, None))

        elif self.mode == "dump_inputs":
            # Save output AND inputs for isolation testing.
            ttnn.dump_tensor(str(self._tensor_path(op_id)), host_tensor)
            if args is not None:
                self._save_inputs(op_id, args, kwargs or {})
            print(
                f"  [{op_id}] saved+inputs "
                f"shape={list(host_tensor.shape)} dtype={host_tensor.dtype}"
                f"{cpu_pcc_str}"
            )
            self.results.append((op_id, None))

        elif self.mode == "isolate":
            # Compare op output (run with golden inputs) against golden output.
            golden_path = self._tensor_path(op_id)
            if not golden_path.exists():
                print(f"  [{op_id}] WARNING: no golden tensor found, skipping")
                self.results.append((op_id, None))
                return

            golden = ttnn.load_tensor(str(golden_path))
            pcc = compute_pcc(
                _ttnn_to_torch_safe(host_tensor, mesh_device),
                _ttnn_to_torch_safe(golden, mesh_device),
            )

            status, marker = self._classify_pcc(pcc)
            print(
                f"  [{op_id}] isolated_pcc={pcc:.6f} {status}{marker}"
                f"{cpu_pcc_str}"
            )
            self.results.append((op_id, pcc))

            if status == "BAD" and self.first_bad_op is None:
                self.first_bad_op = op_id

    def _compute_cpu_golden_pcc(self, op_id, original_op, args, kwargs,
                                host_tensor, mesh_device):
        """Compute PCC between device output and CPU golden for one op.

        Uses the op's built-in ``golden_function`` to compute the CPU
        reference.  Handles multi-device tensors by concatenating shards
        via ``ConcatMeshToTensor``.

        Returns PCC as a float, or ``None`` if the golden function is
        unavailable or fails.
        """
        import ttnn

        golden_fn = getattr(original_op, "golden_function", None)
        if golden_fn is None:
            return None

        try:
            # Convert TTNN tensor args to torch for the golden function.
            torch_args, torch_kwargs = _preprocess_args_for_golden(args, kwargs)
            cpu_output = golden_fn(*torch_args, **torch_kwargs)
        except Exception as e:
            print(f"  [{op_id}] cpu_golden: golden_function failed: {e}")
            return None

        if cpu_output is None:
            return None

        try:
            if not isinstance(cpu_output, torch.Tensor):
                return None
            device_torch = _ttnn_to_torch_safe(host_tensor, mesh_device)
            # The golden function returns a single full tensor, but the
            # device output may be TP-sharded (concatenated along last dim).
            # Trim or broadcast to match shapes for PCC comparison.
            if device_torch.shape != cpu_output.shape:
                # Try to take the first shard-sized slice from device output.
                if device_torch.numel() > cpu_output.numel():
                    device_torch = device_torch.flatten()[: cpu_output.numel()].reshape(
                        cpu_output.shape
                    )
                else:
                    return None
            return compute_pcc(device_torch, cpu_output)
        except Exception as e:
            print(f"  [{op_id}] cpu_golden: comparison failed: {e}")
            return None

    def print_summary(self):
        """Print a human-readable summary of all tracked ops."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if self.mode == "dump" and not self.compare_cpu:
            print(f"Saved {len(self.results)} tensors to {self.tensor_dir}")
            return

        # Device-vs-golden summary (compare mode)
        scored = [(op_id, pcc) for op_id, pcc in self.results if pcc is not None]
        if scored:
            bad = [(op_id, pcc) for op_id, pcc in scored if pcc < PCC_BAD_THRESHOLD]
            warn = [
                (op_id, pcc)
                for op_id, pcc in scored
                if PCC_BAD_THRESHOLD <= pcc < PCC_OK_THRESHOLD
            ]
            ok = [(op_id, pcc) for op_id, pcc in scored if pcc >= PCC_OK_THRESHOLD]

            print(f"Device vs golden — Total ops compared: {len(scored)}")
            print(f"  OK   (PCC >= {PCC_OK_THRESHOLD}): {len(ok)}")
            print(f"  WARN ({PCC_BAD_THRESHOLD}-{PCC_OK_THRESHOLD}):   {len(warn)}")
            print(f"  BAD  (PCC < {PCC_BAD_THRESHOLD}):  {len(bad)}")

            if self.first_bad_op:
                print(f"\nFirst bad op: {self.first_bad_op}")
                print("^ This is likely where the optimizer introduces the error.")

            if bad:
                print("\nAll bad ops:")
                for op_id, pcc in bad:
                    print(f"  {op_id}: PCC={pcc:.6f}")

        # CPU golden summary
        cpu_scored = [
            (op_id, pcc) for op_id, pcc in self.cpu_results if pcc is not None
        ]
        if cpu_scored:
            cpu_bad = [
                (op_id, pcc) for op_id, pcc in cpu_scored if pcc < PCC_BAD_THRESHOLD
            ]
            cpu_warn = [
                (op_id, pcc)
                for op_id, pcc in cpu_scored
                if PCC_BAD_THRESHOLD <= pcc < PCC_OK_THRESHOLD
            ]
            cpu_ok = [
                (op_id, pcc) for op_id, pcc in cpu_scored if pcc >= PCC_OK_THRESHOLD
            ]

            print(f"\nDevice vs CPU golden — Total ops compared: {len(cpu_scored)}")
            print(f"  OK   (PCC >= {PCC_OK_THRESHOLD}): {len(cpu_ok)}")
            print(f"  WARN ({PCC_BAD_THRESHOLD}-{PCC_OK_THRESHOLD}):   {len(cpu_warn)}")
            print(f"  BAD  (PCC < {PCC_BAD_THRESHOLD}):  {len(cpu_bad)}")

            if self.first_bad_cpu_op:
                print(f"\nFirst bad CPU golden op: {self.first_bad_cpu_op}")

            if cpu_bad:
                print("\nAll bad CPU golden ops:")
                for op_id, pcc in cpu_bad:
                    print(f"  {op_id}: cpu_pcc={pcc:.6f}")

        if not scored and not cpu_scored:
            print(f"Saved {len(self.results)} tensors to {self.tensor_dir}")


def _preprocess_args_for_golden(args, kwargs):
    """Convert TTNN tensor arguments to torch for golden function calls.

    Handles multi-device (mesh) tensors by concatenating shards.
    Non-tensor arguments are passed through unchanged.
    """
    import ttnn

    def _convert(obj):
        if isinstance(obj, ttnn.Tensor):
            try:
                host = ttnn.from_device(obj)
            except Exception:
                host = obj
            return _ttnn_to_torch_safe(host)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_convert(e) for e in obj)
        return obj

    # kwargs that are TTNN-specific configs and not accepted by golden functions.
    _skip_kwargs = {
        "memory_config", "program_config", "compute_kernel_config",
        "core_grid", "use_multicore", "queue_id",
    }

    torch_args = tuple(_convert(a) for a in args)
    torch_kwargs = {
        k: _convert(v) for k, v in kwargs.items() if k not in _skip_kwargs
    }
    return torch_args, torch_kwargs


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def _ensure_ttnn_env():
    """Set up ``TT_METAL_RUNTIME_ROOT`` and ``sys.path`` for TTNN imports.

    Mirrors the environment setup performed by the ``modules/run`` shell
    script so that the instrumentation script can be invoked from any
    working directory.
    """
    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("ERROR: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    tt_metal_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
    if not tt_metal_root:
        tt_metal_root = os.path.join(
            tt_mlir_home, "third_party", "tt-metal", "src", "tt-metal"
        )
        os.environ["TT_METAL_RUNTIME_ROOT"] = tt_metal_root
        print(f"TT_METAL_RUNTIME_ROOT set to {tt_metal_root}")

    for subdir in ("ttnn", "tools"):
        p = os.path.join(tt_metal_root, subdir)
        if p not in sys.path:
            sys.path.insert(0, p)


_original_ttnn_ops = {}


def _wrap_ttnn_ops(tracker, only_ops=None):
    """Monkey-patch TTNN compute ops to route results through *tracker*.

    When *only_ops* is provided (a set of op name strings like
    ``{"matmul", "linear"}``), only those ops are wrapped. Otherwise all
    ops in ``COMPUTE_OPS`` and ``NAMESPACED_COMPUTE_OPS`` are wrapped.

    Saves original functions so that :func:`_unwrap_ttnn_ops` can restore them
    between runs (avoids double-wrapping in ``auto`` mode).
    """
    import ttnn

    # Restore originals first to avoid double-wrapping on repeated calls.
    _unwrap_ttnn_ops()

    def make_wrapper(original_fn, op_name):
        if tracker.mode == "isolate":
            def wrapper(*args, **kwargs):
                # Determine op_id without consuming the counter yet.
                count = tracker.op_counter.get(op_name, 0)
                op_id = f"{op_name}_{count}"

                # Try to get device from first tensor arg.
                device = None
                for a in args:
                    if isinstance(a, ttnn.Tensor):
                        try:
                            device = a.device()
                        except Exception:
                            pass
                        if device is not None:
                            break
                if device is None:
                    device = _get_mesh_device()

                # Load golden inputs; fall back to actual inputs if unavailable.
                if device is not None:
                    try:
                        golden_args, golden_kwargs = tracker.load_golden_inputs(
                            op_id, args, kwargs, device,
                        )
                    except Exception as e:
                        print(f"  [{op_id}] WARNING: could not load golden inputs: {e}")
                        golden_args, golden_kwargs = args, kwargs
                else:
                    golden_args, golden_kwargs = args, kwargs

                # Run op with golden inputs + optimized configs.
                result = original_fn(*golden_args, **golden_kwargs)

                if isinstance(result, ttnn.Tensor):
                    tracker.track(
                        op_name, result,
                        original_op=original_fn, args=golden_args, kwargs=golden_kwargs,
                    )

                    # Return golden output downstream so next op gets clean inputs.
                    if device is not None:
                        golden_out = tracker.load_golden_output(
                            op_id, device,
                            layout=result.layout,
                            dtype=result.dtype,
                        )
                        if golden_out is not None:
                            return golden_out

                return result

            return wrapper

        def wrapper(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            if isinstance(result, ttnn.Tensor):
                tracker.track(
                    op_name, result,
                    original_op=original_fn, args=args, kwargs=kwargs,
                )
            return result

        return wrapper

    wrapped = []
    target_ops = only_ops if only_ops else COMPUTE_OPS
    target_ns_ops = set()
    if only_ops:
        # Match only_ops against namespaced ops by last component.
        # e.g. "scaled_dot_product_attention_decode" matches the full path.
        for ns_op in NAMESPACED_COMPUTE_OPS:
            last = ns_op.rsplit(".", 1)[-1]
            if last in only_ops or ns_op in only_ops:
                target_ns_ops.add(ns_op)
    else:
        target_ns_ops = NAMESPACED_COMPUTE_OPS

    for op_name in target_ops:
        if op_name not in COMPUTE_OPS:
            continue
        original = getattr(ttnn, op_name, None)
        if original is not None and callable(original):
            _original_ttnn_ops[("ttnn", op_name)] = original
            setattr(ttnn, op_name, make_wrapper(original, op_name))
            wrapped.append(op_name)

    for op_name in target_ns_ops:
        parts = op_name.split(".")
        module = ttnn
        for part in parts[:-1]:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is None:
            continue
        attr_name = parts[-1]
        original = getattr(module, attr_name, None)
        if original is not None and callable(original):
            mod_key = "ttnn." + ".".join(parts[:-1])
            _original_ttnn_ops[(mod_key, attr_name)] = (module, original)
            setattr(module, attr_name, make_wrapper(original, op_name))
            wrapped.append(op_name)

    print(f"Wrapped {len(wrapped)} TTNN ops: {', '.join(sorted(wrapped))}")


def _unwrap_ttnn_ops():
    """Restore original TTNN ops, undoing any monkey-patching."""
    import ttnn

    for key, value in _original_ttnn_ops.items():
        mod_path, attr_name = key
        if mod_path == "ttnn":
            setattr(ttnn, attr_name, value)
        else:
            module, original_fn = value
            setattr(module, attr_name, original_fn)
    _original_ttnn_ops.clear()


def run_codegen_main(main_py_path, tracker, only_ops=None):
    """Load and execute a generated ``main.py`` with instrumented TTNN ops.

    The function:
      1. Sets up TTNN environment variables and ``sys.path``
      2. Monkey-patches TTNN ops to intercept results via *tracker*
      3. Changes the working directory to the script's parent (generated code
         uses relative paths like ``./tensors/arg0.tensorbin``)
      4. Imports and calls ``main()`` from the generated module
      5. Prints the PCC comparison summary

    When *only_ops* is given (e.g. ``{"matmul"}``), only those ops are
    wrapped.  All other ops run without interception.
    """
    _ensure_ttnn_env()

    main_py_path = Path(main_py_path).resolve()
    main_dir = main_py_path.parent

    if str(main_dir) not in sys.path:
        sys.path.insert(0, str(main_dir))

    original_cwd = os.getcwd()
    os.chdir(main_dir)

    try:
        # Load the module FIRST so that perform_golden_workarounds() in
        # utils.py runs against the original (unwrapped) ttnn ops.
        spec = importlib.util.spec_from_file_location("codegen_main", str(main_py_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Wrap AFTER the module is loaded — this way golden_function
        # attributes and other op metadata are already in place.
        _wrap_ttnn_ops(tracker, only_ops=only_ops)

        # Monkey-patch fill_cache to silently skip on shape mismatch errors.
        # The full-model codegen often crashes at fill_cache when run standalone
        # because KV cache tensors have mismatched shapes.  Skipping lets us
        # continue instrumenting matmul/linear ops in later layers.
        import ttnn as _ttnn_mod
        _original_fill_cache = _ttnn_mod.fill_cache
        def _safe_fill_cache(*args, **kwargs):
            try:
                return _original_fill_cache(*args, **kwargs)
            except RuntimeError:
                pass  # silently skip — KV cache update failed
        _ttnn_mod.fill_cache = _safe_fill_cache

        print("\nRunning generated code...")
        module.main()
    finally:
        os.chdir(original_cwd)

    tracker.print_summary()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _select_vanillify(mode):
    """Return the vanillify function and a human-readable label for *mode*."""
    if mode == "dtype":
        return vanillify_weight_dtype, "bfp_bf8-only (no bfp_bf4 typecasts)"
    return vanillify, "DRAM interleaved, no program configs"


def cmd_vanillify(args):
    """Create a vanilla version of optimized ``main.py``."""
    vanillify_fn, label = _select_vanillify(args.mode)
    source = Path(args.main_py).read_text()
    vanilla = vanillify_fn(source)
    output = args.output or str(Path(args.main_py).parent / "main_vanilla.py")
    Path(output).write_text(vanilla)
    print(f"Vanilla version ({label}) written to: {output}")


def _parse_common_args(args):
    """Extract common flags from parsed args."""
    compare_cpu = getattr(args, "compare_cpu", False)
    activation_only = getattr(args, "activation_only", False)
    ops_str = getattr(args, "ops", None)
    only_ops = set(ops_str.split(",")) if ops_str else None
    return compare_cpu, activation_only, only_ops


def cmd_dump(args):
    """Run generated code and save intermediate tensors as golden references."""
    compare_cpu, activation_only, only_ops = _parse_common_args(args)
    tracker = TensorTracker("dump", args.tensor_dir, compare_cpu=compare_cpu,
                            activation_only=activation_only)
    run_codegen_main(args.main_py, tracker, only_ops=only_ops)


def cmd_compare(args):
    """Run generated code and compare intermediate tensors against golden."""
    if not Path(args.tensor_dir).exists():
        print(f"ERROR: Golden tensor directory '{args.tensor_dir}' not found.")
        print("Run the 'dump' subcommand first to create golden tensors.")
        sys.exit(1)
    compare_cpu, activation_only, only_ops = _parse_common_args(args)
    tracker = TensorTracker("compare", args.tensor_dir, compare_cpu=compare_cpu,
                            activation_only=activation_only)
    run_codegen_main(args.main_py, tracker, only_ops=only_ops)


def cmd_compare_cpu(args):
    """Run generated code and compare each op against its CPU golden output."""
    _, _, only_ops = _parse_common_args(args)
    tracker = TensorTracker("compare_cpu", args.tensor_dir, compare_cpu=True)
    run_codegen_main(args.main_py, tracker, only_ops=only_ops)


def cmd_auto(args):
    """All-in-one: vanillify, dump golden tensors, compare optimized."""
    main_py = Path(args.main_py).resolve()
    main_dir = main_py.parent
    tensor_dir = Path(args.tensor_dir)
    vanillify_fn, label = _select_vanillify(args.mode)
    compare_cpu, activation_only, only_ops = _parse_common_args(args)

    # Step 1 — Create vanilla version
    print("=" * 70)
    print(f"STEP 1: Creating vanilla version ({label})")
    print("=" * 70)
    source = main_py.read_text()
    vanilla = vanillify_fn(source)
    vanilla_path = main_dir / "main_vanilla.py"
    vanilla_path.write_text(vanilla)
    print(f"Written to: {vanilla_path}\n")

    # Step 2 — Run vanilla and dump golden tensors
    print("=" * 70)
    print("STEP 2: Running vanilla version — dumping golden tensors")
    print("=" * 70)
    tracker_dump = TensorTracker("dump", tensor_dir, compare_cpu=compare_cpu,
                                activation_only=activation_only)
    run_codegen_main(str(vanilla_path), tracker_dump, only_ops=only_ops)

    # Reset device between runs
    print("\n\nClosing device for second run...")
    _reset_device_between_runs()

    # Step 3 — Run optimized version and compare
    print("\n" + "=" * 70)
    print("STEP 3: Running optimized version — comparing against golden")
    print("=" * 70)
    tracker_compare = TensorTracker("compare", tensor_dir, compare_cpu=compare_cpu,
                                   activation_only=activation_only)
    run_codegen_main(str(main_py), tracker_compare, only_ops=only_ops)


def cmd_dump_inputs(args):
    """Run generated code and save intermediate tensors AND their inputs."""
    compare_cpu, activation_only, only_ops = _parse_common_args(args)
    tracker = TensorTracker("dump_inputs", args.tensor_dir, compare_cpu=compare_cpu,
                            activation_only=activation_only)
    run_codegen_main(args.main_py, tracker, only_ops=only_ops)


def cmd_isolate(args):
    """Run optimized code with golden inputs injected per-op."""
    if not Path(args.tensor_dir).exists():
        print(f"ERROR: Golden tensor directory '{args.tensor_dir}' not found.")
        print("Run 'dump-inputs' first to save golden tensors and inputs.")
        sys.exit(1)
    inputs_dir = Path(args.tensor_dir) / "inputs"
    if not inputs_dir.exists():
        print(f"ERROR: Inputs directory '{inputs_dir}' not found.")
        print("Run 'dump-inputs' (not 'dump') to save op inputs for isolation.")
        sys.exit(1)
    compare_cpu, activation_only, only_ops = _parse_common_args(args)
    tracker = TensorTracker("isolate", args.tensor_dir, compare_cpu=compare_cpu,
                            activation_only=activation_only)
    run_codegen_main(args.main_py, tracker, only_ops=only_ops)


def _reset_device_between_runs():
    """Close and reset the device singleton between codegen runs."""
    import ttnn

    try:
        import utils as codegen_utils

        if codegen_utils.DeviceGetter._instance is not None:
            ttnn.close_mesh_device(codegen_utils.DeviceGetter._instance)
            codegen_utils.DeviceGetter._instance = None
            codegen_utils.DeviceGetter._mesh_shape = None
    except Exception as e:
        print(f"WARNING: Could not close device cleanly: {e}")

    for mod_name in list(sys.modules.keys()):
        if mod_name == "codegen_main":
            del sys.modules[mod_name]


def cmd_auto_isolate(args):
    """All-in-one: vanillify, dump golden with inputs, isolate optimized."""
    main_py = Path(args.main_py).resolve()
    main_dir = main_py.parent
    tensor_dir = Path(args.tensor_dir)
    vanillify_fn, label = _select_vanillify(args.mode)
    compare_cpu, activation_only, only_ops = _parse_common_args(args)

    # Step 1 — Create vanilla version
    print("=" * 70)
    print(f"STEP 1: Creating vanilla version ({label})")
    print("=" * 70)
    source = main_py.read_text()
    vanilla = vanillify_fn(source)
    vanilla_path = main_dir / "main_vanilla.py"
    vanilla_path.write_text(vanilla)
    print(f"Written to: {vanilla_path}\n")

    # Step 2 — Run vanilla and dump golden tensors + inputs
    print("=" * 70)
    print("STEP 2: Running vanilla version — dumping golden tensors + inputs")
    print("=" * 70)
    tracker_dump = TensorTracker("dump_inputs", tensor_dir, compare_cpu=compare_cpu,
                                activation_only=activation_only)
    run_codegen_main(str(vanilla_path), tracker_dump, only_ops=only_ops)

    # Reset device between runs
    print("\n\nClosing device for second run...")
    _reset_device_between_runs()

    # Step 3 — Run optimized version with golden input injection
    print("\n" + "=" * 70)
    print("STEP 3: Running optimized version — isolated comparison")
    print("=" * 70)
    tracker_isolate = TensorTracker("isolate", tensor_dir, compare_cpu=compare_cpu,
                                   activation_only=activation_only)
    run_codegen_main(str(main_py), tracker_isolate, only_ops=only_ops)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Instrument generated TTNN Python code to compare intermediate "
            "tensors between a vanilla baseline and an optimized version, "
            "pinpointing where PCC diverges.  Two modes are supported:\n\n"
            "  memory  — vanilla = DRAM interleaved, no program configs (default)\n"
            "  dtype   — vanilla = all bfp_bf8 weights (bfp_bf4 typecasts promoted)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mode_kwargs = dict(
        default="memory",
        choices=["memory", "dtype"],
        help=(
            "Vanillification mode. 'memory' (default) replaces memory configs "
            "with DRAM interleaved. 'dtype' replaces bfp_bf4 typecast targets "
            "with bfp_bf8 to isolate weight-precision PCC regressions."
        ),
    )

    # -- vanillify ----------------------------------------------------------
    p_van = subparsers.add_parser(
        "vanillify",
        help="Create vanilla baseline from optimized main.py",
    )
    p_van.add_argument("main_py", help="Path to the generated main.py")
    p_van.add_argument(
        "-o",
        "--output",
        help="Output path (default: main_vanilla.py alongside input)",
    )
    p_van.add_argument("--mode", **mode_kwargs)

    compare_cpu_kwargs = dict(
        action="store_true",
        help=(
            "Also compare each op's device output against its CPU golden "
            "output (computed via the op's built-in golden_function). "
            "Works with tensor-parallel (multi-device) tensors."
        ),
    )

    ops_kwargs = dict(
        default=None,
        help=(
            "Comma-separated list of op names to intercept (e.g. 'matmul,linear'). "
            "Default: all compute ops. Use this to speed up runs by only "
            "wrapping the ops you care about."
        ),
    )

    activation_only_kwargs = dict(
        action="store_true",
        help=(
            "Only save/load the first positional argument (activation) during "
            "dump-inputs/isolate. Keeps arg[1] (weight) from the optimized run "
            "so you can measure the isolated impact of e.g. bfp4 weights."
        ),
    )

    def _add_filter_args(p):
        """Add --ops and --activation-only to a subparser."""
        p.add_argument("--ops", **ops_kwargs)
        p.add_argument("--activation-only", **activation_only_kwargs)

    # -- dump ---------------------------------------------------------------
    p_dump = subparsers.add_parser(
        "dump",
        help="Run generated code and save intermediate tensors",
    )
    p_dump.add_argument("main_py", help="Path to generated main.py")
    p_dump.add_argument(
        "--tensor-dir",
        default="golden_tensors",
        help="Output directory for tensors (default: golden_tensors)",
    )
    p_dump.add_argument("--compare-cpu", **compare_cpu_kwargs)
    _add_filter_args(p_dump)

    # -- compare ------------------------------------------------------------
    p_cmp = subparsers.add_parser(
        "compare",
        help="Run generated code and compare against golden tensors",
    )
    p_cmp.add_argument("main_py", help="Path to generated main.py")
    p_cmp.add_argument(
        "--tensor-dir",
        default="golden_tensors",
        help="Golden tensor directory (default: golden_tensors)",
    )
    p_cmp.add_argument("--compare-cpu", **compare_cpu_kwargs)
    _add_filter_args(p_cmp)

    # -- compare-cpu --------------------------------------------------------
    p_cpu = subparsers.add_parser(
        "compare-cpu",
        help="Run generated code and compare each op against CPU golden",
    )
    p_cpu.add_argument("main_py", help="Path to generated main.py")
    p_cpu.add_argument(
        "--tensor-dir",
        default="golden_tensors",
        help="Directory for any tensor output (default: golden_tensors)",
    )
    _add_filter_args(p_cpu)

    # -- auto ---------------------------------------------------------------
    p_auto = subparsers.add_parser(
        "auto",
        help="All-in-one: vanillify, dump golden, compare optimized",
    )
    p_auto.add_argument("main_py", help="Path to the optimized generated main.py")
    p_auto.add_argument(
        "--tensor-dir",
        default="golden_tensors",
        help="Directory for golden tensors (default: golden_tensors)",
    )
    p_auto.add_argument("--mode", **mode_kwargs)
    p_auto.add_argument("--compare-cpu", **compare_cpu_kwargs)
    _add_filter_args(p_auto)

    # -- dump-inputs --------------------------------------------------------
    p_dinp = subparsers.add_parser(
        "dump-inputs",
        help="Run generated code and save intermediate tensors AND inputs",
    )
    p_dinp.add_argument("main_py", help="Path to generated main.py")
    p_dinp.add_argument(
        "--tensor-dir",
        default="golden_tensors",
        help="Output directory for tensors (default: golden_tensors)",
    )
    p_dinp.add_argument("--compare-cpu", **compare_cpu_kwargs)
    _add_filter_args(p_dinp)

    # -- isolate ------------------------------------------------------------
    p_iso = subparsers.add_parser(
        "isolate",
        help="Run optimized code with golden inputs injected per-op",
    )
    p_iso.add_argument("main_py", help="Path to the optimized generated main.py")
    p_iso.add_argument(
        "--tensor-dir",
        default="golden_tensors",
        help="Golden tensor directory (default: golden_tensors)",
    )
    p_iso.add_argument("--compare-cpu", **compare_cpu_kwargs)
    _add_filter_args(p_iso)

    # -- auto-isolate -------------------------------------------------------
    p_aiso = subparsers.add_parser(
        "auto-isolate",
        help="All-in-one: vanillify, dump golden+inputs, isolate optimized",
    )
    p_aiso.add_argument("main_py", help="Path to the optimized generated main.py")
    p_aiso.add_argument(
        "--tensor-dir",
        default="golden_tensors",
        help="Directory for golden tensors (default: golden_tensors)",
    )
    p_aiso.add_argument("--mode", **mode_kwargs)
    p_aiso.add_argument("--compare-cpu", **compare_cpu_kwargs)
    _add_filter_args(p_aiso)

    args = parser.parse_args()

    commands = {
        "vanillify": cmd_vanillify,
        "dump": cmd_dump,
        "compare": cmd_compare,
        "compare-cpu": cmd_compare_cpu,
        "dump-inputs": cmd_dump_inputs,
        "isolate": cmd_isolate,
        "auto": cmd_auto,
        "auto-isolate": cmd_auto_isolate,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
