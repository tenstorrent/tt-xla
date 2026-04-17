#!/usr/bin/env python3

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Codegen Instrumentation Tool

Compares intermediate tensor outputs between a "vanilla" (DRAM interleaved)
and an optimized version of generated TTNN Python code to pinpoint which
operation introduces numerical divergence.

USAGE:
------

1. All-in-one (recommended):

   python scripts/instrument_codegen.py auto modules/main.py

   This will:
   a) Create a vanilla version (all DRAM interleaved, no program configs)
   b) Run the vanilla version and dump golden tensors
   c) Run the optimized version and compare against golden tensors

2. Step-by-step:

   # Create vanilla baseline
   python scripts/instrument_codegen.py vanillify modules/main.py

   # Run vanilla and dump golden intermediate tensors
   python scripts/instrument_codegen.py dump modules/main_vanilla.py \\
       --tensor-dir golden_tensors

   # Run optimized and compare each op against golden
   python scripts/instrument_codegen.py compare modules/main.py \\
       --tensor-dir golden_tensors

WHAT VANILLIFY DOES:
--------------------

Transforms the optimizer's output into a safe baseline by:
  - Converting ALL memory_config arguments to DRAM interleaved
  - Replacing matmul program_config with None (auto kernel selection)
  - Preserving sharded memory configs for ops with hardware constraints
    (e.g. inputs to paged_update_cache must remain sharded)
  - Keeping to_memory_config calls as DRAM-to-DRAM no-ops to preserve
    tensor lifetime semantics (important when a source tensor feeds
    multiple consumers)

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
# PCC computation
# ---------------------------------------------------------------------------


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
    """

    def __init__(self, mode, tensor_dir):
        self.mode = mode
        self.tensor_dir = Path(tensor_dir).resolve()
        self.op_counter = {}
        self.results = []
        self.first_bad_op = None

        if mode == "dump":
            self.tensor_dir.mkdir(parents=True, exist_ok=True)

    def _next_op_id(self, op_name):
        """Return a unique identifier like ``matmul_0``, ``matmul_1``, etc."""
        count = self.op_counter.get(op_name, 0)
        self.op_counter[op_name] = count + 1
        return f"{op_name}_{count}"

    def _tensor_path(self, op_id):
        return self.tensor_dir / f"{op_id}.tensorbin"

    def track(self, op_name, result_tensor):
        """Save or compare a single op result."""
        import ttnn

        op_id = self._next_op_id(op_name)

        try:
            host_tensor = ttnn.from_device(result_tensor)
        except Exception as e:
            print(f"  [{op_id}] WARNING: could not read back from device: {e}")
            self.results.append((op_id, None))
            return

        if self.mode == "dump":
            ttnn.dump_tensor(str(self._tensor_path(op_id)), host_tensor)
            print(
                f"  [{op_id}] saved "
                f"shape={list(host_tensor.shape)} dtype={host_tensor.dtype}"
            )
            self.results.append((op_id, None))

        elif self.mode == "compare":
            golden_path = self._tensor_path(op_id)
            if not golden_path.exists():
                print(f"  [{op_id}] WARNING: no golden tensor found, skipping")
                self.results.append((op_id, None))
                return

            golden = ttnn.load_tensor(str(golden_path))
            pcc = compute_pcc(ttnn.to_torch(host_tensor), ttnn.to_torch(golden))

            if pcc >= PCC_OK_THRESHOLD:
                status = "OK"
            elif pcc < PCC_BAD_THRESHOLD:
                status = "BAD"
            else:
                status = "WARN"

            marker = " <<<" if status == "BAD" else ""
            print(f"  [{op_id}] PCC={pcc:.6f} {status}{marker}")
            self.results.append((op_id, pcc))

            if status == "BAD" and self.first_bad_op is None:
                self.first_bad_op = op_id

    def print_summary(self):
        """Print a human-readable summary of all tracked ops."""
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if self.mode == "dump":
            print(f"Saved {len(self.results)} tensors to {self.tensor_dir}")
            return

        scored = [(op_id, pcc) for op_id, pcc in self.results if pcc is not None]
        bad = [(op_id, pcc) for op_id, pcc in scored if pcc < PCC_BAD_THRESHOLD]
        warn = [
            (op_id, pcc)
            for op_id, pcc in scored
            if PCC_BAD_THRESHOLD <= pcc < PCC_OK_THRESHOLD
        ]
        ok = [(op_id, pcc) for op_id, pcc in scored if pcc >= PCC_OK_THRESHOLD]

        print(f"Total ops compared: {len(scored)}")
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


def _wrap_ttnn_ops(tracker):
    """Monkey-patch TTNN compute ops to route results through *tracker*.

    Saves original functions so that :func:`_unwrap_ttnn_ops` can restore them
    between runs (avoids double-wrapping in ``auto`` mode).
    """
    import ttnn

    # Restore originals first to avoid double-wrapping on repeated calls.
    _unwrap_ttnn_ops()

    def make_wrapper(original_fn, op_name):
        def wrapper(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            if isinstance(result, ttnn.Tensor):
                tracker.track(op_name, result)
            return result

        return wrapper

    wrapped = []

    for op_name in COMPUTE_OPS:
        original = getattr(ttnn, op_name, None)
        if original is not None and callable(original):
            _original_ttnn_ops[("ttnn", op_name)] = original
            setattr(ttnn, op_name, make_wrapper(original, op_name))
            wrapped.append(op_name)

    for op_name in NAMESPACED_COMPUTE_OPS:
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


def run_codegen_main(main_py_path, tracker):
    """Load and execute a generated ``main.py`` with instrumented TTNN ops.

    The function:
      1. Sets up TTNN environment variables and ``sys.path``
      2. Monkey-patches TTNN ops to intercept results via *tracker*
      3. Changes the working directory to the script's parent (generated code
         uses relative paths like ``./tensors/arg0.tensorbin``)
      4. Imports and calls ``main()`` from the generated module
      5. Prints the PCC comparison summary
    """
    _ensure_ttnn_env()

    main_py_path = Path(main_py_path).resolve()
    main_dir = main_py_path.parent

    if str(main_dir) not in sys.path:
        sys.path.insert(0, str(main_dir))

    _wrap_ttnn_ops(tracker)

    original_cwd = os.getcwd()
    os.chdir(main_dir)

    try:
        spec = importlib.util.spec_from_file_location("codegen_main", str(main_py_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print("\nRunning generated code...")
        module.main()
    finally:
        os.chdir(original_cwd)

    tracker.print_summary()


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_vanillify(args):
    """Create a vanilla DRAM-interleaved version of optimized ``main.py``."""
    source = Path(args.main_py).read_text()
    vanilla = vanillify(source)
    output = args.output or str(Path(args.main_py).parent / "main_vanilla.py")
    Path(output).write_text(vanilla)
    print(f"Vanilla version written to: {output}")


def cmd_dump(args):
    """Run generated code and save intermediate tensors as golden references."""
    tracker = TensorTracker("dump", args.tensor_dir)
    run_codegen_main(args.main_py, tracker)


def cmd_compare(args):
    """Run generated code and compare intermediate tensors against golden."""
    if not Path(args.tensor_dir).exists():
        print(f"ERROR: Golden tensor directory '{args.tensor_dir}' not found.")
        print("Run the 'dump' subcommand first to create golden tensors.")
        sys.exit(1)
    tracker = TensorTracker("compare", args.tensor_dir)
    run_codegen_main(args.main_py, tracker)


def cmd_auto(args):
    """All-in-one: vanillify, dump golden tensors, compare optimized."""
    main_py = Path(args.main_py).resolve()
    main_dir = main_py.parent
    tensor_dir = Path(args.tensor_dir)

    # Step 1 — Create vanilla version
    print("=" * 70)
    print("STEP 1: Creating vanilla version (DRAM interleaved, no program configs)")
    print("=" * 70)
    source = main_py.read_text()
    vanilla = vanillify(source)
    vanilla_path = main_dir / "main_vanilla.py"
    vanilla_path.write_text(vanilla)
    print(f"Written to: {vanilla_path}\n")

    # Step 2 — Run vanilla and dump golden tensors
    print("=" * 70)
    print("STEP 2: Running vanilla version — dumping golden tensors")
    print("=" * 70)
    tracker_dump = TensorTracker("dump", tensor_dir)
    run_codegen_main(str(vanilla_path), tracker_dump)

    # Reset device between runs
    print("\n\nClosing device for second run...")
    import ttnn

    try:
        import utils as codegen_utils

        if codegen_utils.DeviceGetter._instance is not None:
            ttnn.close_mesh_device(codegen_utils.DeviceGetter._instance)
            codegen_utils.DeviceGetter._instance = None
            codegen_utils.DeviceGetter._mesh_shape = None
    except Exception as e:
        print(f"WARNING: Could not close device cleanly: {e}")

    # Clear the cached codegen module so the second run starts fresh.
    for mod_name in list(sys.modules.keys()):
        if mod_name == "codegen_main":
            del sys.modules[mod_name]

    # Step 3 — Run optimized version and compare
    print("\n" + "=" * 70)
    print("STEP 3: Running optimized version — comparing against golden")
    print("=" * 70)
    tracker_compare = TensorTracker("compare", tensor_dir)
    run_codegen_main(str(main_py), tracker_compare)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Instrument generated TTNN Python code to compare intermediate "
            "tensors between a vanilla (DRAM interleaved) baseline and an "
            "optimized version, pinpointing where PCC diverges."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- vanillify ----------------------------------------------------------
    p_van = subparsers.add_parser(
        "vanillify",
        help="Create vanilla baseline (DRAM interleaved, no program configs)",
    )
    p_van.add_argument("main_py", help="Path to the generated main.py")
    p_van.add_argument(
        "-o",
        "--output",
        help="Output path (default: main_vanilla.py alongside input)",
    )

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

    args = parser.parse_args()

    commands = {
        "vanillify": cmd_vanillify,
        "dump": cmd_dump,
        "compare": cmd_compare,
        "auto": cmd_auto,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
