#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Autoresearch loop verify-command harness for codegen-emitted GPT-OSS LLM modules.

Loads a codegen artifact directory (containing graph_0/.../graph_3/ subdirs from
benchmark_llm_torch_xla with CODEGEN_EXPORT_PATH set), shares one mesh device
+ ring fabric across the graphs, runs the requested graphs (default: prefill
graph_2 + decode graph_3) end-to-end via main(), times each, and prints
parseable metrics for the autoresearch loop.

Stages: (current) times one or more graphs end-to-end. Future: capture _main
outputs to compute TOP1/TOP5 against .refpt reference data.

Outputs to grep:
  AUTORESEARCH_<GRAPH>_TOTAL_MS=<float>     (per graph_N timed)
  AUTORESEARCH_TOTAL_MS=<float>             (sum across all graphs run)
  AUTORESEARCH_EXIT=<0|1>

Usage:
    python3 run_codegen_decode.py <codegen_dir> [--graphs graph_2,graph_3]

Where <codegen_dir> contains graph_0/, graph_1/, graph_2/, graph_3/ produced
by `pytest --accuracy-testing CODEGEN_EXPORT_PATH=<dir>`. Patches each graph's
utils.py to use FabricConfig.FABRIC_1D_RING (matching emitted Topology.Ring
ops) before invocation. See autoresearch_logs/CODEGEN_BUGS.md for context.
"""

import argparse
import importlib
import os
import sys
import time

import ttnn

GRAPH_DIRS = ("graph_0", "graph_1", "graph_2", "graph_3")
DECODE_GRAPH = "graph_3"  # logits decode — has logits, smallest main.py


def patch_fabric_to_ring(codegen_dir: str) -> None:
    """Idempotently rewrite emitted utils.py FABRIC_1D → FABRIC_1D_RING.

    Codegen bug: emitted utils.DeviceGetter sets FABRIC_1D but emitted main.py
    uses Topology.Ring; without this patch, reduce_scatter fails with
    "Could not find any forwarding direction".
    """
    for graph in GRAPH_DIRS:
        utils_py = os.path.join(codegen_dir, graph, "utils.py")
        if not os.path.exists(utils_py):
            continue
        with open(utils_py) as f:
            text = f.read()
        if "FABRIC_1D_RING" in text:
            continue  # already patched
        text = text.replace(
            "FabricConfig.FABRIC_1D)",
            "FabricConfig.FABRIC_1D_RING)",
        )
        with open(utils_py, "w") as f:
            f.write(text)
        print(f"[harness] patched {utils_py}: FABRIC_1D → FABRIC_1D_RING")


def import_graph_in_process(graph_dir: str, mesh_device, mesh_shape):
    """Import an emitted graph_N/ as a sibling-utils package, with shared mesh.

    The emitted main.py does `import utils` which expects to resolve to the
    sibling utils.py (DeviceGetter, load_tensor). To avoid pytest's
    tests/benchmark/utils.py shadowing it (codegen bug 1), we put graph_dir
    at the front of sys.path and `cd` into it (since main.py uses relative
    paths like "./tensors/argN.tensorbin").

    Pre-seeds utils.DeviceGetter._instance with our mesh_device so the
    emitted code's call to DeviceGetter.get_device(mesh_shape) returns our
    pre-opened device instead of trying to open a second one.
    """
    if graph_dir not in sys.path:
        sys.path.insert(0, graph_dir)

    # Force fresh imports to pick up file edits (e.g., between autoresearch loop iters).
    for name in ("utils", "main"):
        sys.modules.pop(name, None)

    utils_mod = importlib.import_module("utils")
    utils_mod.DeviceGetter._instance = mesh_device
    utils_mod.DeviceGetter._mesh_shape = mesh_shape

    main_mod = importlib.import_module("main")
    return main_mod, utils_mod


def run_one_graph(graph: str, codegen_dir: str, mesh_device, mesh_shape) -> float:
    """Run one emitted graph_N/main.py end-to-end. Returns elapsed ms.

    Imports graph_N's utils + main fresh (sys.modules cleared), pre-seeds
    DeviceGetter._instance with the shared mesh device, chdir's into the graph
    so relative tensorbin paths resolve, runs main(), then cleans up.
    """
    graph_dir = os.path.join(codegen_dir, graph)
    old_cwd = os.getcwd()
    os.chdir(graph_dir)
    print(f"[harness] chdir to {graph_dir}")

    try:
        main_mod, utils_mod = import_graph_in_process(graph_dir, mesh_device, mesh_shape)
        print(f"[harness] {graph}/main imported")

        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter_ns()
        # main() does: get_device (cached, ours), load_inputs_for__main, _main(inputs, device).
        # Trace capture happens on the first call to capture_or_execute_trace_*_main inside _main.
        ret = main_mod.main()
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter_ns()

        latency_ms = (t1 - t0) / 1e6
        print(f"[harness] graph={graph} main() returned {ret!r}")
        print(f"AUTORESEARCH_{graph.upper()}_TOTAL_MS={latency_ms:.3f}")
        return latency_ms
    finally:
        os.chdir(old_cwd)
        if graph_dir in sys.path:
            sys.path.remove(graph_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codegen_dir", help="Directory containing graph_0..graph_3 subdirs")
    parser.add_argument("--mesh-shape", default="4,8", help="Mesh shape, comma-separated")
    parser.add_argument(
        "--graphs",
        default="graph_2,graph_3",
        help="Comma-separated graphs to run, in order (default: graph_2,graph_3 = logits prefill+decode)",
    )
    args = parser.parse_args()

    codegen_dir = os.path.abspath(args.codegen_dir)
    mesh_shape = tuple(int(x) for x in args.mesh_shape.split(","))
    graphs_to_run = [g.strip() for g in args.graphs.split(",") if g.strip()]
    for g in graphs_to_run:
        if g not in GRAPH_DIRS:
            raise SystemExit(f"unknown graph '{g}'; expected one of {GRAPH_DIRS}")

    print(f"[harness] codegen_dir={codegen_dir}")
    print(f"[harness] mesh_shape={mesh_shape}")
    print(f"[harness] graphs={graphs_to_run}")

    # Apply fabric patch idempotently
    patch_fabric_to_ring(codegen_dir)

    # Open shared mesh device with ring fabric (must be set before open_mesh_device)
    print(f"[harness] setting fabric to FABRIC_1D_RING")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)

    print(f"[harness] opening mesh device shape={mesh_shape}")
    l1_small_size = 1 << 15  # matches DeviceGetter default
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(mesh_shape),
        l1_small_size=l1_small_size,
    )
    print(f"[harness] device opened: {mesh_device}")

    exit_code = 0
    total_ms = 0.0
    try:
        for graph in graphs_to_run:
            total_ms += run_one_graph(graph, codegen_dir, mesh_device, mesh_shape)
    except Exception as exc:
        print(f"[harness] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            ttnn.close_mesh_device(mesh_device)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as exc:
            print(f"[harness] cleanup warning: {exc}", file=sys.stderr)

    print(f"AUTORESEARCH_TOTAL_MS={total_ms:.3f}")
    print(f"AUTORESEARCH_EXIT={exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
