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
import hashlib
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


def _describe_tensor(t, idx: int) -> str:
    """Return a short string with shape/dtype of a return tensor (best-effort)."""
    try:
        shape = tuple(t.shape) if hasattr(t, "shape") else "?"
        dtype = getattr(t, "dtype", "?")
        layout = getattr(t, "layout", "?")
        on_dev = getattr(t, "device", lambda: None)()
        on_dev = type(on_dev).__name__ if on_dev is not None else "host"
        return f"  [{idx}] type={type(t).__name__} shape={shape} dtype={dtype} layout={layout} loc={on_dev}"
    except Exception as exc:
        return f"  [{idx}] type={type(t).__name__} (introspection failed: {exc})"


def _gather_via_composer(tensor, mesh):
    """Try to bring a sharded mesh tensor to a host torch.Tensor via mesh composers.

    Tries a few common shardings (batch-sharded on first axis, replicated on
    second; etc). Returns torch tensor on success, None if all attempts fail.
    """
    import ttnn as _ttnn
    try:
        host = _ttnn.from_device(tensor)
    except Exception:
        host = tensor
    try:
        host = _ttnn.to_layout(host, _ttnn.Layout.ROW_MAJOR)
    except Exception:
        pass

    # Try various 2D mesh composers — guess sharding axis
    attempts = [
        # batch-on-first-mesh-axis, replicated on second
        lambda: _ttnn.ConcatMesh2dToTensor(mesh, dims=(0, None)),
        lambda: _ttnn.ConcatMesh2dToTensor(mesh, dims=(None, 0)),
        # treating both axes as sharded along the same dim
        lambda: _ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 0)),
        # 1D fallback
        lambda: _ttnn.ConcatMeshToTensor(mesh, dim=0),
    ]
    for build in attempts:
        try:
            composer = build()
            return _ttnn.to_torch(host, mesh_composer=composer)
        except Exception:
            continue
    # Last resort: try plain to_torch (works if replicated single-shard)
    try:
        return _ttnn.to_torch(host)
    except Exception:
        return None


def _per_shard_fingerprint(tensor, mesh=None) -> str:
    """Compute a SHA-16 fingerprint of a ttnn tensor's content.

    Strategy (in order):
      1. Try gather-via-composer (works for replicated or simply-sharded
         tensors that the host can reconstruct).
      2. Fall back to per-shard hashing via ttnn.get_device_tensors. Each
         shard's bytes are seeded with shape+dtype so different-shape tensors
         can never collide even if their per-shard to_torch fails the same
         way (the bug we hit before).

    Same data with same sharding → same hash. Different data or different
    shape → different hash.
    """
    import ttnn as _ttnn
    hasher = hashlib.sha256()

    # Seed with logical shape + dtype — guarantees graph_2 (64,64,vocab) and
    # graph_3 (64,1,vocab) bf16 logits can never collide even on extraction failure.
    try:
        shape = tuple(tensor.shape)
        dtype = str(getattr(tensor, "dtype", "?"))
        hasher.update(f"shape={shape};dtype={dtype}|".encode())
    except Exception:
        hasher.update(b"shape=?;dtype=?|")

    # Strategy 1: gather via composer
    if mesh is not None:
        try:
            gathered = _gather_via_composer(tensor, mesh)
            if gathered is not None:
                hasher.update(b"gathered|")
                hasher.update(gathered.contiguous().to(_torch_dtype_for_hashing(gathered)).numpy().tobytes())
                return hasher.hexdigest()[:16]
        except Exception:
            pass  # fall through to per-shard

    # Strategy 2: per-shard hash
    hasher.update(b"per-shard|")
    try:
        shards = _ttnn.get_device_tensors(tensor)
    except Exception as exc:
        hasher.update(f"<get_device_tensors_failed:{exc}>".encode())
        return hasher.hexdigest()[:16]
    for i, shard in enumerate(shards):
        try:
            host_shard = _ttnn.from_device(shard)
            try:
                host_shard = _ttnn.to_layout(host_shard, _ttnn.Layout.ROW_MAJOR)
            except Exception:
                pass
            torch_shard = _ttnn.to_torch(host_shard)
            hasher.update(f"shard{i}_shape={tuple(torch_shard.shape)};dtype={torch_shard.dtype}|".encode())
            hasher.update(torch_shard.contiguous().to(_torch_dtype_for_hashing(torch_shard)).numpy().tobytes())
        except Exception as exc:
            hasher.update(f"<shard{i}_failed:{type(exc).__name__}:{exc}>".encode())
    return hasher.hexdigest()[:16]


def _torch_dtype_for_hashing(t):
    """Pick a torch dtype that numpy can serialize. bf16 isn't natively
    supported by numpy (used to be), so cast to float32 for stable bytes.
    """
    import torch as _torch
    if t.dtype in (_torch.bfloat16, _torch.float16):
        return _torch.float32
    return t.dtype


def run_one_graph(graph: str, codegen_dir: str, mesh_device, mesh_shape) -> float:
    """Run one emitted graph_N/main.py end-to-end. Returns elapsed ms.

    Replicates what the emitted main() does (get_device, load_inputs_for__main,
    _main(loaded, device)) but captures _main's return list and prints the
    structure of each output tensor. Also pre-seeds DeviceGetter._instance with
    the harness-owned mesh and chdir's into the graph so relative tensorbin
    paths resolve.
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
        # Replicate main()'s body so we can capture _main's return.
        loaded_inputs = main_mod.load_inputs_for__main()
        outputs = main_mod._main(loaded_inputs, mesh_device)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter_ns()

        latency_ms = (t1 - t0) / 1e6
        print(f"[harness] graph={graph} _main returned {len(outputs) if hasattr(outputs, '__len__') else 'non-list'} item(s):")
        if hasattr(outputs, "__iter__"):
            for i, t in enumerate(outputs):
                print(_describe_tensor(t, i))
        else:
            print(f"  unexpected: {type(outputs).__name__}")
        print(f"AUTORESEARCH_{graph.upper()}_TOTAL_MS={latency_ms:.3f}")

        # Stage 4: extract per-graph fingerprints from _main's return for the
        # autoresearch guard. Approach:
        #   (a) per-shard SHA-16 over outputs[3] (logits) and outputs[4] (pred
        #       tokens) — always works, doesn't need to know sharding;
        #   (b) opportunistically try to gather pred_tokens via mesh composer
        #       and print the decoded ints if it succeeds.
        # The loop uses (a) as a stable per-iter signature; (b) is for humans
        # to read the actual predicted tokens when the gather works.
        try:
            if hasattr(outputs, "__len__") and len(outputs) >= 5:
                logits_sha = _per_shard_fingerprint(outputs[3], mesh=mesh_device)
                tokens_sha = _per_shard_fingerprint(outputs[4], mesh=mesh_device)
                print(f"AUTORESEARCH_{graph.upper()}_LOGITS_SHA16={logits_sha}")
                print(f"AUTORESEARCH_{graph.upper()}_TOKENS_SHA16={tokens_sha}")
                gathered = _gather_via_composer(outputs[4], mesh_device)
                if gathered is not None:
                    flat = gathered.flatten().tolist()
                    pred_str = ",".join(str(int(x)) for x in flat)
                    print(f"AUTORESEARCH_{graph.upper()}_PRED_TOKENS={pred_str}")
                else:
                    print(f"[harness] gather composer failed for {graph} — relying on SHA16 fingerprints")
        except Exception as exc:
            print(f"[harness] fingerprint extract failed for {graph}: {type(exc).__name__}: {exc}", file=sys.stderr)

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

    is_single_device = mesh_shape == (1, 1)

    if not is_single_device:
        # Apply fabric patch idempotently (only relevant for multi-chip emit)
        patch_fabric_to_ring(codegen_dir)

        # Open shared mesh device with ring fabric (must be set before open_mesh_device)
        print(f"[harness] setting fabric to FABRIC_1D_RING")
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    else:
        print(f"[harness] single-device mode: skipping fabric setup")

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
            if not is_single_device:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception as exc:
            print(f"[harness] cleanup warning: {exc}", file=sys.stderr)

    print(f"AUTORESEARCH_TOTAL_MS={total_ms:.3f}")
    print(f"AUTORESEARCH_EXIT={exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
