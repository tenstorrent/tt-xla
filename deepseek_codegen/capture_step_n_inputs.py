"""Path B implementation: convert CPU per-step snapshots into valid
TTNN tensorbins for codegen `_main` to consume.

This sidesteps the broken codegen-hook step-N capture (see
MULTI_TOKEN_CAPTURE.md) by going directly: CPU tensors -> ttnn.from_torch
with explicit mesh sharding -> ttnn.dump_tensor. No torch_xla, no
torch.compile cache, no mark_static_address conflicts.

Workflow:
  # 1) Run the benchmark with --save-cpu-snapshots-to to capture CPU state
  pytest -svv tests/benchmark/test_llms.py::test_deepseek_v3_2_exp_tp_galaxy_2_layers \
      --decode-only --max-output-tokens 3 \
      --save-cpu-snapshots-to /home/mvasiljevic/tt-xla/deepseek_codegen/cpu_snapshots

  # 2) Run this post-processor (no torch_xla involved). It reads the
  #    snapshots and produces tensors_step{2,3,...}/ alongside the existing
  #    tensors/ directory (which is Set A = step 1).
  python deepseek_codegen/capture_step_n_inputs.py \
      --snapshots ./deepseek_codegen/cpu_snapshots \
      --base ./deepseek_codegen/graph_0/tensors \
      --output ./deepseek_codegen/graph_0

  # 3) Verify each step via pcc.py --verify-set-b:
  cd deepseek_codegen && python pcc.py --verify-set-b ./tensors_step2

The mesh-sharding spec mirrors what the benchmark's perf path uses:
  * 4x8 mesh, "batch" along the size-4 axis (axis 0), other axis size-8.
  * Batch-sharded args (KV caches, indexer k_cache, input_ids) use
    ShardTensor2dMesh(dims=(0, None)) -- shard dim 0 across mesh axis 0,
    replicate across mesh axis 1.
  * Replicated args (cache_position, the small mask/scaling tensors arg49,
    arg50) use ReplicateTensorToMesh.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

# Import codegen utilities (DeviceGetter for the mesh device).
THIS_DIR = Path(__file__).resolve().parent  # .../deepseek_codegen/
GRAPH_DIR = THIS_DIR / "graph_0"
sys.path.insert(0, str(GRAPH_DIR))

import ttnn  # noqa: E402
import utils as gu  # noqa: E402

# The codegen-emitted `_main` expects 10 activation arg indices in this order:
#   activations[0] = arg4   INT32 TILE        input_ids
#   activations[1] = arg7   INT32 ROW_MAJOR   cache_position
#   activations[2] = arg9   BF16  TILE        layer-0 indexer k_cache
#   activations[3] = arg18  BF16  TILE        layer-0 compressed_kv
#   activations[4] = arg23  BF16  TILE        layer-0 k_pe
#   activations[5] = arg30  BF16  TILE        layer-1 indexer k_cache
#   activations[6] = arg33  BF16  TILE        layer-1 compressed_kv
#   activations[7] = arg34  BF16  TILE        layer-1 k_pe
#   activations[8] = arg49  BF16  ROW_MAJOR   (static; copied from base)
#   activations[9] = arg50  BF16  ROW_MAJOR   (static; copied from base)
ARG_LAYOUT = {
    "arg4":  (ttnn.DataType.INT32,    ttnn.Layout.TILE,       True),   # batch-sharded
    "arg7":  (ttnn.DataType.INT32,    ttnn.Layout.ROW_MAJOR,  False),  # replicated
    "arg9":  (ttnn.DataType.BFLOAT16, ttnn.Layout.TILE,       True),
    "arg18": (ttnn.DataType.BFLOAT16, ttnn.Layout.TILE,       True),
    "arg23": (ttnn.DataType.BFLOAT16, ttnn.Layout.TILE,       True),
    "arg30": (ttnn.DataType.BFLOAT16, ttnn.Layout.TILE,       True),
    "arg33": (ttnn.DataType.BFLOAT16, ttnn.Layout.TILE,       True),
    "arg34": (ttnn.DataType.BFLOAT16, ttnn.Layout.TILE,       True),
}

DYNAMIC_ARGS = list(ARG_LAYOUT.keys())  # arg4, arg7, arg9, arg18, ..., arg34


def _resolve_cpu_tensor(snap: dict, arg_name: str) -> torch.Tensor:
    """Map an argN name to the corresponding CPU tensor from the snapshot."""
    pkv = snap["past_key_values"]
    ix = snap["indexer_k_caches"]
    return {
        "arg4":  snap["input_ids"],
        "arg7":  snap["cache_position"],
        "arg9":  ix[0],
        "arg18": pkv.layers[0].compressed_kv,
        "arg23": pkv.layers[0].k_pe,
        "arg30": ix[1] if len(ix) > 1 else None,
        "arg33": pkv.layers[1].compressed_kv,
        "arg34": pkv.layers[1].k_pe,
    }[arg_name]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", required=True,
                    help="Directory with snapshot_step{k}.pt files.")
    ap.add_argument("--base", required=True,
                    help="Directory holding the step-1 / static argN.tensorbin "
                         "files (e.g. deepseek_codegen/graph_0/tensors). "
                         "All argN.tensorbin files are copied as-is, then the "
                         "8 dynamic ones are overwritten per step.")
    ap.add_argument("--output", required=True,
                    help="Where to write tensors_step{k}/ subdirectories.")
    ap.add_argument("--mesh-shape", default="4,8",
                    help="Mesh shape as ROWS,COLS (default: 4,8).")
    args = ap.parse_args()

    mesh_shape = tuple(int(x) for x in args.mesh_shape.split(","))
    assert len(mesh_shape) == 2, "Path B currently expects a 2D mesh"

    snapshots_dir = Path(args.snapshots)
    base_dir = Path(args.base)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_paths = sorted(snapshots_dir.glob("snapshot_step*.pt"),
                            key=lambda p: int(p.stem.replace("snapshot_step", "")))
    if not snapshot_paths:
        print(f"No snapshot_step*.pt files in {snapshots_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(snapshot_paths)} snapshot(s): "
          f"{[p.name for p in snapshot_paths]}")

    # Open the ttnn mesh device. (DeviceGetter is a singleton.)
    device = gu.DeviceGetter.get_device(mesh_shape)
    interleaved_dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )
    sharded_mapper = ttnn.ShardTensor2dMesh(
        device, mesh_shape=mesh_shape, dims=(0, None)
    )
    replicated_mapper = ttnn.ReplicateTensorToMesh(device)

    def _dump(cpu_t: torch.Tensor, dst: Path, dtype, layout, sharded: bool):
        mapper = sharded_mapper if sharded else replicated_mapper
        # ttnn.from_torch wants a CPU torch tensor. Ensure it's detached and
        # on cpu (snapshots are saved as CPU tensors so this is a no-op).
        if not isinstance(cpu_t, torch.Tensor):
            raise TypeError(f"expected torch.Tensor, got {type(cpu_t)}")
        host_t = cpu_t.detach().cpu()
        # Some integer tensors need to be cast to int32 explicitly
        if dtype == ttnn.DataType.INT32 and host_t.dtype != torch.int32:
            host_t = host_t.to(torch.int32)
        dev_t = ttnn.from_torch(
            host_t,
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=interleaved_dram,
            mesh_mapper=mapper,
        )
        ttnn.dump_tensor(str(dst), dev_t)
        ttnn.deallocate(dev_t, False)

    for snap_path in snapshot_paths:
        step_num = int(snap_path.stem.replace("snapshot_step", ""))
        if step_num == 1:
            # Step 1's tensors already exist as the base/Set A directory; skip.
            print(f"step 1: skipping (base dir is the canonical step-1 inputs)")
            continue
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        out_dir = output_dir / f"tensors_step{step_num}"
        if out_dir.is_dir():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        # Copy all base argN.tensorbin first (static args stay as-is).
        for f in sorted(base_dir.glob("*.tensorbin")):
            shutil.copy(f, out_dir / f.name)

        # Overwrite the 8 dynamic args from the CPU snapshot.
        for arg_name in DYNAMIC_ARGS:
            cpu_t = _resolve_cpu_tensor(snap, arg_name)
            if cpu_t is None:
                print(f"  step {step_num} {arg_name}: snapshot has no value, "
                      f"keeping base copy")
                continue
            dtype, layout, sharded = ARG_LAYOUT[arg_name]
            _dump(cpu_t, out_dir / f"{arg_name}.tensorbin",
                  dtype, layout, sharded)
        print(f"step {step_num}: wrote {out_dir} (base from {base_dir} + "
              f"{len(DYNAMIC_ARGS)} dynamic args from {snap_path.name})")

    print("\nDone. Validate with:")
    print(f"  cd {THIS_DIR} && python pcc.py --verify-set-b "
          f"<output_subdir e.g. ./graph_0/tensors_step2>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
