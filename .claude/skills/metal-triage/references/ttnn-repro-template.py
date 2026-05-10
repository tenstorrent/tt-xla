# Template TTNN single-op repro. Substitute the TODO markers with values
# extracted from the failing op's MLIR signature in run.log:
#   - <MLIR_OP_LINE>: the full `Executing operation:` MLIR line from the dump,
#     pasted verbatim as a comment so the issue reader can map back to source.
#   - <MESH_SHAPE>: e.g. (1, 2) for n300, (1, 8) for an 8-chip box. Read from the
#     ttnn_layout encoding in the MLIR (the `mesh_shape` attribute) or default
#     to (1, 2) if the op only fans out across one axis.
#   - <INPUT_SHAPE>, <INPUT_DTYPE>: from the operand tensor type, e.g.
#     `tensor<1x1x256x256xbf16, ...>` -> shape (1, 1, 256, 256), dtype bfloat16.
#   - <SHARD_DIM>: from the mesh sharding annotation in the layout encoding.
#     Use ShardTensorToMesh for sharded inputs and ReplicateTensorToMesh for
#     replicated ones.
#   - <OP_CALL>: e.g. `ttnn.all_gather(a_tt, dim=3, cluster_axis=1)`. Match the
#     attribute names exactly to what the MLIR op carries.

import ttnn
import torch

# <MLIR_OP_LINE>
device = ttnn.open_mesh_device(ttnn.MeshShape<MESH_SHAPE>)

a = torch.randn<INPUT_SHAPE>

mesh_mapper_1d = ttnn.ShardTensorToMesh(device, dim=<SHARD_DIM>)
a_tt = ttnn.from_torch(
    a,
    dtype=ttnn.<INPUT_DTYPE>,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    mesh_mapper=mesh_mapper_1d,
)

print(a_tt)

res_tt = <OP_CALL>

print(res_tt)
