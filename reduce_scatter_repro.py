"""Test ttnn.reduce_scatter on a (2,2) mesh along cluster_axis=0 (the batch axis),
scatter_dim=0 — the exact op the 2D-mesh decode graph uses on Q/K/V outputs
(tensor<32x512> -> <16x512>, sum over the 2 batch-axis devices, scatter dim0)."""
import torch, ttnn

torch.manual_seed(0)
MB, MM = 2, 2                 # mesh (batch, model)
ROWS, COLS = 32, 512

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(MB, MM))
try:
    # Build a distinct logical value per device so the sum is checkable.
    # Device (b,m) holds tile value = 100*b + m (constant tensor [ROWS,COLS]).
    per_dev = []
    for b in range(MB):
        for m in range(MM):
            per_dev.append(torch.full((ROWS, COLS), float(100 * b + m), dtype=torch.bfloat16))
    # stack along a new leading mesh dim then shard 1 tensor per device
    big = torch.cat(per_dev, dim=0)  # [4*ROWS, COLS]; shard dim0 -> [ROWS,COLS] per device
    t = ttnn.from_torch(
        big, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    # reduce_scatter sum over cluster_axis=0 (batch axis), scatter_dim=0
    out = ttnn.reduce_scatter(
        t, dim=0, cluster_axis=0, topology=ttnn.Topology.Linear,
    )
    got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    print("out gathered shape:", list(got.shape))
    # Reference: for each model-col m, sum device values over batch rows b -> value 100*0+m + 100*1+m = 100+2m
    # each device then keeps ROWS/2=16 rows. Report distinct values per device group.
    print("unique values in gathered output:", torch.unique(got).tolist()[:10])
    print("expected per model-col m: sum_b(100b+m) =", [sum(100*b+m for b in range(MB)) for m in range(MM)])
finally:
    ttnn.close_mesh_device(mesh)
