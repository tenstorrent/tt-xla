"""EXACT config from the model's ttnn IR (2d_decode_DEFAULT_ttnn.mlir, custom-call.78):
  ttnn.distributed_rms_norm / fused_rms_minimal, cluster_axis=0, num_devices=2
  program_config: compute_with_storage_grid_size=(11,6), block_h=1, block_w=2, subblock_w=1
  input ttnn_layout59: L1 width_sharded, per-device [1,1,32,4096], shard (32,64)=1x2 tiles,
    core_ranges = [(0,0)-(10,4), (0,5)-(8,5)]  (64 cores, irregular)
  epsilon = 9.99999974e-6
Reproduces the model's exact op invocation on this Blackhole (2,2) mesh.
"""
import torch, ttnn

torch.manual_seed(1234)
HIDDEN = 8192
NUM_DEV = 2
BATCH = 32
SEQ_LEN = 32
EPS = 9.99999974e-6

def torch_rms(x, gamma, eps):
    xf = x.float(); var = xf.pow(2).mean(-1, keepdim=True)
    return ((xf * torch.rsqrt(var + eps)) * gamma.float()).to(torch.bfloat16)

def pcc(a, b):
    a=torch.nan_to_num(a.flatten().float()); b=torch.nan_to_num(b.flatten().float())
    va=a-a.mean(); vb=b-b.mean(); d=(va.norm()*vb.norm()).item()
    return (torch.dot(va,vb).item()/d) if d else float("nan")

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
try:
    # EXACT irregular input shard grid from the IR: (0,0)-(10,4) U (0,5)-(8,5) = 64 cores
    input_shard_grid = ttnn.CoreRangeSet({
        ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(10,4)),
        ttnn.CoreRange(ttnn.CoreCoord(0,5), ttnn.CoreCoord(8,5)),
    })
    # CCL fabric subdevice on non-overlapping cores (rows 6-9)
    ccl_crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(10,9))})  # must contain compute grid
    wsd = ttnn.SubDevice([ccl_crs]); sdm = mesh.create_sub_device_manager([wsd], 0)
    mesh.load_sub_device_manager(sdm); mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    SHARD_W = 64   # 2 tiles (block_w=2)
    SHARD_H = 32
    print(f"cores={input_shard_grid.num_cores()} shard=({SHARD_H},{SHARD_W}) block_w={SHARD_W//32}")

    in_mem = ttnn.create_sharded_memory_config(shape=(SHARD_H, SHARD_W), core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(11,6), subblock_w=1, block_h=1, block_w=2, inplace=False)
    sem = ttnn.create_global_semaphore(mesh, input_shard_grid, 0)

    ag_mem = ttnn.create_sharded_memory_config(shape=(32,32),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(0,0))}),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    tt_stats = ttnn.from_torch(torch.zeros([1,1,32,NUM_DEV], dtype=torch.bfloat16), device=mesh, layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16, memory_config=ag_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(3,None), mesh_shape=list(ttnn.MeshShape(2,2))))

    x = torch.randn((1,1,SEQ_LEN,HIDDEN)); g = torch.randn((1,1,1,HIDDEN))
    tt_in = ttnn.as_tensor(x, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT, memory_config=in_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(3,None), mesh_shape=list(ttnn.MeshShape(2,2))))
    tt_g = ttnn.as_tensor(g.reshape([1,1,HIDDEN//32,32]), dtype=ttnn.bfloat16, device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(2,None), mesh_shape=list(ttnn.MeshShape(2,2))))

    tt_out = ttnn.fused_rms_minimal(tt_in, ln_cfg, 0, mesh, sem, topology=ttnn.Topology.Linear,
        memory_config=in_mem, epsilon=EPS, dtype=ttnn.bfloat16, weight=tt_g, residual_input_tensor=None,
        stats=tt_stats, use_noc1_only=False)
    ttnn.synchronize_device(mesh)

    got = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3,0), mesh_shape=(2,2)))[0].unsqueeze(0)
    ref = torch_rms(x, g, EPS)
    print(f"ref max={ref.abs().max().item():.4f} std={ref.float().std().item():.4f}")
    print(f"dev max={got.abs().max().item():.6g} std={got.float().std().item():.6g} naninf={bool(torch.isnan(got.float()).any() or torch.isinf(got.float()).any())}")
    print(f"PCC(EXACT-config fused_rms_minimal vs torch) = {pcc(got, ref):.6f}")
    mesh.reset_sub_device_stall_group()
finally:
    ttnn.close_mesh_device(mesh)
