"""Standalone ttnn/metal repro of the decode distributed_rms_norm (fused_rms_minimal)
with the EXACT config tt-mlir emits for llama_3_1_70b_tp_qb2 2D-mesh decode:
  hidden=8192, cluster_axis=0 with num_devices=2, batch(seq_len tile)=32,
  input sharded WIDTH on an 11x6 grid -> block_w=2, epsilon=1e-5.

If PCC passes here, the metal op + config are correct in isolation => the bug is in
how tt-xla/tt-mlir SETS UP the op (stats buffer / input layout / surrounding graph),
not the op itself. If it explodes/fails, the op or this config is the culprit.
"""
import torch, ttnn

torch.manual_seed(1234)

import os
HIDDEN = int(os.environ.get('RMS_HIDDEN','7168'))
NUM_DEV = 2          # cluster_axis=0 devices (the batch axis of the (2,2) mesh)
BATCH = int(os.environ.get('RMS_BATCH','8'))           # seq_len tile dim for decode
SEQ_LEN = 32
EPS = 1e-5

def torch_rms(x, gamma, eps):
    # x: [1,1,SEQ,HIDDEN]; rms over last dim
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return (out * gamma.float()).to(torch.bfloat16)

def pcc(a, b):
    a=a.flatten().float(); b=b.flatten().float()
    a=torch.nan_to_num(a); b=torch.nan_to_num(b)
    va=a-a.mean(); vb=b-b.mean()
    d=(va.norm()*vb.norm()).item()
    return (torch.dot(va,vb).item()/d) if d else float("nan")

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))  # model's mesh; cluster_axis=0 has 2 devs
try:
    # subdevice for CCL
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(10,9))})
    wsd = ttnn.SubDevice([crs]); sdm = mesh.create_sub_device_manager([wsd], 0)
    mesh.load_sub_device_manager(sdm); mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])

    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(int(os.environ.get('RMS_GX','3')),int(os.environ.get('RMS_GY','6'))))})
    num_cores = input_shard_grid.num_cores()
    total_cores = num_cores * NUM_DEV
    shard_w = ttnn.core.roundup(HIDDEN // total_cores, ttnn.TILE_SIZE)
    shard_h = ttnn.core.roundup(BATCH, ttnn.TILE_SIZE)
    print(f"HIDDEN={HIDDEN} BATCH={BATCH} num_cores={num_cores} total={total_cores} shard_w_per_core={shard_w} block_w={shard_w//32}")

    in_mem = ttnn.create_sharded_memory_config(shape=(shard_h, shard_w), core_grid=input_shard_grid,
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    ln_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(int(os.environ.get('RMS_GX','3'))+1, int(os.environ.get('RMS_GY','6'))+1), subblock_w=1, block_h=1, block_w=shard_w//ttnn.TILE_SIZE, inplace=False)
    sem = ttnn.create_global_semaphore(mesh, input_shard_grid, 0)

    ag_mem = ttnn.create_sharded_memory_config(shape=(32,32),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0), ttnn.CoreCoord(0,0))}),
        strategy=ttnn.ShardStrategy.WIDTH, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
    tt_stats = ttnn.from_torch(torch.zeros([1,1,32,NUM_DEV], dtype=torch.bfloat16), device=mesh, layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16, memory_config=ag_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(3,None), mesh_shape=list(ttnn.MeshShape(2,2))))
    out_mem = in_mem

    x = torch.randn((1,1,SEQ_LEN,HIDDEN))
    g = torch.randn((1,1,1,HIDDEN))
    tt_in = ttnn.as_tensor(x, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT, memory_config=in_mem,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(3,None), mesh_shape=list(ttnn.MeshShape(2,2))))
    tt_g = ttnn.as_tensor(g.reshape([1,1,HIDDEN//32,32]), dtype=ttnn.bfloat16, device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(2,None), mesh_shape=list(ttnn.MeshShape(2,2))))

    tt_out = ttnn.fused_rms_minimal(tt_in, ln_cfg, 0, mesh, sem, topology=ttnn.Topology.Linear,
        memory_config=out_mem, epsilon=EPS, dtype=ttnn.bfloat16, weight=tt_g, residual_input_tensor=None,
        stats=tt_stats, use_noc1_only=False)
    ttnn.synchronize_device(mesh)

    got = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3,0), mesh_shape=(2,2)))[0].unsqueeze(0)
    ref = torch_rms(x, g, EPS)
    print(f"ref  max={ref.abs().max().item():.4f} std={ref.float().std().item():.4f}")
    print(f"dev  max={got.abs().max().item():.6g} std={got.float().std().item():.6g} naninf={bool(torch.isnan(got.float()).any() or torch.isinf(got.float()).any())}")
    print(f"PCC(fused_rms_minimal vs torch) = {pcc(got, ref):.6f}")
    mesh.reset_sub_device_stall_group()
finally:
    ttnn.close_mesh_device(mesh)
