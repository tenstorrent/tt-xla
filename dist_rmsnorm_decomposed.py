"""Standalone repro of the DECOMPOSED distributed rms norm — the path the model
actually runs for the decode (rms_norm_pre_all_gather -> all_gather(cluster_axis=0)
-> rms_norm_post_all_gather), NOT the fused rms_allgather I tested earlier.

hidden=8192 sharded on cluster_axis=0 (2 devs) -> 4096/dev, 32 rows (decode), eps=1e-5.
RMS_MEM=dram (interleaved, like prefill -> expected OK) | l1 (width-sharded, like decode -> suspected bad).
Run `tt-smi -r` before each run; wait ~20s.
"""
import os, torch, ttnn
torch.manual_seed(0)
HIDDEN=8192; NUM=2; ROWS=32; EPS=1e-5
MEM=os.environ.get("RMS_MEM","dram")

def torch_rms(x,g,eps):
    xf=x.float(); v=xf.pow(2).mean(-1,keepdim=True)
    return ((xf*torch.rsqrt(v+eps))*g.float()).to(torch.bfloat16)
def pcc(a,b):
    a=torch.nan_to_num(a.flatten().float()); b=torch.nan_to_num(b.flatten().float())
    va=a-a.mean(); vb=b-b.mean(); d=(va.norm()*vb.norm()).item()
    return (torch.dot(va,vb).item()/d) if d else float("nan")

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh=ttnn.open_mesh_device(ttnn.MeshShape(2,2))
try:
    shard2d=lambda d: ttnn.ShardTensor2dMesh(mesh, dims=(d,None), mesh_shape=list(ttnn.MeshShape(2,2)))
    if MEM=="l1":
        grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0,0),ttnn.CoreCoord(7,7))})  # 8x8=64 rect
        in_mem=ttnn.create_sharded_memory_config(shape=(32,64),core_grid=grid,strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,use_height_and_width_as_shard_shape=True)
    else:
        in_mem=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    dram=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    print(f"MEM={MEM}")

    x=torch.randn((1,1,ROWS,HIDDEN)); g=torch.randn((1,1,1,HIDDEN))
    tt_in=ttnn.as_tensor(x, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT, memory_config=in_mem, mesh_mapper=shard2d(3))
    tt_g=ttnn.as_tensor(g.reshape([1,1,HIDDEN//32,32]), dtype=ttnn.bfloat16, device=mesh, layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=dram, mesh_mapper=shard2d(2))

    stats=ttnn.rms_norm_pre_all_gather(tt_in, dtype=ttnn.bfloat16, memory_config=dram)
    print(f"pre_all_gather stats shape/dev0 max={ttnn.to_torch(stats, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh,dims=(3,0),mesh_shape=(2,2))).abs().max().item():.4g}")
    gathered=ttnn.all_gather(stats, dim=3, cluster_axis=0, memory_config=dram, topology=ttnn.Topology.Ring)
    out=ttnn.rms_norm_post_all_gather(tt_in, stats=gathered, epsilon=EPS, weight=tt_g, memory_config=in_mem)
    ttnn.synchronize_device(mesh)
    got=ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh,dims=(3,0),mesh_shape=(2,2)))[0].unsqueeze(0)
    ref=torch_rms(x,g,EPS)
    print(f"dev max={got.abs().max().item():.6g} std={got.float().std().item():.4g} naninf={bool(torch.isnan(got.float()).any() or torch.isinf(got.float()).any())}")
    print(f"PCC(decomposed norm, MEM={MEM}) = {pcc(got,ref):.6f}")
finally:
    ttnn.close_mesh_device(mesh)
