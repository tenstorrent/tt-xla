import ttnn, inspect
print("ttnn OK:", ttnn.__file__)
for name in ["scaled_dot_product_attention_decode", "paged_update_cache", "update_cache", "all_gather", "reduce_scatter", "all_to_all"]:
    obj = None
    for mod in [getattr(ttnn, "transformer", None), getattr(ttnn, "experimental", None), ttnn]:
        if mod is not None and hasattr(mod, name):
            obj = getattr(mod, name); break
    print(f"op {name}: {'FOUND' if obj else 'missing'}")
for a in ["open_mesh_device", "MeshShape", "ShardTensorToMesh", "ReplicateTensorToMesh", "ConcatMeshToTensor", "from_torch", "to_torch"]:
    print(f"attr {a}: {hasattr(ttnn, a)}")
