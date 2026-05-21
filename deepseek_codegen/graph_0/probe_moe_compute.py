"""Probe: inspect raw weight shapes for layer 1 MoE experts and validate moe_compute wiring assumptions.

Run with:
    cd deepseek_codegen/graph_0 && source ../venv/bin/activate && bash run -t  # (with main.py temporarily replaced)
Or:
    docker exec ... bash -lc "source venv/activate && cd deepseek_codegen/graph_0 && python -u probe_moe_compute.py"
"""
import sys
import math
import ttnn
import utils


def main():
    device = utils.DeviceGetter.get_device((4, 8))
    print(f"Mesh: {device}")

    # Load the layer-1 MoE expert weights from the same .tensorbin files main.py uses.
    # arg54 = gate_proj, arg53 = up_proj, arg52 = down_proj.
    for name, path in [
        ("gate_proj (w0)", "./tensors/arg54.tensorbin"),
        ("up_proj   (w1)", "./tensors/arg53.tensorbin"),
        ("down_proj (w2)", "./tensors/arg52.tensorbin"),
    ]:
        t = ttnn.load_tensor(path)
        print(f"  {name:18s}  shape={t.shape}  dtype={t.dtype}  layout={t.layout}  on_device={t.device() is not None}")

    print("---")
    # See what prepare_w0_w1_tensor_for_moe_compute expects vs what we have.
    from ttnn.experimental.moe_compute_utils import (
        prepare_w0_w1_tensor_for_moe_compute,
        prepare_w2_tensor_for_moe_compute,
        get_weight_core_shard_maps,
        get_weight_mem_configs,
        DS_PAD_CORES,
        DS_W0_W1_SHARD_VALS,
        DS_W2_SHARD_VALS,
    )

    w0w1_shard_map, w2_shard_map, dram_core_range_set = get_weight_core_shard_maps(
        device, DS_PAD_CORES, DS_W0_W1_SHARD_VALS, DS_W2_SHARD_VALS
    )
    print(f"w0w1_shard_map: {w0w1_shard_map}  (sum={sum(w0w1_shard_map)})")
    print(f"w2_shard_map:   {w2_shard_map}")
    print(f"dram_cores:     {dram_core_range_set}")

    w0w1_mc, w2_mc, K_shard, w2_N = get_weight_mem_configs(
        num_layers=1, experts_per_device=8, hidden_size=7168, intermediate_size=2048,
        w0_w1_shard_map=w0w1_shard_map, w2_shard_map=w2_shard_map,
        dram_core_range_set=dram_core_range_set, has_bias=False,
    )
    print(f"w0w1 mem config: {w0w1_mc}")
    print(f"w2   mem config: {w2_mc}")
    print(f"K_shard={K_shard}  w2_N={w2_N}")

    print("---")
    print(f"experts.moe_compute exposed: {hasattr(ttnn.experimental, 'moe_compute')}")
    print(f"MoEActivationFunction options: {[x for x in dir(ttnn.experimental.MoEActivationFunction)] if hasattr(ttnn.experimental, 'MoEActivationFunction') else 'N/A'}")

    if utils.DeviceGetter._instance is not None:
        ttnn.close_mesh_device(utils.DeviceGetter._instance)
        utils.DeviceGetter._instance = None


if __name__ == "__main__":
    main()
