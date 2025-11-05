import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import io
import numpy as np

# MLIR code provided by the user
MLIR_CODE = """
#dram = #ttnn.buffer_type<dram>
#loc1 = loc("p0.1")
#loc2 = loc("p1.2")
#loc3 = loc("p2.4")
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 101664, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 32, dram_unreserved_end = 1073131840, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_f4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<64x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<2x1024x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x4096x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 32 + d2, d3), <1x1>, memref<64x1024x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<2048x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 32768 + d2, d3), <1x1>, memref<1024x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<2048x1x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<2048x8xf32, #system_memory>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x32xf32, #system_memory>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 32768 + d2, d3), <1x1>, memref<1024x2x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 32768 + d2, d3), <1x1>, memref<32768x64xf32, #system_memory>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 32768 + d2, d3), <1x1>, memref<32768x64xf32, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 131072 + d1 * 131072 + d2, d3), <1x1>, memref<4096x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x4096x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module @SyncTensorsGraph.15 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module @SyncTensorsGraph.15 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false, ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>, ttcore.system_desc = #system_desc} {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = 1x1, chipIds = [0]> loc(#loc)
      func.func @main(%arg0: tensor<32xf32, #ttnn_layout> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>} loc("p0.1"), %arg1: tensor<64x32x8xf32, #ttnn_layout1> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>} loc("p1.2"), %arg2: tensor<1x64x32768xf32, #ttnn_layout2> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>} loc("p2.4")) -> (tensor<1x32x131072xf32, #ttnn_layout3> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device loc(#loc4)
        %1 = "ttnn.reshape"(%arg2) <{shape = [1 : i32, 64 : i32, 1 : i32, 32768 : i32]}> : (tensor<1x64x32768xf32, #ttnn_layout2>) -> tensor<1x64x1x32768xf32, #ttnn_layout4> loc(#loc4)
        "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x64x32768xf32, #ttnn_layout2>) -> () loc(#loc4)
        %2 = "ttnn.reshape"(%arg1) <{shape = [64 : i32, 32 : i32, 1 : i32, 8 : i32]}> : (tensor<64x32x8xf32, #ttnn_layout1>) -> tensor<64x32x1x8xf32, #ttnn_layout5> loc(#loc5)
        "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x32x8xf32, #ttnn_layout1>) -> () loc(#loc5)
        %3 = "ttnn.permute"(%1) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x64x1x32768xf32, #ttnn_layout4>) -> tensor<1x1x32768x64xf32, #ttnn_layout6> loc(#loc8)
        "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x64x1x32768xf32, #ttnn_layout4>) -> () loc(#loc8)
        %4 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1 : i32, 32 : i32]}> : (tensor<32xf32, #ttnn_layout>) -> tensor<1x1x1x32xf32, #ttnn_layout7> loc(#loc9)
        "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<32xf32, #ttnn_layout>) -> () loc(#loc9)
        %5 = "ttnn.from_device"(%2) : (tensor<64x32x1x8xf32, #ttnn_layout5>) -> tensor<64x32x1x8xf32, #ttnn_layout8> loc(#loc10)
        "ttnn.deallocate"(%2) <{force = false}> : (tensor<64x32x1x8xf32, #ttnn_layout5>) -> () loc(#loc10)
        %6 = "ttnn.to_layout"(%5) <{layout = #ttnn.layout<row_major>}> : (tensor<64x32x1x8xf32, #ttnn_layout8>) -> tensor<64x32x1x8xf32, #ttnn_layout9> loc(#loc10)
        "ttnn.deallocate"(%5) <{force = false}> : (tensor<64x32x1x8xf32, #ttnn_layout8>) -> () loc(#loc10)
        %7 = "ttnn.from_device"(%4) : (tensor<1x1x1x32xf32, #ttnn_layout7>) -> tensor<1x1x1x32xf32, #ttnn_layout10> loc(#loc9)
        "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1x32xf32, #ttnn_layout7>) -> () loc(#loc9)
        %8 = "ttnn.to_layout"(%7) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x1x32xf32, #ttnn_layout10>) -> tensor<1x1x1x32xf32, #ttnn_layout11> loc(#loc9)
        "ttnn.deallocate"(%7) <{force = false}> : (tensor<1x1x1x32xf32, #ttnn_layout10>) -> () loc(#loc9)
        %9 = "ttnn.from_device"(%3) : (tensor<1x1x32768x64xf32, #ttnn_layout6>) -> tensor<1x1x32768x64xf32, #ttnn_layout12> loc(#loc11)
        "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x32768x64xf32, #ttnn_layout6>) -> () loc(#loc11)
        %10 = "ttnn.to_layout"(%9) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x32768x64xf32, #ttnn_layout12>) -> tensor<1x1x32768x64xf32, #ttnn_layout13> loc(#loc11)
        "ttnn.deallocate"(%9) <{force = false}> : (tensor<1x1x32768x64xf32, #ttnn_layout12>) -> () loc(#loc11)
        %11 = "ttnn.to_device"(%10, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x32768x64xf32, #ttnn_layout13>, !ttnn.device) -> tensor<1x1x32768x64xf32, #ttnn_layout14> loc(#loc11)
        "ttnn.deallocate"(%10) <{force = false}> : (tensor<1x1x32768x64xf32, #ttnn_layout13>) -> () loc(#loc11)
        %12 = "ttnn.conv_transpose2d"(%11, %6, %8, %0) <{...}> : (...) -> tensor<1x1x131072x32xbf16, #ttnn_layout15> loc(#loc6)
        "ttnn.deallocate"(%11) <{force = false}> : (tensor<1x1x32768x64xf32, #ttnn_layout14>) -> () loc(#loc6)
        "ttnn.deallocate"(%8) <{force = false}> : (tensor<1x1x1x32xf32, #ttnn_layout11>) -> () loc(#loc6)
        "ttnn.deallocate"(%6) <{force = false}> : (tensor<64x32x1x8xf32, #ttnn_layout9>) -> () loc(#loc6)
        %13 = "ttnn.permute"(%12) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x1x131072x32xbf16, #ttnn_layout15>) -> tensor<1x32x1x131072xf32, #ttnn_layout16> loc(#loc6)
        "ttnn.deallocate"(%12) <{force = false}> : (tensor<1x1x131072x32xbf16, #ttnn_layout15>) -> () loc(#loc6)
        %14 = "ttnn.reshape"(%13) <{shape = [1 : i32, 32 : i32, 131072 : i32]}> : (tensor<1x32x1x131072xf32, #ttnn_layout16>) -> tensor<1x32x131072xf32, #ttnn_layout3> loc(#loc7)
        "ttnn.deallocate"(%13) <{force = false}> : (tensor<1x32x1x131072xf32, #ttnn_layout16>) -> () loc(#loc7)
        return %14 : tensor<1x32x131072xf32, #ttnn_layout3> loc(#loc)
      } loc(#loc)
    } loc(#loc)
  } loc(#loc)
} loc(#loc)
"""

def calculate_memory_size(shape_dtype):
    """Calculate memory size in bytes from shape_dtype string like '32xf32' or '1x64x32768xf32'."""
    # Split by 'x' to separate dimensions from dtype
    parts = shape_dtype.split('x')
    
    # The last part is the dtype
    dtype = parts[-1]
    
    # All other parts are dimensions
    dimensions = parts[:-1]
    
    # Calculate product of dimensions
    num_elements = 1
    for dim in dimensions:
        try:
            num_elements *= int(dim)
        except ValueError:
            # If dimension parsing fails, default to 1
            num_elements *= 1
    
    # Determine bytes per element based on dtype
    bytes_per_element = 4  # default to 32-bit
    
    if dtype in ['f32', 'u32', 'i32', 'si32']:
        bytes_per_element = 4  # 32-bit
    elif dtype in ['f16', 'bf16', 'u16']:
        bytes_per_element = 2  # 16-bit
    elif dtype in ['u8']:
        bytes_per_element = 1  # 8-bit
    elif dtype.startswith('bfp_f8') or dtype.startswith('bfp_bf8'):
        bytes_per_element = 1  # 8-bit
    elif dtype.startswith('bfp_f4') or dtype.startswith('bfp_bf4'):
        bytes_per_element = 1  # 4-bit, but stored as 1 byte typically
    elif dtype.startswith('bfp_f2') or dtype.startswith('bfp_bf2'):
        bytes_per_element = 1  # 2-bit, but stored as 1 byte typically
    
    return num_elements * bytes_per_element

def parse_mlir_lifetimes(code):
    """Parses the MLIR code to extract tensor lifetimes."""
    
    # Regex to find the main function and its tensor arguments
    func_def_re = re.compile(r"func\.func @main\((.*)\) ->")
    # Regex to find tensor arguments in the function signature
    arg_re = re.compile(r"(%\w+): tensor<([^,>]+)")
    
    # Regex for tensor creation (lines starting with %num =)
    creation_re = re.compile(r"^(%\d+) = .* -> tensor<([^,>]+)")
    
    # Regex for deallocation
    dealloc_re = re.compile(r'"ttnn\.deallocate"\((%\w+)\)')
    
    # Regex for the return statement
    return_re = re.compile(r"^\s*return (%\w+)")

    tensors = {}
    timestamp = 0
    max_timestamp = 0
    in_func = False

    for line in io.StringIO(code):
        line = line.strip()

        if not in_func:
            match = func_def_re.search(line)
            if match:
                in_func = True
                timestamp = 0  # Operations start *after* this, at timestamp 1
                arg_string = match.group(1)
                
                # Find all arguments and log them as starting at time 0
                for arg_match in arg_re.findall(arg_string):
                    name, shape_dtype = arg_match
                    tensors[name] = {
                        "name": name,
                        "shape_dtype": shape_dtype,
                        "start_time": 0,
                        "end_time": None
                    }
        else:
            if not line or line == "func.func":
                continue
                
            if line.startswith("}"):
                in_func = False
                continue

            # This line is an operation, increment timestamp
            timestamp += 1
            max_timestamp = timestamp

            # 1. Check for tensor creation
            create_match = creation_re.search(line)
            if create_match:
                name, shape_dtype = create_match.groups()
                if name not in tensors:
                    tensors[name] = {
                        "name": name,
                        "shape_dtype": shape_dtype,
                        "start_time": timestamp,
                        "end_time": None
                    }
                continue

            # 2. Check for deallocation
            dealloc_match = dealloc_re.search(line)
            if dealloc_match:
                name = dealloc_match.group(1)
                if name in tensors:
                    if tensors[name]["end_time"] is None:
                        tensors[name]["end_time"] = timestamp
                continue

            # 3. Check for return
            if return_re.search(line):
                break

    # Post-process: Any tensor without an end_time lives until the end
    for tensor in tensors.values():
        if tensor["end_time"] is None:
            tensor["end_time"] = max_timestamp

    return tensors, max_timestamp

def plot_tensor_lifetimes(tensors, N):
    """
    Plots the tensor lifetimes, packing them into the minimum number
    of rows ("slots") to visualize concurrent allocations.
    """
    
    if not tensors:
        print("No tensors found to plot.")
        return

    # Sort tensors by their start time
    tensor_list = sorted(tensors.values(), key=lambda t: t['start_time'])

    # row_free_time[i] = timestamp when row 'i' becomes free
    row_free_time = []
    tensor_placements = [] # Stores (tensor_dict, row_index)
    
    for tensor in tensor_list:
        start = tensor['start_time']
        end = tensor['end_time']
        
        # Find the first row that is free *at or before* this tensor's start time
        found_row = False
        for i, free_time in enumerate(row_free_time):
            if start >= free_time:
                # This row is available. Place the tensor here.
                row_free_time[i] = end  # Update when this row becomes free next
                tensor_placements.append((tensor, i))
                found_row = True
                break
        
        if not found_row:
            # No available rows, must create a new one
            new_row_index = len(row_free_time)
            row_free_time.append(end)
            tensor_placements.append((tensor, new_row_index))

    num_rows = len(row_free_time)
    print(f"Total number of operations (N) = {N}")
    print(f"Maximum concurrent tensors (peak slots) = {num_rows}")

    # --- Calculate memory sizes for all tensors ---
    memory_sizes = []
    for tensor, _ in tensor_placements:
        memory_size = calculate_memory_size(tensor['shape_dtype'])
        memory_sizes.append(memory_size)
        tensor['memory_size'] = memory_size
    
    # Normalize heights to a reasonable range (min_height to max_height)
    min_height = 0.3
    max_height = 1.5
    min_memory = min(memory_sizes)
    max_memory = max(memory_sizes)
    
    # Calculate normalized heights
    if max_memory > min_memory:
        # Scale proportionally
        heights = [min_height + (max_height - min_height) * (size - min_memory) / (max_memory - min_memory) 
                   for size in memory_sizes]
    else:
        # All tensors have the same size, use default height
        heights = [0.7] * len(tensor_placements)

    # --- Plotting ---
    
    fig, ax = plt.subplots(figsize=(30, max(10, num_rows * 1.2)))

    # Use a single dark blue color for all bars
    dark_blue = '#003366'

    for (tensor, row), height in zip(tensor_placements, heights):
        start = tensor["start_time"]
        end = tensor["end_time"]
        duration = end - start
        
        # Draw the bar with height proportional to memory size
        ax.barh(row, duration, left=start, height=height, align='center', 
                edgecolor='black', color=dark_blue, alpha=0.8)
        
        # Add text label inside the bar (include memory size in MB)
        memory_mb = tensor['memory_size'] / (1024 * 1024)
        label_text = f"{tensor['name']}: {tensor['shape_dtype']} ({memory_mb:.2f} MB)"
        # Use white text for contrast with dark blue background
        text_color = 'white'
        
        ax.text(start + duration / 2, row, label_text, 
                ha='center', va='center', color=text_color, 
                fontsize=9, fontweight='medium', clip_on=True)

    ax.set_yticks(range(num_rows))
    ax.set_yticklabels([f"Slot {i}" for i in range(num_rows)], fontsize=10)
    ax.invert_yaxis()  # Puts Slot 0 at the top

    ax.set_xlabel("Operation Timestamp (N)", fontsize=12)
    ax.set_ylabel("Memory Allocation Slot", fontsize=12)
    ax.set_title("Tensor Allocation Slots vs. Time (Packed)", fontsize=16, fontweight='bold')
    
    # Set x-axis to only show integer timestamps
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(left=-0.5, right=N + 0.5)

    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    print("Displaying packed tensor lifetime plot...")
    fig.savefig("ttnn_tensor_lifetimes.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    # 1. Parse the MLIR code
    parsed_tensors, N = parse_mlir_lifetimes(MLIR_CODE)
    
    # 2. Plot the results
    plot_tensor_lifetimes(parsed_tensors, N)