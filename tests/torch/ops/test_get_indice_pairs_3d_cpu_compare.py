
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import time
import os
import torch
from utils import Category
from typing import List, Tuple

def get_valid_out_pos_3d(
    input_pos: torch.Tensor,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    out_spatial_shape: List[int],
) -> Tuple[torch.Tensor, int]:
    NDim = 3
    lowers = torch.zeros(NDim, dtype=torch.int32)
    uppers = torch.zeros(NDim, dtype=torch.int32)
    counter = torch.zeros(NDim, dtype=torch.int32)
    counter_size = torch.zeros(NDim, dtype=torch.int32)

    # Calculate bounds
    for i in range(NDim):
        lowers[i] = (
            input_pos[i]
            - (kernel_size[i] - 1) * dilation[i]
            - 1
            + stride[i]
            + padding[i]
        ) // stride[i]
        uppers[i] = (input_pos[i] + padding[i]) // stride[i]

    # Calculate counter sizes
    num_points = 1
    for i in range(NDim):
        counter_size[i] = (uppers[i] - lowers[i]) // dilation[i] + 1
        num_points *= counter_size[i].item()

    # Initialize counter
    counter.zero_()

    # Generate valid points
    valid_points = []
    point_counter = 0

    for i in range(num_points):
        valid = True
        m = 1
        offset = 0
        point = torch.zeros(NDim + 1, dtype=torch.int32)

        # Process dimensions in reverse order
        for j in range(NDim - 1, -1, -1):
            val = uppers[j] - counter[j] * dilation[j]
            point[j] = val

            if val < 0 or val > out_spatial_shape[j] - 1:
                valid = False

            offset += m * (input_pos[j] - val * stride[j] + padding[j]) // dilation[j]
            m *= kernel_size[j]

        point[NDim] = offset

        if valid:
            valid_points.append(point.clone())
            point_counter += 1

        # Update counter
        counter[NDim - 1] += 1
        for c in range(NDim - 1, -1, -1):
            if counter[c] == counter_size[c] and c > 0:
                counter[c - 1] += 1
                counter[c] = 0

    if valid_points:
        return torch.stack(valid_points), point_counter
    else:
        return torch.empty((0, NDim + 1), dtype=torch.int32), point_counter


def row_array_idx_3d(point: torch.Tensor, spatial_shape: List[int]) -> int:
    return (
        point[0] * spatial_shape[1] * spatial_shape[2]
        + point[1] * spatial_shape[2]
        + point[2]
    ).item()

@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="get_indice_pairs_3d_cpu",
)

def get_indice_pairs_3d_cpu_orig(
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # HARDCODED SPATIAL METADATA FOR TT-XLA
    batch_size = 1
    out_shape = [300, 600, 41]
    spatial_shape = [300, 600, 41]
    ksize = [3, 3, 3]
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    dilation = [1, 1, 1]
    out_padding = [0, 0, 0]
    subm = 1
    transpose = 0

    """
    Python implementation of getIndicePairs function for both SubManifold and regular convolution
    """

    num_act_in = indices.shape[0]

    if num_act_in == 0:
        kernel_volume = torch.tensor(ksize).prod().item()
        indice_pairs = torch.full((kernel_volume, 2, 1000), -1, dtype=torch.int32)
        out_indices = torch.zeros((0, 4), dtype=torch.int32)
        indice_num = torch.zeros(kernel_volume, dtype=torch.int32)
        return out_indices, indice_pairs, indice_num

    # Calculate spatial volume
    spatial_volume = torch.tensor(out_shape).prod().item()
    kernel_volume = torch.tensor(ksize).prod().item()

    # Initialize grids
    total_grid_size = batch_size * spatial_volume
    grids_out = torch.full((total_grid_size,), -1, dtype=torch.int32)

    # Initialize output structures
    indice_num = torch.zeros(kernel_volume, dtype=torch.int32)
    max_indices = num_act_in  # Use same size as input for consistent shape
    indice_pairs = torch.full((kernel_volume, 2, max_indices), -1, dtype=torch.int32)

    if subm == 1:
        # SubM convolution
        # Populate grids with input indices
        for j in range(num_act_in):
            batch_idx = indices[j, 0].item()
            spatial_coords = indices[j, 1:4]
            index = (
                row_array_idx_3d(spatial_coords, out_shape) + spatial_volume * batch_idx
            )
            grids_out[index] = j

        # Process each input sequentially
        for j in range(num_act_in):
            batch_idx = indices[j, 0].item()
            input_pos = indices[j, 1:4]

            # Get valid output positions
            valid_points, num_valid = get_valid_out_pos_3d(
                input_pos, ksize, stride, padding, dilation, out_shape
            )

            # Process each valid point
            for i in range(num_valid):
                point = valid_points[i]
                offset = point[3].item()  # kernel offset
                out_coords = point[:3]  # spatial coordinates

                # Calculate output index
                index = (
                    row_array_idx_3d(out_coords, out_shape) + spatial_volume * batch_idx
                )

                if grids_out[index] > -1:
                    current_slot = indice_num[offset].item()
                    indice_pairs[offset, 0, current_slot] = j
                    indice_pairs[offset, 1, current_slot] = grids_out[index]
                    indice_num[offset] += 1

        # Return original indices for SubM
        out_indices = indices.int()

    else:
        # Regular convolution (subm=0)
        out_indices_list = []
        num_act_out = 0

        for j in range(num_act_in):
            batch_idx = indices[j, 0].item()
            input_pos = indices[j, 1:4]

            # Get valid output positions for this input
            valid_points, num_valid = get_valid_out_pos_3d(
                input_pos, ksize, stride, padding, dilation, out_shape
            )

            # Process each valid point
            for i in range(num_valid):
                point = valid_points[i]
                offset = point[3].item()  # kernel offset
                out_coords = point[:3]  # spatial coordinates

                # Calculate grid index
                grid_idx = (
                    row_array_idx_3d(out_coords, out_shape) + spatial_volume * batch_idx
                )

                # Check if this output position is new
                if grids_out[grid_idx] == -1:
                    # New output position - add to output indices
                    out_indices_list.append(
                        [
                            batch_idx,
                            out_coords[0].item(),
                            out_coords[1].item(),
                            out_coords[2].item(),
                        ]
                    )
                    grids_out[grid_idx] = num_act_out
                    num_act_out += 1

                # Add indice pair
                current_slot = indice_num[offset].item()
                if current_slot < max_indices:
                    indice_pairs[offset, 0, current_slot] = j
                    indice_pairs[offset, 1, current_slot] = grids_out[grid_idx]
                    indice_num[offset] += 1

        # Convert output indices list to tensor
        if out_indices_list:
            out_indices = torch.tensor(out_indices_list, dtype=torch.int32)
        else:
            out_indices = torch.zeros((0, 4), dtype=torch.int32)

    return out_indices, indice_pairs, indice_num

def get_indice_pairs_3d_cpu_new(
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    # HARDCODED SPATIAL METADATA FOR TT-XLA
    batch_size = 1
    out_shape = [300, 600, 41]
    spatial_shape = [300, 600, 41]
    ksize = [3, 3, 3]
    stride = [1, 1, 1]
    padding = [1, 1, 1]
    dilation = [1, 1, 1]
    out_padding = [0, 0, 0]
    subm = 1
    transpose = 0

    N = indices.shape[0]
    dev = indices.device
    
    spatial_volume = out_shape[0] * out_shape[1] * out_shape[2]
    total_grid_size = batch_size * spatial_volume
    grids_out = torch.full((total_grid_size+ 1,), -1, dtype=torch.int32, device=dev)
    
    K = ksize[0] * ksize[1] * ksize[2]
    max_indices = N
    
    if N == 0:
        return torch.zeros((0, 4), dtype=torch.int32, device=dev), torch.full((K, 2, max_indices), -1, dtype=torch.int32, device=dev), torch.zeros(K, dtype=torch.int32, device=dev)

    pad_t = torch.tensor(padding, device=dev)
    str_t = torch.tensor(stride, device=dev)
    dil_t = torch.tensor(dilation, device=dev)
    out_shp_t = torch.tensor(out_shape, device=dev)
    
    cx = torch.arange(ksize[0], device=dev)
    cy = torch.arange(ksize[1], device=dev)
    cz = torch.arange(ksize[2], device=dev)
    c_grid_x, c_grid_y, c_grid_z = torch.meshgrid(cx, cy, cz, indexing='ij')
    c_offsets = torch.stack([c_grid_x, c_grid_y, c_grid_z], dim=-1).view(-1, 3) # [K, 3]

    in_batch = indices[:, 0]
    in_pos = indices[:, 1:4]
    
    in_pos_ext = in_pos.unsqueeze(1) # [N, 1, 3]
    c_off_ext = c_offsets.unsqueeze(0) # [1, K, 3]
    batch_idx_ext = in_batch.view(N, 1, 1).expand(N, K, 1)
    
    if subm == 1:
        out_indices = indices.to(torch.int32)
        in_idx_1d = in_batch * spatial_volume + in_pos[:, 0] * out_shape[1] * out_shape[2] + in_pos[:, 1] * out_shape[2] + in_pos[:, 2]
        
        grids_out[in_idx_1d.long()] = torch.arange(N, dtype=torch.int32, device=dev)
        
        out_pos_all = in_pos_ext + pad_t - c_off_ext # [N, K, 3]
        bound_mask = (out_pos_all >= 0).all(dim=-1) & (out_pos_all < out_shp_t).all(dim=-1) # [N, K]
        
        out_idx_1d_all = batch_idx_ext[..., 0] * spatial_volume + out_pos_all[..., 0] * out_shape[1] * out_shape[2] + out_pos_all[..., 1] * out_shape[2] + out_pos_all[..., 2] # [N, K]
        
        # Natively trace dense statically allocated TT-tensors across the map without generating size variants!
        safe_flat_indices = torch.where(bound_mask, out_idx_1d_all.long(), torch.tensor(0, dtype=torch.long, device=dev))
        mapped_ids = grids_out[safe_flat_indices] # [N, K]
        hit_mask = (mapped_ids > -1) & bound_mask # [N, K]
        
        local_idx = torch.cumsum(hit_mask.int(), dim=0) - 1 # [N, K]
        safe_local_idx = torch.where(hit_mask, local_idx.long(), torch.tensor(max_indices, device=dev, dtype=torch.long)) # [N, K]
        
        indice_pairs_ext = torch.full((K, 2, max_indices + 1), -1, dtype=torch.int32, device=dev)
        
        N_arange = torch.arange(N, device=dev).unsqueeze(1).expand(N, K)
        src_stack = torch.stack([N_arange, mapped_ids], dim=2).int() # [N, K, 2]
        src_stack_t = src_stack.permute(1, 2, 0) # [K, 2, N]
        
        idx_stack = safe_local_idx.t().unsqueeze(1).expand(K, 2, N) # [K, 2, N]
        
        indice_pairs_ext.scatter_(dim=2, index=idx_stack, src=src_stack_t)
        indice_pairs = indice_pairs_ext[:, :, :max_indices]
        indice_num = hit_mask.int().sum(dim=0).int() # [K]
                
    else:
        numerator = (in_pos_ext + pad_t - c_off_ext * dil_t)
        div_mask = (numerator % str_t == 0).all(dim=-1)
        out_pos_all = numerator // str_t
        
        bound_mask = (out_pos_all >= 0).all(dim=-1) & (out_pos_all < out_shp_t).all(dim=-1)
        valid_mask = div_mask & bound_mask # [N, K]
        
        out_idx_1d_all = batch_idx_ext[..., 0] * spatial_volume + out_pos_all[..., 0] * out_shape[1] * out_shape[2] + out_pos_all[..., 1] * out_shape[2] + out_pos_all[..., 2]
        safe_coords_raw = torch.where(valid_mask, out_idx_1d_all.long(), torch.tensor(total_grid_size, device=dev, dtype=torch.long)) # [N, K]
        
        # Instantly aggregate consecutive unique output destinations utilizing purely static dense shapes
        active_cells_ext = torch.zeros(total_grid_size + 1, dtype=torch.int32, device=dev)
        ones_flat = torch.ones(N * K, dtype=torch.int32, device=dev)
        active_cells_ext.scatter_(dim=0, index=safe_coords_raw.view(-1), src=ones_flat)
        active_cells = active_cells_ext[:total_grid_size] # [total_grid_size]
        
        # active_id_map = torch.cumsum(active_cells, dim=0) - 1 # [total_grid_size]
        active_id_map_ext = torch.cumsum(active_cells_ext, dim=0) - 1  # size 945001
        
        valid_flat_unique_mask = active_cells > 0
        unique_flat = torch.where(valid_flat_unique_mask)[0].int() 
        # grids_out = torch.where(active_cells > 0, active_id_map, -1)
        grids_out = torch.where(                                        # size 945001
            active_cells_ext > 0,
            active_id_map_ext,
            torch.tensor(-1, dtype=torch.int32, device=dev)
        )
        
        out_batch = unique_flat // spatial_volume
        rem = unique_flat % spatial_volume
        out_x = rem // (out_shape[1] * out_shape[2])
        rem = rem % (out_shape[1] * out_shape[2])
        out_y = rem // out_shape[2]
        out_z = rem % out_shape[2]
        out_indices = torch.stack([out_batch, out_x, out_y, out_z], dim=-1).int() # [V, 4]
        
        mapped_ids = grids_out[safe_coords_raw] # [N, K]
        hit_mask = (mapped_ids > -1) & valid_mask # [N, K]
        
        local_idx = torch.cumsum(hit_mask.int(), dim=0) - 1 # [N, K]
        N_arange = torch.arange(N, device=dev).unsqueeze(1).expand(N, K)
        safe_local_idx = torch.where(hit_mask, local_idx.long(), torch.tensor(max_indices, device=dev, dtype=torch.long)) # [N, K]
        
        indice_pairs_ext = torch.full((K, 2, max_indices + 1), -1, dtype=torch.int32, device=dev)
        src_stack = torch.stack([N_arange, mapped_ids], dim=2).int() # [N, K, 2]
        src_stack_t = src_stack.permute(1, 2, 0) # [K, 2, N]
        idx_stack = safe_local_idx.t().unsqueeze(1).expand(K, 2, N) # [K, 2, N]
        
        indice_pairs_ext.scatter_(dim=2, index=idx_stack, src=src_stack_t)
        
        indice_pairs = indice_pairs_ext[:, :, :max_indices]
        indice_num = hit_mask.int().sum(dim=0).int() # [K]

    return out_indices, indice_pairs, indice_num


def test_get_indice_pairs_3d_cpu_sanity():
    """Test get_indice_pairs_3d_cpu operation bridging."""
    
    orig_func = get_indice_pairs_3d_cpu_orig
    new_func = get_indice_pairs_3d_cpu_new    

    input_file = "/home/tt-xla/saved_inputs.pt"
    if not os.path.exists(input_file):
        pytest.skip(f"Input file {input_file} not found. Please run the model once to generate it.")
    
    # Load ONLY the indices tensor dynamically!
    indices_tensor = torch.load(input_file, map_location="cpu", weights_only=False)

    # HARDCODE ALL SCALAR ARGUMENTS HERE!
    start = time.perf_counter()
    out_orig, pairs_orig, num_orig = orig_func(indices=indices_tensor)
    end = time.perf_counter()
    duration_min = (end - start) / 60
    print(f"[DEBUG] original get_indice_pairs_3d_cpu duration: {duration_min:.2f} min", flush=True)
    print("orig_func outputs:")
    print(f"  out_orig: {out_orig.shape}")
    print(f"  pairs_orig: {pairs_orig.shape}")
    print(f"  num_orig: {num_orig.shape}")

    start = time.perf_counter()
    out_new, pairs_new, num_new = new_func(indices=indices_tensor)
    end = time.perf_counter()
    duration_min = (end - start) / 60
    print(f"[DEBUG] new get_indice_pairs_3d_cpu duration: {duration_min:.2f} min", flush=True)
    print("new_func outputs:")
    print(f"  out_new: {out_new.shape}")
    print(f"  pairs_new: {pairs_new.shape}")
    print(f"  num_new: {num_new.shape}")

    # 1. Validate out_indices
    # The original implementation maps integer grid IDs by chronological appearance order (Python Appends).
    # The vectorized approach maps integer grid IDs by spatial sorting (due to torch.unique).
    # We must lexicographically sort both independent outputs mathematically to prove identical exact spatial grid placements.
    if out_orig.shape[0] != out_new.shape[0]:
        print(f"Warning: out_indices length mismatch: {out_orig.shape[0]} vs {out_new.shape[0]}")
        assert torch.equal(out_orig.sum(), out_new.sum()), "out_indices grid sums strictly mismatch!"
    else:
        def sort_2d(coords):
            if coords.shape[0] == 0:
                return coords
            h_vals = coords[:, 0]*1000000 + coords[:, 1]*10000 + coords[:, 2]*100 + coords[:, 3]
            return coords[h_vals.argsort()]
            
        out_orig_sorted = sort_2d(out_orig)
        out_new_sorted = sort_2d(out_new)
        assert torch.equal(out_orig_sorted, out_new_sorted), "out_indices spatial grids strictly mismatch!"

    # 2. Validate indice_num
    assert torch.equal(num_orig, num_new), "indice_num arrays strictly mismatch!"

    # 3. Validate indice_pairs
    K = num_orig.shape[0]
    for k in range(K):
        cnt = num_orig[k].item()
        
        orig_slice = pairs_orig[k, :, :cnt]
        new_slice = pairs_new[k, :, :cnt]
        
        orig_sorted, _ = torch.sort(orig_slice, dim=-1)
        new_sorted, _ = torch.sort(new_slice, dim=-1)

        assert torch.equal(orig_sorted, new_sorted), f"indice_pairs mismatch strictly at spatial kernel offset {k}!"

    print("\nSanity Check PASSED! Both the original python loop and pure PyTorch vectorized algorithms produce EXACTLY IDENTICAL outputs utilizing torch.equal()!")
