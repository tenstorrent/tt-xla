import pytest
import os
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
from typing import Tuple

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
        
        # valid_flat_unique_mask = active_cells > 0
        # unique_flat = torch.where(valid_flat_unique_mask)[0].int() 
        # grids_out = torch.where(active_cells > 0, active_id_map, -1)

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

def test_sanity_new():
    input_file = "/home/tt-xla/saved_inputs.pt"
    if not os.path.exists(input_file):
        pytest.skip(f"Input file {input_file} not found. Generate it from the main model first.")

    # Load ONLY the indices tensor dynamically!
    inputs = torch.load(input_file, map_location="cpu", weights_only=False)
    indices_tensor = inputs["indices"] if isinstance(inputs, dict) else inputs

    class NewModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, indices):
            # Evaluate using the isolated standalone function dynamically hosted directly in this script!
            return get_indice_pairs_3d_cpu_new(indices=indices)

    model = NewModel()

    tester = OpTester(comparison_config=ComparisonConfig(), framework=Framework.TORCH)
    
    # TT-MLIR Tracing: Expose ONLY the PyTorch tensor down to the MLIR graph arguments
    workload = Workload(
        framework=Framework.TORCH,
        model=model,
        args=[indices_tensor],
    )

    logger.info("Executing New Pure PyTorch Vectorized algorithm across XLA TT-MLIR Workload...")
    tester.test(workload)
