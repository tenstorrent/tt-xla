"""
PCC Debug Script #2: Drill into bounds-check + matmul index computation.

Previous test found:
  - Coordinates, floor, embedding are all fine individually
  - 162,398 flat index mismatches exist
  - Bug is in bounds-check/clamping OR matmul index flattening

This script replicates the EXACT codegen ops for Corner 0 (floor_x, floor_y)
step by step and compares each intermediate against PyTorch reference.

Usage:
    python grid_codegen/test_pcc_debug2.py
"""

import torch
import torch.nn.functional as F
import ttnn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils


def compute_pcc(a, b):
    """PCC between two flat tensors."""
    if isinstance(a, ttnn.Tensor):
        a = ttnn.to_torch(a).float().flatten()
    else:
        a = a.float().flatten()
    b = b.float().flatten()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() == 0 and b.std() == 0:
        return 1.0
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def exact_match_pct(a, b):
    """% exact match for integer tensors."""
    if isinstance(a, ttnn.Tensor):
        a = ttnn.to_torch(a).flatten()
    else:
        a = a.flatten()
    b = b.flatten()
    n = min(len(a), len(b))
    a, b = a[:n].long(), b[:n].long()
    match = (a == b).float().mean().item() * 100
    mismatch = (a != b).sum().item()
    return match, mismatch


def check(name, tt_val, ref_val, is_index=False):
    """Print comparison result for one intermediate."""
    pcc = compute_pcc(tt_val, ref_val)
    status = "✓" if pcc > 0.999 else "✗ <<<< ISSUE"
    print(f"  {name:45s} PCC: {pcc:.6f}  {status}")
    if is_index:
        match, mis = exact_match_pct(tt_val, ref_val)
        idx_status = "✓" if match > 99.9 else f"✗ <<<< {mis} MISMATCHES"
        print(f"  {'  └─ exact match':45s}      {match:.2f}%   {idx_status}")
    return pcc


def test_pcc_debug2():
    print("=" * 70)
    print("PCC DEBUG #2: Bounds-check + Matmul index drill-down")
    print("=" * 70)

    # Load inputs
    grid_cpu = torch.load("/home/tt-xla/grid_sample_grid_l.pt", map_location='cpu').to(torch.bfloat16)
    value_cpu = torch.load("/home/tt-xla/grid_sample_value_l.pt", map_location='cpu').to(torch.bfloat16)
    N, C, H, W = value_cpu.shape  # 8, 32, 25, 34
    print(f"value: {value_cpu.shape}, grid: {grid_cpu.shape}")

    device = utils.DeviceGetter.get_device((1, 1))
    MC = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

    # ====================================================================
    # PyTorch reference (float32) — Step by step matching codegen exactly
    # ====================================================================
    grid_f32 = grid_cpu.float()
    value_f32 = value_cpu.float()

    # Coord transform
    grid_5d = grid_f32.unsqueeze(1).expand(-1, C, -1, -1, -1)  # [8,32,850,4,2]
    grid_x = grid_5d[..., 0]  # [8,32,850,4]
    grid_y = grid_5d[..., 1]

    ref_ix = grid_x * 17.0 + 16.5
    ref_iy = grid_y * 12.5 + 12.0
    ref_ix_floor = torch.floor(ref_ix)
    ref_iy_floor = torch.floor(ref_iy)

    # Bounds checks for corner 0: (ix_floor, iy_floor)
    # From codegen: ge_0 = ix_floor >= 0, gt_2 = 34.0 > ix_floor
    #               ge_1 = iy_floor >= 0, gt_3 = 25.0 > iy_floor
    ref_ge_0 = (ref_ix_floor >= 0).float()         # ix_floor >= 0
    ref_gt_2 = (34.0 > ref_ix_floor).float()       # 34 > ix_floor
    ref_ge_1 = (ref_iy_floor >= 0).float()          # iy_floor >= 0
    ref_gt_3 = (25.0 > ref_iy_floor).float()        # 25 > iy_floor
    ref_and_0 = (ref_ge_1 * ref_gt_3).float()       # y in bounds
    ref_and_1 = (ref_gt_2 * ref_and_0).float()      # x < 34 AND y in bounds
    ref_and_2 = (ref_ge_0 * ref_and_1).float()      # full in-bounds mask

    # Safe y-index (codegen lines 733-770):
    # typecast_11 = INT32(floor_y), typecast_13 = FLOAT32(typecast_11)
    # where_2 = where(in_bounds, typecast_13, 0.0)
    # typecast_14 = INT32(where_2)
    # Then clamp: gt_4 = (25 > typecast_14), add_4 = typecast_14 + 25
    # where_3 = where(gt_4, add_4_as_float, where_2)
    # typecast_17 = INT32(where_3)
    ref_iy_int = ref_iy_floor.long()
    ref_iy_safe_float = torch.where(ref_and_2.bool(), ref_iy_floor, torch.zeros_like(ref_iy_floor))
    ref_iy_safe_int = ref_iy_safe_float.long()

    # Clamp chain for y: if iy_safe < 25 (already positive from where), keep it
    # The codegen does: gt_4 = (25 > iy_safe_int) — for values already clamped to 0, this is True
    # add_4 = iy_safe_int + 25 — this is the "wrap-around" for negative values
    # Since we already set out-of-bounds to 0, and 0 < 25, gt_4 is True, so where_3 = add_4 in float
    # Wait — let me re-read the codegen more carefully...

    # Safe x-index (same pattern, codegen lines 831-920)
    ref_ix_safe_float = torch.where(ref_and_2.bool(), ref_ix_floor, torch.zeros_like(ref_ix_floor))
    ref_ix_safe_int = ref_ix_safe_float.long()

    # Batch/channel indices (from const_eval_10)
    ref_batch_idx = torch.arange(N).reshape(N, 1, 1, 1).expand(N, C, 850, 4)
    ref_chan_idx = torch.arange(C).reshape(1, C, 1, 1).expand(N, C, 850, 4)

    # Flat index for corner 0
    ref_flat_00 = ref_batch_idx * (C * H * W) + ref_chan_idx * (H * W) + ref_iy_safe_int * W + ref_ix_safe_int

    # ====================================================================
    # TTNN: Replicate codegen step by step for corner 0
    # ====================================================================
    print("\n--- Replicating codegen for Corner 0 (ix_floor, iy_floor) ---\n")

    # Prepare inputs on device
    tt_grid = ttnn.from_torch(grid_cpu, device=device, layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.BFLOAT16)
    tt_value = ttnn.from_torch(value_cpu, device=device, layout=ttnn.Layout.ROW_MAJOR, dtype=ttnn.DataType.BFLOAT16)

    # --- Coord transform (already verified, but needed as inputs) ---
    tt_grid_tiled = ttnn.to_layout(tt_grid, ttnn.Layout.TILE, None, memory_config=None)
    tt_grid_f32 = ttnn.typecast(tt_grid_tiled, ttnn.DataType.FLOAT32, memory_config=MC)
    ttnn.deallocate(tt_grid_tiled, False)
    tt_grid_5d = ttnn.reshape(tt_grid_f32, [8, 1, 850, 4, 2], memory_config=MC)
    ttnn.deallocate(tt_grid_f32, False)
    tt_grid_rep = ttnn.repeat(tt_grid_5d, ttnn.Shape([1, 32, 1, 1, 1]), memory_config=MC)
    ttnn.deallocate(tt_grid_5d, False)

    tt_gx_5d = ttnn.slice(tt_grid_rep, [0,0,0,0,0], [8,32,850,4,1], [1,1,1,1,1], memory_config=MC)
    tt_gx = ttnn.reshape(tt_gx_5d, [8,32,850,4], memory_config=MC)
    ttnn.deallocate(tt_gx_5d, False)
    tt_gy_5d = ttnn.slice(tt_grid_rep, [0,0,0,0,1], [8,32,850,4,2], [1,1,1,1,1], memory_config=MC)
    ttnn.deallocate(tt_grid_rep, False)
    tt_gy = ttnn.reshape(tt_gy_5d, [8,32,850,4], memory_config=MC)
    ttnn.deallocate(tt_gy_5d, False)

    # Constants - Matching main.py exactly
    c17 = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=17.0, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    c16_5 = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=16.5, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    c12_5 = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=12.5, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    c12 = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=12.0, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    
    # Bound constants from main.py: _0_2 = 0.0 (f32), _0_16 = 34.0 (f32), _0_4 = 25.0 (f32)
    c0_f = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=0.0, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    c34_f = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=34.0, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    c25_f = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=25.0, dtype=ttnn.DataType.FLOAT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    
    # Integer constants for indexing
    c0_i = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=0, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    c25_i = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=25, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    c34_i = ttnn.full(shape=ttnn.Shape([1,1,1,1]), fill_value=34, dtype=ttnn.DataType.INT32, layout=ttnn.Layout.TILE, device=device, memory_config=MC)
    
    # ix = gx * 17 + 16.5
    tt_ix_pre = ttnn.multiply(tt_gx, c17, dtype=ttnn.DataType.FLOAT32, memory_config=MC)
    tt_ix = ttnn.add(tt_ix_pre, c16_5, dtype=ttnn.DataType.FLOAT32, memory_config=MC)
    ttnn.deallocate(tt_ix_pre, False)

    # iy = gy * 12.5 + 12
    tt_iy_pre = ttnn.multiply(tt_gy, c12_5, dtype=ttnn.DataType.FLOAT32, memory_config=MC)
    tt_iy = ttnn.add(tt_iy_pre, c12, dtype=ttnn.DataType.FLOAT32, memory_config=MC)
    ttnn.deallocate(tt_iy_pre, False)

    # floor
    tt_ix_floor = ttnn.floor(tt_ix, memory_config=MC)
    tt_iy_floor = ttnn.floor(tt_iy, memory_config=MC)

    check("ix", tt_ix, ref_ix)
    check("iy", tt_iy, ref_iy)
    check("ix_floor", tt_ix_floor, ref_ix_floor, is_index=True)
    check("iy_floor", tt_iy_floor, ref_iy_floor, is_index=True)

    # ====================================================================
    # Bounds checking
    # ====================================================================
    print("\n--- Bounds Checking Ops ---\n")

    # ge_0: ix_floor >= 0.0 (f32)  (codegen line 632)
    tt_ge_0 = ttnn.ge(tt_ix_floor, c0_f, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    check("ge_0: ix_floor >= 0", tt_ge_0, ref_ge_0)

    # gt_2: 34.0 (f32) > ix_floor  (codegen line 640)
    tt_gt_2 = ttnn.gt(c34_f, tt_ix_floor, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    check("gt_2: 34 > ix_floor", tt_gt_2, ref_gt_2)

    # ge_1: iy_floor >= 0.0 (f32)  (codegen line 690)
    tt_ge_1 = ttnn.ge(tt_iy_floor, c0_f, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    check("ge_1: iy_floor >= 0", tt_ge_1, ref_ge_1)

    # gt_3: 25.0 (f32) > iy_floor  (codegen line 698)
    tt_gt_3 = ttnn.gt(c25_f, tt_iy_floor, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    check("gt_3: 25 > iy_floor", tt_gt_3, ref_gt_3)

    # logical_and_0: ge_1 AND gt_3  (y in bounds)  (codegen line 706)
    tt_and_0 = ttnn.logical_and(tt_ge_1, tt_gt_3, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    check("and_0: (iy>=0) AND (25>iy)", tt_and_0, ref_and_0)

    # logical_and_1: gt_2 AND and_0  (codegen line 716)
    tt_and_1 = ttnn.logical_and(tt_gt_2, tt_and_0, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    check("and_1: (34>ix) AND y_inbounds", tt_and_1, ref_and_1)

    # logical_and_2: ge_0 AND and_1  (full in-bounds mask)  (codegen line 724)
    tt_and_2 = ttnn.logical_and(tt_ge_0, tt_and_1, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    check("and_2: FULL in-bounds mask", tt_and_2, ref_and_2, is_index=True)

    # ====================================================================
    # Safe index computation — where + clamp chain
    # Codegen lines 733-770 (y-index), 831-920 (x-index)
    # ====================================================================
    print("\n--- Safe Index (y-coord) ---\n")

    # typecast_11: INT32(floor_y)  (codegen line 733)
    tt_tc11 = ttnn.typecast(tt_iy_floor, ttnn.DataType.INT32, memory_config=MC)
    ref_tc11 = ref_iy_floor.to(torch.int32)
    check("typecast_11: INT32(iy_floor)", tt_tc11, ref_tc11, is_index=True)

    # typecast_12: FLOAT32(in_bounds_mask)  (codegen line 740)
    tt_tc12 = ttnn.typecast(tt_and_2, ttnn.DataType.FLOAT32, memory_config=MC)
    check("typecast_12: FLOAT32(mask)", tt_tc12, ref_and_2)

    # typecast_13: FLOAT32(INT32(floor_y))  (codegen line 748)
    tt_tc13 = ttnn.typecast(tt_tc11, ttnn.DataType.FLOAT32, memory_config=MC)
    ref_tc13 = ref_iy_floor.float()  # same as ref_iy_floor since it was already float
    check("typecast_13: FLOAT32(INT32(iy_floor))", tt_tc13, ref_tc13, is_index=True)

    # Zeros for out-of-bounds: repeat c0_i to [8,32,850,4] (from const_eval_1)
    tt_zeros = ttnn.repeat(c0_f, ttnn.Shape([8, 32, 850, 4]), memory_config=MC)

    # where_2: where(mask, iy_floor_as_float, 0.0)  (codegen line 756)
    tt_where_2 = ttnn.where(tt_tc12, tt_tc13, tt_zeros, memory_config=MC)
    ref_where_2 = torch.where(ref_and_2.bool(), ref_iy_floor, torch.zeros_like(ref_iy_floor))
    check("where_2: safe_iy (masked)", tt_where_2, ref_where_2, is_index=True)

    # typecast_14: INT32(where_2)  (codegen line 764)
    tt_tc14 = ttnn.typecast(tt_where_2, ttnn.DataType.INT32, memory_config=MC)
    ref_tc14 = ref_where_2.to(torch.int32)
    check("typecast_14: INT32(safe_iy)", tt_tc14, ref_tc14, is_index=True)

    # gt_4: (25 > typecast_14)  — checks if safe_iy < H  (codegen line 771)
    tt_gt_4 = ttnn.gt(c25_i, tt_tc14, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    ref_gt_4 = (25 > ref_tc14.long()).float()
    check("gt_4: (25 > safe_iy_int)", tt_gt_4, ref_gt_4)

    # add_4: typecast_14 + 25  (modular wrap)  (codegen line 779)
    tt_add_4 = ttnn.add(tt_tc14, c25_i, dtype=ttnn.DataType.INT32, memory_config=MC)
    ref_add_4 = (ref_tc14.long() + 25)
    check("add_4: safe_iy + 25", tt_add_4, ref_add_4, is_index=True)

    # typecast_15 = FLOAT32(gt_4)
    tt_tc15 = ttnn.typecast(tt_gt_4, ttnn.DataType.FLOAT32, memory_config=MC)
    # typecast_16 = FLOAT32(add_4)
    tt_tc16 = ttnn.typecast(tt_add_4, ttnn.DataType.FLOAT32, memory_config=MC)

    # where_3: where(gt_4_float, add_4_float, where_2)  (codegen line 804)
    tt_where_3 = ttnn.where(tt_tc15, tt_tc16, tt_where_2, memory_config=MC)
    ref_where_3 = torch.where(ref_gt_4.bool(), ref_add_4.float(), ref_where_2)
    check("where_3: clamped_safe_iy", tt_where_3, ref_where_3, is_index=True)

    # typecast_17 = INT32(where_3)  — final safe y index  (codegen line 815)
    tt_tc17 = ttnn.typecast(tt_where_3, ttnn.DataType.INT32, memory_config=MC)
    ref_tc17 = ref_where_3.to(torch.int32)
    check("typecast_17: FINAL safe_iy_int", tt_tc17, ref_tc17, is_index=True)

    # ====================================================================
    # Safe index (x-coord) — same pattern, codegen lines 831-920
    # ====================================================================
    print("\n--- Safe Index (x-coord) ---\n")

    # typecast_18 = INT32(ix_floor)
    tt_tc18 = ttnn.typecast(tt_ix_floor, ttnn.DataType.INT32, memory_config=MC)
    ref_tc18 = ref_ix_floor.to(torch.int32)
    check("typecast_18: INT32(ix_floor)", tt_tc18, ref_tc18, is_index=True)

    # typecast_19 = FLOAT32(typecast_18)
    tt_tc19 = ttnn.typecast(tt_tc18, ttnn.DataType.FLOAT32, memory_config=MC)
    ref_tc19 = ref_ix_floor.float()
    check("typecast_19: FLOAT32(INT32(ix_floor))", tt_tc19, ref_tc19, is_index=True)

    # where_4: where(mask, ix_floor_as_float, 0.0)
    tt_where_4 = ttnn.where(tt_tc12, tt_tc19, tt_zeros, memory_config=MC)
    ref_where_4 = torch.where(ref_and_2.bool(), ref_ix_floor, torch.zeros_like(ref_ix_floor))
    check("where_4: safe_ix (masked)", tt_where_4, ref_where_4, is_index=True)

    # typecast_20 = INT32(where_4)
    tt_tc20 = ttnn.typecast(tt_where_4, ttnn.DataType.INT32, memory_config=MC)
    ref_tc20 = ref_where_4.to(torch.int32)
    check("typecast_20: INT32(safe_ix)", tt_tc20, ref_tc20, is_index=True)

    # gt_5: (34 > typecast_20)
    tt_gt_5 = ttnn.gt(c34_i, tt_tc20, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)
    ref_gt_5 = (34 > ref_tc20.long()).float()
    check("gt_5: (34 > safe_ix_int)", tt_gt_5, ref_gt_5)

    # add_5: typecast_20 + 34
    tt_add_5 = ttnn.add(tt_tc20, c34_i, dtype=ttnn.DataType.INT32, memory_config=MC)
    ref_add_5 = (ref_tc20.long() + 34)
    check("add_5: safe_ix + 34", tt_add_5, ref_add_5, is_index=True)

    # typecast_21 = FLOAT32(gt_5), typecast_22 = FLOAT32(add_5)
    tt_tc21 = ttnn.typecast(tt_gt_5, ttnn.DataType.FLOAT32, memory_config=MC)
    tt_tc22 = ttnn.typecast(tt_add_5, ttnn.DataType.FLOAT32, memory_config=MC)

    # where_5: where(gt_5_float, add_5_float, where_4)
    tt_where_5 = ttnn.where(tt_tc21, tt_tc22, tt_where_4, memory_config=MC)
    ref_where_5 = torch.where(ref_gt_5.bool(), ref_add_5.float(), ref_where_4)
    check("where_5: clamped_safe_ix", tt_where_5, ref_where_5, is_index=True)

    # typecast_23 = INT32(where_5)  — final safe x index
    tt_tc23 = ttnn.typecast(tt_where_5, ttnn.DataType.INT32, memory_config=MC)
    ref_tc23 = ref_where_5.to(torch.int32)
    check("typecast_23: FINAL safe_ix_int", tt_tc23, ref_tc23, is_index=True)

    # ====================================================================
    # Matmul-based flat index computation (codegen lines 921-970)
    # ====================================================================
    print("\n--- Matmul Index Flattening ---\n")

    # The codegen builds: concat([batch_idx_5d, chan_idx_5d, safe_iy_5d, safe_ix_5d], dim=-1)
    # Then matmul with strides [27200, 850, 34, 1]

    # Reshape safe indices to 5D for concat
    tt_tc17_5d = ttnn.reshape(tt_tc17, [8, 32, 850, 4, 1], memory_config=MC)
    tt_tc23_5d = ttnn.reshape(tt_tc23, [8, 32, 850, 4, 1], memory_config=MC)

    # Build batch and channel index tensors (from const_eval_10)
    # _2 = batch index reshaped [8,32,850,4,1], _3 = chan index reshaped [8,32,850,4,1]
    # These come from const_eval_10 and are pre-computed
    batch_tensor = ttnn.Tensor(
        [0, 1, 2, 3, 4, 5, 6, 7], [8],
        ttnn.DataType.INT32, ttnn.Layout.TILE, device, memory_config=MC)
    batch_reshaped = ttnn.reshape(batch_tensor, [8, 1, 1, 1], memory_config=MC)

    # Build the full batch index [8,32,850,4,1] via the codegen's where approach
    # The codegen uses a complex gt/where/repeat pattern, but the result should just be
    # batch_idx repeated. Let's check what const_eval_10 actually produces.
    # For simplicity, let's build it in PyTorch and send to device.
    ref_batch_5d = ref_batch_idx.reshape(N, C, 850, 4, 1).to(torch.int32)
    ref_chan_5d = ref_chan_idx.reshape(N, C, 850, 4, 1).to(torch.int32)

    tt_batch_5d = ttnn.from_torch(ref_batch_5d, device=device, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.INT32)
    tt_chan_5d = ttnn.from_torch(ref_chan_5d, device=device, layout=ttnn.Layout.TILE, dtype=ttnn.DataType.INT32)

    # Concat [batch, chan, y, x] along dim -1 → [8,32,850,4,4]
    concat_list = [tt_batch_5d, tt_chan_5d, tt_tc17_5d, tt_tc23_5d]
    tt_concat = ttnn.concat(concat_list, 4, memory_config=MC)

    ref_tc17_5d = ref_tc17.reshape(N, C, 850, 4, 1)
    ref_tc23_5d = ref_tc23.reshape(N, C, 850, 4, 1)
    ref_concat = torch.cat([ref_batch_5d, ref_chan_5d, ref_tc17_5d, ref_tc23_5d], dim=-1)
    check("concat [batch,chan,y,x]", tt_concat, ref_concat, is_index=True)

    # Typecast to FLOAT32 for matmul
    tt_concat_f32 = ttnn.typecast(tt_concat, ttnn.DataType.FLOAT32, memory_config=MC)
    ref_concat_f32 = ref_concat.float()
    check("concat as FLOAT32", tt_concat_f32, ref_concat_f32)

    # Stride tensor [27200, 850, 34, 1] → [4, 1]
    stride_tensor = ttnn.Tensor(
        [27200.0, 850.0, 34.0, 1.0], [4, 1],
        ttnn.DataType.FLOAT32, ttnn.Layout.TILE, device, memory_config=MC)
    ref_stride = torch.tensor([[27200.0], [850.0], [34.0], [1.0]])

    # Matmul: [8,32,850,4,4] × [4,1] → [8,32,850,4,1]
    tt_matmul = ttnn.matmul(
        tt_concat_f32, stride_tensor,
        transpose_a=False, transpose_b=False, memory_config=MC,
        dtype=ttnn.DataType.FLOAT32, program_config=None, activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True))

    ref_matmul = torch.matmul(ref_concat_f32, ref_stride)
    check("matmul flat index", tt_matmul, ref_matmul, is_index=True)

    # Reshape to [1, 870400]
    tt_flat = ttnn.reshape(tt_matmul, [1, 870400], memory_config=MC)
    ref_flat = ref_matmul.reshape(1, 870400)
    check("flat index [1, 870400]", tt_flat, ref_flat, is_index=True)

    # Typecast to UINT32
    tt_flat_u32 = ttnn.typecast(tt_flat, ttnn.DataType.UINT32, memory_config=MC)
    ref_flat_u32 = ref_flat.to(torch.int32)  # closest equivalent
    check("typecast to UINT32", tt_flat_u32, ref_flat_u32, is_index=True)

    # ====================================================================
    # Embedding lookup
    # ====================================================================
    print("\n--- Embedding Lookup (Corner 0) ---\n")

    tt_flat_rm = ttnn.to_layout(tt_flat_u32, ttnn.Layout.ROW_MAJOR, None, memory_config=None)

    # Prepare value as flat [217600, 1] in BF16 ROW_MAJOR
    tt_value_tiled = ttnn.to_layout(tt_value, ttnn.Layout.TILE, None, memory_config=None)
    tt_value_f32 = ttnn.typecast(tt_value_tiled, ttnn.DataType.FLOAT32, memory_config=MC)
    ttnn.deallocate(tt_value_tiled, False)
    tt_value_flat = ttnn.reshape(tt_value_f32, [217600, 1], memory_config=MC)
    ttnn.deallocate(tt_value_f32, False)
    tt_value_bf16 = ttnn.typecast(tt_value_flat, ttnn.DataType.BFLOAT16, memory_config=MC)
    ttnn.deallocate(tt_value_flat, False)
    tt_value_rm = ttnn.to_layout(tt_value_bf16, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(tt_value_bf16, False)

    tt_emb = ttnn.embedding(tt_flat_rm, tt_value_rm, padding_idx=None,
        layout=ttnn.Layout.TILE, dtype=ttnn.DataType.BFLOAT16, memory_config=MC)

    ref_flat_clamped = ref_flat_u32.long().flatten().clamp(0, 217599)
    ref_value_flat = value_cpu.float().reshape(-1, 1)
    ref_emb = ref_value_flat[ref_flat_clamped]

    tt_emb_f32 = ttnn.typecast(tt_emb, ttnn.DataType.FLOAT32, memory_config=MC)
    tt_emb_reshaped = ttnn.reshape(tt_emb_f32, [8, 32, 850, 4], memory_config=MC)
    ref_emb_reshaped = ref_emb.reshape(8, 32, 850, 4)

    check("embedding output (corner 0)", tt_emb_reshaped, ref_emb_reshaped)

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: The FIRST op above showing ✗ is the culprit.")
    print("=" * 70)


if __name__ == "__main__":
    test_pcc_debug2()
