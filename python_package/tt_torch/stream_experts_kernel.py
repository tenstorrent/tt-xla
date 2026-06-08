# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Raw-kernel artifact builder for the interleaved-DRAM selective-experts matmul.

Path A: every operand is a plain INTERLEAVED DRAM tensor (the layout the tt-xla
compiler hands an opaque ``ttnn.tt_lang_op`` operand), read/written via
``InterleavedAddrGenFast``. The N output columns are split across compute cores so
the selected experts stream in parallel. Weights are bfp4 (the custom op applies a
weight_dtype_override) -> 4x less DRAM traffic than bf16, the memory-bound win.

Operand order (set by custom_ops.stream_experts_matmul): 0=in0 [R,1,K], 1=in1
[E,K,N] (bfp4), 2=index [1,R] int32, 3=out [R,1,N] bf16; R = T*k.
"""

from __future__ import annotations

import os
from typing import Any, List, Sequence

TILE = 32
_BF16_TILE_BYTES = TILE * TILE * 2   # 2048
_BFP8_TILE_BYTES = 1024 + 64         # 1088 (bfp8_b: 8-bit mantissa + per-face exp)
_BFP4_TILE_BYTES = 512 + 64          # 576 (bfp4_b: 4-bit mantissa + per-face exp)
_INT32_TILE_BYTES = TILE * TILE * 4  # 4096
_DF_FLOAT16_B = 5                    # tt::DataFormat::Float16_b
_DF_BFP8_B = 6                       # tt::DataFormat::Bfp8_b
_DF_BFP4_B = 7                       # tt::DataFormat::Bfp4_b
_DF_INT32 = 8                        # tt::DataFormat::Int32

# Expert-weight streaming dtype. "bfp8"/"bfp4" -> the custom op applies the matching
# weight_dtype_override and the artifact reads in1 at this format. Keep bfp8 (matches
# the dense linears): bfp4 saves DRAM but halves decode top-1 (39% vs 80%) for ~2ms.
_IN1_DTYPE = "bfp8"
_IN1_DF = {"bf16": _DF_FLOAT16_B, "bfp8": _DF_BFP8_B, "bfp4": _DF_BFP4_B}[_IN1_DTYPE]
_IN1_PAGE = {"bf16": _BF16_TILE_BYTES, "bfp8": _BFP8_TILE_BYTES, "bfp4": _BFP4_TILE_BYTES}[_IN1_DTYPE]
_IN1_CB_FMT = {"bf16": "BFloat16", "bfp8": "BFP_BFloat8", "bfp4": "BFP_BFloat4"}[_IN1_DTYPE]
WEIGHT_DTYPE_OVERRIDE = {"bf16": None, "bfp8": "bfp_bf8", "bfp4": "bfp_bf4"}[_IN1_DTYPE]
_MAX_CORES = 12                      # cap; pick the largest divisor of Nt <= this

KERNEL_PATH = (
    "models/demos/deepseek_v3_b1/micro_ops/dram_streaming_experts_matmul/kernels/"
    "interleaved_experts_matmul_kernel.cpp"
)
_TT_METAL_HOME_FALLBACK = (
    "/localdev/sshon/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal"
)
STREAM_EXPERTS_KERNEL_ID = "tt_interleaved_experts_matmul"
STREAM_EXPERTS_VERSION = "wh-interleaved-mc-bfp8-v1"

# CB indices (working L1 scratch).
_CB_IN0, _CB_IN1, _CB_INDEX, _CB_OUT = 0, 1, 2, 16


def _read_kernel_source() -> str:
    for home in (os.environ.get("TT_METAL_HOME", ""), _TT_METAL_HOME_FALLBACK):
        if not home:
            continue
        try:
            with open(os.path.join(home, KERNEL_PATH)) as f:
                return f.read()
        except OSError:
            continue
    return ""


def _pick_num_cores(nt: int) -> int:
    for nc in range(min(nt, _MAX_CORES), 0, -1):
        if nt % nc == 0:
            return nc
    return 1


def _ct(name: str, value: int) -> List[Any]:
    return [name, int(value)]


def build_stream_experts_artifact(
    *,
    shapes: Sequence[Sequence[int]],
    dtypes: Sequence[str] = (),
    layouts: Sequence[str] = (),
    mesh_shape: Sequence[int] = (),
    arg_roles: Any = None,
    shard_spec: Any = None,
) -> dict:
    """Build the multi-core interleaved selective-experts matmul artifact."""
    if len(shapes) < 4:
        raise ValueError(f"interleaved experts artifact needs 4 operands, got {len(shapes)}")

    R = int(shapes[3][0])         # out [R,1,N]: R = T*k output rows
    in0_rows = int(shapes[0][0])  # in0 [in0_rows,1,K]: T (gate_up, reused) or R (down)
    k_reuse = max(1, R // in0_rows)  # experts/token sharing one in0 (gate_up=k, down=1)
    K = int(shapes[0][-1])
    E = int(shapes[1][0])         # in1 [E,K,N]
    N = int(shapes[1][-1])
    assert K % TILE == 0, f"K({K}) not tile-aligned"
    assert N % TILE == 0, f"N({N}) not tile-aligned"
    Kt, Nt = K // TILE, N // TILE
    idx_tiles = (R + TILE - 1) // TILE
    num_cores = _pick_num_cores(Nt)
    per_core_n = Nt // num_cores

    ct = [
        _ct("iem_cb_in0", _CB_IN0), _ct("iem_cb_in1", _CB_IN1),
        _ct("iem_cb_index", _CB_INDEX), _ct("iem_cb_out", _CB_OUT),
        _ct("iem_k_tiles", Kt), _ct("iem_n_tiles", Nt),
        _ct("iem_num_rows", R), _ct("iem_num_experts", E),
        _ct("iem_in0_page", _BF16_TILE_BYTES), _ct("iem_in1_page", _IN1_PAGE),
        _ct("iem_index_page", _INT32_TILE_BYTES), _ct("iem_out_page", _BF16_TILE_BYTES),
        _ct("iem_in0_df", _DF_FLOAT16_B), _ct("iem_in1_df", _IN1_DF),
        _ct("iem_out_df", _DF_FLOAT16_B), _ct("iem_index_df", _DF_INT32),
        _ct("iem_per_core_n", per_core_n),  # index 16
        _ct("iem_k_reuse", k_reuse),        # index 17
    ]

    # Compute cores: row-major on the 8-wide Tensix grid. core_id = list position.
    core_list = [[i % 8, i // 8] for i in range(num_cores)]
    per_core_rt = [[["iem_core_id", i]] for i in range(num_cores)]

    kernel_source = _read_kernel_source()
    if not kernel_source:
        raise RuntimeError(f"could not read kernel source for {KERNEL_PATH}")

    return {
        "kind": "raw_generic",
        "format_version": 1,
        "kernel_source": kernel_source,
        "core_list": core_list,
        "kernels": [
            {"thread_type": "ncrisc", "ct_args": ct,
             "per_core_rt_args": per_core_rt, "common_addr_operands": [0, 1, 2, 3]},
            {"thread_type": "trisc", "ct_args": ct,
             "compute_config": {"math_fidelity": "LoFi", "fp32_dest_acc_en": True,
                                "math_approx_mode": False}},
            {"thread_type": "brisc", "ct_args": ct,
             "per_core_rt_args": per_core_rt, "common_addr_operands": [0, 1, 2, 3]},
        ],
        "cb_descriptors": [
            {"cb": _CB_IN0, "data_format": "BFloat16", "page_size": _BF16_TILE_BYTES,
             "total_size": 2 * Kt * _BF16_TILE_BYTES},
            {"cb": _CB_IN1, "data_format": _IN1_CB_FMT, "page_size": _IN1_PAGE,
             "total_size": 3 * Kt * _IN1_PAGE},
            {"cb": _CB_INDEX, "data_format": "Int32", "page_size": _INT32_TILE_BYTES,
             "total_size": idx_tiles * _INT32_TILE_BYTES},
            {"cb": _CB_OUT, "data_format": "BFloat16", "page_size": _BF16_TILE_BYTES,
             "total_size": 2 * _BF16_TILE_BYTES},
        ],
        "num_tensors": 4,
        "_shape_derived": {"R": R, "K": K, "N": N, "E": E, "Kt": Kt, "Nt": Nt,
                           "num_cores": num_cores, "per_core_n": per_core_n},
    }


def register() -> None:
    """Register the interleaved-experts raw kernel with the tt-lang registry (idempotent)."""
    from . import tt_lang

    try:
        tt_lang.get_registered_kernel(STREAM_EXPERTS_KERNEL_ID)
        return
    except KeyError:
        pass
    tt_lang.register_raw_kernel(
        kernel_id=STREAM_EXPERTS_KERNEL_ID,
        artifact_builder=build_stream_experts_artifact,
        arg_roles=("in", "in", "in", "out"),
        version_tag=STREAM_EXPERTS_VERSION,
    )
