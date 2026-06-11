#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone TTNN perf repro for the split_query_key_value_and_split_heads patch.

Extracted from:
  modules/irs/ttnn_glm_4_7_tp_galaxy_4_layers_1lyr_bs64_isl128_rund8a2_g1_*.mlir

The patch under test (per-device, after the 4x8 galaxy mesh shard) is:

    %23 : [16, 1792]            bf16, TILE, DRAM interleaved   (output of ttnn.linear)
    %24 = reshape(%23)       -> [16, 1, 1792]
    q, k, v = split_query_key_value_and_split_heads(%24,
                  num_heads=12, num_kv_heads=1, transpose_key=false)
              q -> [16, 12, 1, 128]
              k -> [16,  1, 1, 128]
              v -> [16,  1, 1, 128]
    %56 = reshape(q)         -> [1, 16, 12, 128]   (in model: after rms_norm + rotary)
    %42 = reshape(k)         -> [1, 16,  1, 128]   (in model: after rms_norm + rotary)
    %25 = reshape(v)         -> [1, 16,  1, 128]

This script reproduces the input reshape + split_qkv + a reshape on each of the three
outputs (query/key/value) so they can be profiled in isolation and compared against a
baseline while iterating on perf optimizations of the split_qkv kernel. The query/key
reshapes sit after rms_norm + rotary_embedding in the full model, but those ops preserve
shape and layout, so reshaping the raw outputs reproduces the same TM cost.

How to run with Tracy (device-side op perf):

    export TT_METAL_HOME=/path/to/tt-metal
    python -m tracy -r -p -v -o split_qkv_baseline \\
        scripts/ttnn_split_qkv_repro.py --iters 200

This produces an `ops_perf_results_*.csv` under the tracy output dir. Filter the rows
whose OP CODE contains "NlpCreateQKVHeads" / "split_query_key_value" (and the
ReshapeDeviceOperation rows around the signpost) to read DEVICE FW / KERNEL durations.

Plain run (host-side wall time only, no device profiler):

    python scripts/ttnn_split_qkv_repro.py --iters 200
"""

import argparse
import time

import ttnn

try:
    # Provided by tt-metal `tools/tracy`. Emits a Tracy marker that shows up in the
    # profiler trace so the measured region is easy to isolate. No-op-ish when the
    # profiler is not attached.
    from tracy import signpost
except Exception:  # pragma: no cover - tracy not importable outside tt-metal

    def signpost(header, message=None):
        pass


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=16, help="per-device batch (galaxy 64/4=16)")
    p.add_argument("--seq", type=int, default=1, help="sequence size")
    p.add_argument("--num-heads", type=int, default=12)
    p.add_argument("--num-kv-heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--transpose-key", action="store_true", default=False)
    p.add_argument(
        "--variant",
        choices=("general", "decode"),
        default="general",
        help="general = split_query_key_value_and_split_heads (+ output reshapes); "
        "decode = experimental.nlp_create_qkv_heads_decode (outputs already in [1,B,H,D], "
        "input reshape is a free view -> reshapes are much cheaper)",
    )
    p.add_argument("--iters", type=int, default=200, help="profiled iterations")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations (not measured)")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument(
        "--check",
        action="store_true",
        help="run a torch reference and report max abs diff for the value output",
    )
    return p.parse_args()


def make_input(device, batch, seq, hidden):
    """Builds the [batch, hidden] linear output that feeds the patch.

    Matches #ttnn_layout62: bf16, TILE layout, DRAM interleaved.
    """
    import torch

    torch.manual_seed(0)
    host = torch.randn(batch, hidden, dtype=torch.bfloat16)
    dev = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return host, dev


def run_patch(linear_out, *, variant, batch, seq, hidden, num_heads, num_kv_heads, transpose_key):
    """Run the patch and return (query, key, value) in the [seq, batch, heads, head_dim] layout.

    Both variants produce value-identical outputs with the same shapes; they differ only in
    which TTNN op + reshapes are used to get there (see --variant).
    """
    if variant == "decode":
        return _run_patch_decode(linear_out, batch=batch, seq=seq, hidden=hidden,
                                 num_heads=num_heads, num_kv_heads=num_kv_heads)
    return _run_patch_general(linear_out, batch=batch, seq=seq, hidden=hidden,
                              num_heads=num_heads, num_kv_heads=num_kv_heads,
                              transpose_key=transpose_key)


def _run_patch_general(linear_out, *, batch, seq, hidden, num_heads, num_kv_heads, transpose_key):
    """Baseline IR sequence: input reshape -> split_qkv -> reshape on each output.

    Each of the three split_qkv outputs is reshaped into the attention/cache layout
    `[seq, batch, heads, head_dim]` (the leading batch is moved to dim 1). In the full
    model the query/key reshapes happen after rms_norm + rotary_embedding, but those ops
    preserve shape and layout, so reshaping the raw outputs reproduces the same TM cost:

        %25 value: [16,1,1,128]  -> [1,16,1,128]   (#ttnn_layout65 -> #ttnn_layout66)
        %42 key:   [16,1,1,128]  -> [1,16,1,128]   (#ttnn_layout65 -> #ttnn_layout66)
        %56 query: [16,12,1,128] -> [1,16,12,128]  (#ttnn_layout64 -> #ttnn_layout66)

    Device ops produced: input reshape (real), split_qkv, query reshape (real). The
    key/value reshapes are free views.
    """
    head_dim = hidden // (num_heads + 2 * num_kv_heads)

    # %24 = reshape([batch, hidden]) -> [batch, seq, hidden]  (real reshape: re-tiles 16 rows)
    x = ttnn.reshape(linear_out, (batch, seq, hidden))

    # q, k, v = split_query_key_value_and_split_heads(...)
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
        x,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        transpose_key=transpose_key,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Reshape on each split_qkv output (move batch to dim 1).
    query_r = ttnn.reshape(query, (seq, batch, num_heads, head_dim))  # [16,12,1,128] -> [1,16,12,128]
    key_r = ttnn.reshape(key, (seq, batch, num_kv_heads, head_dim))  # [16,1,1,128] -> [1,16,1,128]
    value_r = ttnn.reshape(value, (seq, batch, num_kv_heads, head_dim))  # [16,1,1,128] -> [1,16,1,128]

    ttnn.deallocate(x)
    return query_r, key_r, value_r


def _run_patch_decode(linear_out, *, batch, seq, hidden, num_heads, num_kv_heads):
    """Decode variant: experimental.nlp_create_qkv_heads_decode.

    The decode op consumes a fused `[1, 1, B, hidden]` matrix and emits Q/K/V directly in
    `[1, B, num_heads, head_dim]` / `[1, B, num_kv_heads, head_dim]` (height-sharded), which
    is exactly the layout the baseline produces only *after* the expensive query reshape.

    Reshapes are much cheaper here:
      - input reshape [B, hidden] -> [1, 1, B, hidden] is a FREE view (last-2-dim tiling of
        [B, hidden] is unchanged), vs the baseline input reshape which re-tiles the 16 rows.
      - NO output reshapes are needed (the op already returns [1, B, heads, head_dim]).
    """
    # Free view: [B, hidden] -> [1, S, B, hidden]  (S == seq == 1)
    x = ttnn.reshape(linear_out, (1, seq, batch, hidden))

    query, key, value = ttnn.experimental.nlp_create_qkv_heads_decode(
        x,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    ttnn.deallocate(x)
    return query, key, value


def main():
    args = parse_args()

    hidden = (args.num_heads + 2 * args.num_kv_heads) * args.head_dim
    print(
        f"[config] variant={args.variant} batch={args.batch} seq={args.seq} hidden={hidden} "
        f"(num_heads={args.num_heads}, num_kv_heads={args.num_kv_heads}, head_dim={args.head_dim}) "
        f"transpose_key={args.transpose_key}"
    )

    device = ttnn.open_device(device_id=args.device_id)
    try:
        host_in, linear_out = make_input(device, args.batch, args.seq, hidden)

        common = dict(
            variant=args.variant,
            batch=args.batch,
            seq=args.seq,
            hidden=hidden,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            transpose_key=args.transpose_key,
        )

        # Warmup (kernel compile + caching), not measured.
        for _ in range(args.warmup):
            qr, kr, vr = run_patch(linear_out, **common)
            for t in (qr, kr, vr):
                ttnn.deallocate(t)
        ttnn.synchronize_device(device)

        # Measured region. The signposts bracket the iterations so the patch is easy
        # to isolate in the Tracy timeline / ops_perf csv.
        signpost("split_qkv_patch_start")
        t0 = time.perf_counter()
        for _ in range(args.iters):
            qr, kr, vr = run_patch(linear_out, **common)
            for t in (qr, kr, vr):
                ttnn.deallocate(t)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        signpost("split_qkv_patch_end")

        per_iter_us = (t1 - t0) / args.iters * 1e6
        print(f"[result] {args.iters} iters, host wall time: {per_iter_us:.2f} us/iter")

        if args.check:
            _check(host_in, linear_out, **common)
    finally:
        ttnn.close_device(device)


def _check(host_in, linear_out, *, variant, batch, seq, hidden, num_heads, num_kv_heads, transpose_key):
    import torch

    qr, kr, vr = run_patch(linear_out, variant=variant, batch=batch, seq=seq, hidden=hidden,
                           num_heads=num_heads, num_kv_heads=num_kv_heads,
                           transpose_key=transpose_key)
    head_dim = hidden // (num_heads + 2 * num_kv_heads)

    x = host_in.reshape(batch, seq, hidden).to(torch.float32)
    tensor = x.reshape(batch, seq, num_heads + 2 * num_kv_heads, head_dim)
    q_flat = tensor[..., :num_heads, :]
    k_flat = tensor[..., num_heads:num_heads + num_kv_heads, :]
    v_flat = tensor[..., num_heads + num_kv_heads:, :]

    def ref_reshape(t, heads):
        # split_qkv permutes to [batch, heads, seq, head_dim], then reshape moves batch to dim 1.
        t = t.permute(0, 2, 1, 3).contiguous()
        return t.reshape(seq, batch, heads, head_dim)

    refs = {
        "query": (ref_reshape(q_flat, num_heads), qr),
        "key": (ref_reshape(k_flat, num_kv_heads), kr),
        "value": (ref_reshape(v_flat, num_kv_heads), vr),
    }
    for name, (ref, dev) in refs.items():
        got = ttnn.to_torch(dev).to(torch.float32)
        print(f"[check] {name} reshape: ref {tuple(ref.shape)} vs got {tuple(got.shape)}, "
              f"max|diff|={(got - ref).abs().max().item():.4g}")
    for t in (qr, kr, vr):
        ttnn.deallocate(t)


if __name__ == "__main__":
    main()
