# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Row-invariance of tt's prefill SDPA kernel, called directly.

Narrows the Falcon3 batched-prefill defect to a single op. Every batch row is
given *identical* q/k/v, so a row-independent kernel must return identical rows.
If row 0 differs from rows 1..B-1, the row index is entering the arithmetic.

This is the shape the earlier op-level probe missed: prefill attention has
q_len > 1 and is causal, whereas that probe only covered decode (q_len == 1).

Layout follows attention_impls/attention.py: the op is handed
[users, n_heads, tokens, head_dim].

Usage:
  python sdpa_row_invariance.py                      # causal, B=8
  python sdpa_row_invariance.py --batch 32 --seq 128
  python sdpa_row_invariance.py --mask               # explicit mask instead
"""

import argparse

import torch
import torch_xla
import torch_xla.core.xla_model as xm

# Falcon3-7B-Instruct attention geometry.
N_HEADS = 12
N_KV_HEADS = 4
HEAD_DIM = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--heads", type=int, default=N_HEADS)
    ap.add_argument("--kv-heads", type=int, default=N_KV_HEADS)
    ap.add_argument("--head-dim", type=int, default=HEAD_DIM)
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--no-causal", dest="causal", action="store_false")
    ap.add_argument("--mask", action="store_true", help="pass an explicit attn_mask")
    args = ap.parse_args()

    dev = xm.xla_device()
    torch.manual_seed(0)

    # One row's worth of data, then replicated across the batch: every row is
    # mathematically identical.
    q1 = (torch.randn(1, args.heads, args.seq, args.head_dim) * 0.05).to(torch.bfloat16)
    k1 = (torch.randn(1, args.kv_heads, args.seq, args.head_dim) * 0.05).to(
        torch.bfloat16
    )
    v1 = (torch.randn(1, args.kv_heads, args.seq, args.head_dim) * 0.05).to(
        torch.bfloat16
    )

    def run(b):
        q = q1.repeat(b, 1, 1, 1).to(dev)
        k = k1.repeat(b, 1, 1, 1).to(dev)
        v = v1.repeat(b, 1, 1, 1).to(dev)
        kwargs = {"scale": 1.0 / (args.head_dim**0.5)}
        if args.mask:
            m = torch.full((args.seq, args.seq), float("-inf"))
            m = torch.triu(m, diagonal=1)
            kwargs["attn_mask"] = (
                m.to(torch.bfloat16).expand(b, 1, args.seq, args.seq).to(dev)
            )
            kwargs["is_causal"] = False
        else:
            kwargs["is_causal"] = args.causal
        out = torch.ops.tt.scaled_dot_product_attention(q, k, v, **kwargs)
        torch_xla.sync()
        return out.to("cpu").float()

    print(
        f"batch={args.batch} seq={args.seq} heads={args.heads} "
        f"kv_heads={args.kv_heads} head_dim={args.head_dim} "
        f"causal={args.causal} explicit_mask={args.mask}"
    )

    single = run(1)
    batched = run(args.batch)

    # Rows within the batched call, compared to row 0 of that same call.
    print(f"\n  {'row':>4} {'max|row - row0|':>18} {'== row0':>9} {'== B=1 run':>12}")
    ref0 = batched[0]
    b1 = single[0]
    distinct = set()
    for i in range(args.batch):
        d = (batched[i] - ref0).abs().max().item()
        eq0 = torch.equal(batched[i], ref0)
        eqs = torch.equal(batched[i], b1)
        distinct.add(round(d, 12))
        if i < 10 or not eq0:
            print(f"  {i:>4} {d:>18.8g} {str(eq0):>9} {str(eqs):>12}")

    n_diff = sum(1 for i in range(args.batch) if not torch.equal(batched[i], ref0))
    print()
    print(f"  rows differing from row 0 : {n_diff}/{args.batch}")
    print(f"  row 0 == single-row run   : {torch.equal(ref0, b1)}")
    if n_diff:
        print("  => ROW-DEPENDENT: identical inputs give different rows")
    else:
        print("  => row-invariant at this shape")


if __name__ == "__main__":
    main()
