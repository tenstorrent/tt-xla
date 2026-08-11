# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Op-level batch-invariance probe -- no vLLM, no server, no KV cache.

Batch invariance is the property that a row's result does not depend on which
other rows share its batch. It is mathematically guaranteed for every op here
(each output row reads only its own input row), so any difference is an
implementation defect, not a numerics tolerance question.

Two independent perturbations, because they fail for different reasons:

  WIDTH   row 0 alone (B=1) vs row 0 inside a wider batch (B=n). Sensitive to
          batch-dependent tiling, reduction order, or kernel/config selection.
  CONTENT same B, row 0 identical, *other* rows' data changed. Nothing legal
          can move row 0. A hit here means cross-row contamination -- values
          from a neighbouring row reaching row 0's arithmetic.

Reported as exact-match plus max |delta| in ULPs of bf16 at row 0's scale, so a
last-bit rounding difference is distinguishable from a real corruption.

Usage:  python batch_invariance_probe.py [--hidden 3072] [--reps 2]
"""

import argparse
import functools

import jax
import jax.numpy as jnp
import numpy as np

# Falcon3-7B-Instruct geometry, so a hit here maps onto the model directly.
HIDDEN = 3072
INTERMEDIATE = 23040
N_HEADS = 12
N_KV_HEADS = 4
HEAD_DIM = 256
VOCAB = 131072


def bf16_ulp(x):
    """Size of one bf16 mantissa step at magnitude x (8 mantissa bits)."""
    m = float(np.max(np.abs(np.asarray(x, dtype=np.float32))))
    if m == 0.0:
        return 1.0
    return 2.0 ** (np.floor(np.log2(m)) - 8)


def report(name, kind, ref, got):
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    delta = float(np.max(np.abs(ref - got)))
    ulps = delta / bf16_ulp(ref)
    n_diff = int(np.sum(ref != got))
    verdict = "OK" if delta == 0.0 else "VARIANT"
    print(
        f"  {name:<26} {kind:<8} {verdict:<8} "
        f"maxdelta={delta:<12.6g} ulps={ulps:<9.2f} "
        f"elems_differing={n_diff}/{ref.size}"
    )
    return delta == 0.0


# --- ops under test. Each takes a [B, ...] batch and is row-independent. ---


def op_linear(x, w):
    return x @ w


def op_rmsnorm(x, w):
    v = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(v + 1e-6).astype(x.dtype)) * w


def op_mlp(x, wg, wu, wd):
    return (jax.nn.silu(x @ wg) * (x @ wu)) @ wd


def op_sdpa_decode(q, k, v):
    """One decode step: q is a single position, k/v are the cache."""
    q = q.astype(jnp.float32)
    # GQA: repeat kv heads to match q heads.
    rep = q.shape[1] // k.shape[1]
    kk = jnp.repeat(k.astype(jnp.float32), rep, axis=1)
    vv = jnp.repeat(v.astype(jnp.float32), rep, axis=1)
    s = jnp.einsum("bhqd,bhkd->bhqk", q, kk) / np.sqrt(q.shape[-1])
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), vv)


def op_sdpa_ragged(q, k, v, mask):
    """Decode attention over a batch padded to the running max length.

    This is what a real serving batch looks like: each row has its own history
    length, all rows are padded to the longest, and the surplus is masked. Row 0
    is mathematically unaffected by how long its neighbours are -- the masked
    terms contribute exactly zero -- so any movement here is a defect that a
    same-length batch cannot expose.
    """
    q = q.astype(jnp.float32)
    rep = q.shape[1] // k.shape[1]
    kk = jnp.repeat(k.astype(jnp.float32), rep, axis=1)
    vv = jnp.repeat(v.astype(jnp.float32), rep, axis=1)
    s = jnp.einsum("bhqd,bhkd->bhqk", q, kk) / np.sqrt(q.shape[-1])
    s = jnp.where(mask[:, None, None, :], s, jnp.float32(-1e30))
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), vv)


def op_sdpa_paged(q, k_pool, v_pool, block_table):
    """Decode attention reading KV out of a shared paged pool.

    block_table[b] lists the physical blocks holding row b's history. Row 0's
    result must not depend on *which* physical blocks it was handed, nor on what
    is stored in blocks belonging to other rows.
    """
    q = q.astype(jnp.float32)
    # Gather this row's blocks out of the pool and flatten to a sequence.
    kb = jnp.take(k_pool, block_table, axis=0)  # [B, blocks, bs, kvh, d]
    vb = jnp.take(v_pool, block_table, axis=0)
    b, nb, bs, kvh, d = kb.shape
    kk = kb.transpose(0, 3, 1, 2, 4).reshape(b, kvh, nb * bs, d).astype(jnp.float32)
    vv = vb.transpose(0, 3, 1, 2, 4).reshape(b, kvh, nb * bs, d).astype(jnp.float32)
    rep = q.shape[1] // kvh
    kk = jnp.repeat(kk, rep, axis=1)
    vv = jnp.repeat(vv, rep, axis=1)
    s = jnp.einsum("bhqd,bhkd->bhqk", q, kk) / np.sqrt(q.shape[-1])
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1), vv)


def build_cases(rng, batch, seq, dtype):
    """Return {name: (fn, batched_args, static_args)} for one batch size."""

    def r(*shape):
        return jnp.asarray(rng.standard_normal(shape) * 0.05, dtype=dtype)

    w = r(HIDDEN, HIDDEN)
    nw = r(HIDDEN)
    wg, wu = r(HIDDEN, INTERMEDIATE), r(HIDDEN, INTERMEDIATE)
    wd = r(INTERMEDIATE, HIDDEN)

    return {
        # batched args are perturbed; static args are shared across batch sizes
        "linear": (op_linear, [r(batch, HIDDEN)], [w]),
        "rmsnorm": (op_rmsnorm, [r(batch, HIDDEN)], [nw]),
        "mlp": (op_mlp, [r(batch, HIDDEN)], [wg, wu, wd]),
        "sdpa_decode": (
            op_sdpa_decode,
            [
                r(batch, N_HEADS, 1, HEAD_DIM),
                r(batch, N_KV_HEADS, seq, HEAD_DIM),
                r(batch, N_KV_HEADS, seq, HEAD_DIM),
            ],
            [],
        ),
    }


def run(fn, batched, static, device):
    args = [jax.device_put(a, device) for a in batched + static]
    out = jax.jit(fn)(*args)
    return np.asarray(jax.device_get(out))


def probe(device, batch_levels, seq, dtype, reps, rng_seed):
    all_ok = True
    for name in ("linear", "rmsnorm", "mlp", "sdpa_decode"):
        print(f"\n{name}  (dtype={dtype.__name__}, seq={seq})")
        rng = np.random.default_rng(rng_seed)
        big = max(batch_levels)
        cases = build_cases(rng, big, seq, dtype)
        fn, batched, static = cases[name]
        row0 = [a[:1] for a in batched]

        # Reference: row 0 computed alone.
        ref = run(fn, row0, static, device)

        for _ in range(reps):
            # WIDTH: row 0 placed at index 0 of a wider batch.
            for b in batch_levels:
                if b == 1:
                    continue
                wide = [
                    jnp.concatenate([r0, a[1:b]], axis=0)
                    for r0, a in zip(row0, batched)
                ]
                got = run(fn, wide, static, device)[:1]
                all_ok &= report(f"{name} B=1 vs B={b}", "width", ref, got)

            # CONTENT: same width, only the *other* rows change.
            b = big
            alt_rng = np.random.default_rng(rng_seed + 999)
            alt = build_cases(alt_rng, b, seq, dtype)[name][1]
            a_batch = [
                jnp.concatenate([r0, a[1:b]], axis=0) for r0, a in zip(row0, batched)
            ]
            b_batch = [
                jnp.concatenate([r0, a[1:b]], axis=0) for r0, a in zip(row0, alt)
            ]
            ga = run(fn, a_batch, static, device)[:1]
            gb = run(fn, b_batch, static, device)[:1]
            all_ok &= report(f"{name} neighbours@B={b}", "content", ga, gb)
    return all_ok


def probe_head(device, batch_levels, dtype, rng_seed):
    """LM head at true vocab width, then greedy argmax on top of it.

    Two things the smaller ops cannot expose: a reduction wide enough that
    batch-dependent K-splitting would change summation order, and the fact that
    greedy decoding consumes an *index*, so an arbitrarily small logit shift near
    a tie flips a token outright.
    """
    print(f"\nlm_head + argmax  (vocab={VOCAB}, dtype={dtype.__name__})")
    rng = np.random.default_rng(rng_seed)
    big = max(batch_levels)
    x = jnp.asarray(rng.standard_normal((big, HIDDEN)) * 0.05, dtype)
    w = jnp.asarray(rng.standard_normal((HIDDEN, VOCAB)) * 0.02, dtype)
    row0 = x[:1]

    def logits_and_pick(xx):
        out = run(op_linear, [xx], [w], device)
        return out, np.argmax(out.astype(np.float32), axis=-1)

    ref_logits, ref_pick = logits_and_pick(row0)

    for b in batch_levels:
        if b == 1:
            continue
        wide = jnp.concatenate([row0, x[1:b]], axis=0)
        got, pick = logits_and_pick(wide)
        report(f"lm_head B=1 vs B={b}", "width", ref_logits, got[:1])
        same = int(ref_pick[0]) == int(pick[0])
        print(
            f"  {'argmax B=1 vs B=' + str(b):<26} {'width':<8} "
            f"{'OK' if same else 'TOKEN-FLIP':<8} "
            f"token {int(ref_pick[0])} -> {int(pick[0])}"
        )

    # Content: fixed width, neighbours' activations replaced.
    alt = jnp.asarray(
        np.random.default_rng(rng_seed + 999).standard_normal((big, HIDDEN)) * 0.05,
        dtype,
    )
    a = jnp.concatenate([row0, x[1:big]], axis=0)
    b_ = jnp.concatenate([row0, alt[1:big]], axis=0)
    la, pa = logits_and_pick(a)
    lb, pb = logits_and_pick(b_)
    report(f"lm_head neighbours@B={big}", "content", la[:1], lb[:1])
    print(
        f"  {'argmax neighbours':<26} {'content':<8} "
        f"{'OK' if int(pa[0]) == int(pb[0]) else 'TOKEN-FLIP':<8} "
        f"token {int(pa[0])} -> {int(pb[0])}"
    )


def probe_ragged(device, batch, pad, dtype, rng_seed):
    """Row 0's history is fixed; only the neighbours' lengths change.

    Padded shape is held constant so both runs compile the same graph -- the only
    difference is mask content for rows 1..B-1.
    """
    print(f"\nsdpa_ragged  (B={batch}, padded_seq={pad}, dtype={dtype.__name__})")
    rng = np.random.default_rng(rng_seed)
    q = jnp.asarray(rng.standard_normal((batch, N_HEADS, 1, HEAD_DIM)) * 0.05, dtype)
    k = jnp.asarray(
        rng.standard_normal((batch, N_KV_HEADS, pad, HEAD_DIM)) * 0.05, dtype
    )
    v = jnp.asarray(
        rng.standard_normal((batch, N_KV_HEADS, pad, HEAD_DIM)) * 0.05, dtype
    )
    row0_len = 100

    def mask_for(neighbour_len):
        lens = np.full(batch, neighbour_len, dtype=np.int32)
        lens[0] = row0_len
        return jnp.asarray(np.arange(pad)[None, :] < lens[:, None], dtype=bool)

    ref = None
    for nlen in (row0_len, 200, 400, pad):
        got = run(op_sdpa_ragged, [q, k, v, mask_for(nlen)], [], device)[:1]
        if ref is None:
            ref = got
            print(f"  reference: neighbours all len={nlen}")
            continue
        report(f"neighbour_len {row0_len}->{nlen}", "content", ref, got)


def probe_paged(device, batch, dtype, rng_seed, block_size=32, n_blocks=64):
    """Row 0's logical history is fixed; its physical block placement changes."""
    print(f"\nsdpa_paged  (B={batch}, block_size={block_size}, dtype={dtype.__name__})")
    rng = np.random.default_rng(rng_seed)
    shape = (n_blocks, block_size, N_KV_HEADS, HEAD_DIM)
    k_pool = np.asarray(rng.standard_normal(shape) * 0.05, dtype=np.float32)
    v_pool = np.asarray(rng.standard_normal(shape) * 0.05, dtype=np.float32)
    q = jnp.asarray(rng.standard_normal((batch, N_HEADS, 1, HEAD_DIM)) * 0.05, dtype)
    blocks_per_row = 4

    def run_with(table):
        return run(
            op_sdpa_paged,
            [q],
            [
                jnp.asarray(k_pool, dtype),
                jnp.asarray(v_pool, dtype),
                jnp.asarray(table, dtype=jnp.int32),
            ],
            device,
        )[:1]

    # Layout A: row 0 gets blocks 0..3, contiguous.
    tbl_a = np.arange(batch * blocks_per_row).reshape(batch, blocks_per_row)
    ref = run_with(tbl_a)

    # Layout B: row 0's blocks copied to scattered physical slots, same content.
    tbl_b = tbl_a.copy()
    scattered = [61, 5, 40, 17]
    for j, dst in enumerate(scattered):
        k_pool[dst] = k_pool[tbl_a[0, j]]
        v_pool[dst] = v_pool[tbl_a[0, j]]
        tbl_b[0, j] = dst
    report("scattered blocks, same content", "layout", ref, run_with(tbl_b))

    # Layout C: row 0 untouched, other rows' blocks repointed elsewhere.
    tbl_c = tbl_a.copy()
    tbl_c[1:] = (tbl_c[1:] + 20) % n_blocks
    report("neighbours' blocks repointed", "content", ref, run_with(tbl_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="1,2,8,16,17,32")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="tt")
    ap.add_argument(
        "--mode", default="all", choices=["all", "dense", "ragged", "paged", "head"]
    )
    args = ap.parse_args()

    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[args.dtype]
    device = jax.devices(args.backend)[0]
    print(f"device: {device}   backend: {args.backend}")
    levels = [int(x) for x in args.batches.split(",")]

    ok = True
    if args.mode in ("all", "dense"):
        ok &= probe(device, levels, args.seq, dtype, args.reps, args.seed)
    if args.mode in ("all", "ragged"):
        probe_ragged(device, max(levels), args.seq, dtype, args.seed)
    if args.mode in ("all", "paged"):
        probe_paged(device, max(levels), dtype, args.seed)
    if args.mode in ("all", "head"):
        probe_head(device, levels, dtype, args.seed)
    print()
    print("=" * 74)
    print("ALL BATCH-INVARIANT" if ok else "BATCH-VARIANT -- see VARIANT rows above")
    print("=" * 74)


if __name__ == "__main__":
    main()
