# GLM-4 decode — optimization log

Running log of perf optimizations we apply to the GLM-4 decode attention path, with the
code change, correctness (PCC) and the measured perf delta for each. Newest entries on top.

Conventions:
- All numbers captured on `wormhole_b0`, single device, via Tracy + `tt-perf-report`.
- "Device kernel" = sum of per-op DEVICE KERNEL DURATION medians for the patch region
  (host→device gaps excluded; they vanish under program-cache/tracing).
- PCC measured against a torch reference (or against the baseline variant) over the same input.

---

## #3 — compiler pass: fuse split_qkv → `nlp_create_qkv_heads_decode` in tt-mlir (TRIED, REVERTED)

**Status:** fires + numerically correct, but **regresses GLM-4 e2e perf** → reverted.
**tt-mlir branch:** `mvasiljevic/glm-hidden-replicated-sharding`
- fix commit `81706d57f` ("Shape-driven decode QKV fusion through RMSNorm + partial RoPE")
- revert commit `7c70dfb2f` on top (so the active build does **not** fire it; history keeps it).
**Pattern:** `NLPCreateQKVHeadsDecodeFusing` in
`lib/Dialect/TTNN/Transforms/Fusing/SplitQKVFusingPatterns.cpp`.

### What we tried

Push entries #1/#2 (the Python wins) down into the compiler so any decode model gets them.
A shape-driven matcher (`matchDecodeChain`, **no model-specific constants**) recognizes two
shapes of the split-output → decode-layout-change chain and rewrites the generic
`split_query_key_value_and_split_heads` into `nlp_create_qkv_heads_decode`:

- **Direct:** `split [B,H,1,D]` → decode layout change `[B,H,1,D]→[1,B,H,D]`
  (matches both the explicit `permute [2,0,1,3]` and the equivalent `reshape`).
- **NormRope:** `split` → `rms_norm` → partial RoPE (`slice`/`rotary_embedding`/`slice`/`concat`)
  → decode layout change. `rebuildDecodeChain` reconstructs the RMSNorm + partial-RoPE in the
  `[1,B,H,D]` decode layout after the fused op.

It fires on GLM (4 `nlp_create_qkv_heads_decode`, 0 `split_query_key_value` in the decode
graphs g1/g5) and PCC passes (decode 0.861, prefill 0.936). **Prefill is byte-identical** to the
pre-change baseline (verified by normalized IR diff of g0/g2/g3/g4 vs run `9f5f`); only the
decode graphs change.

### Why it regresses on GLM (the lesson)

| Config | sps | decode PCC | prefill PCC |
|--------|----:|-----------:|------------:|
| opt1, baseline (no fusion) | ~7 (one-off 10) | — | — |
| opt1, **with fusion** | **6.07** (range 5.5–6.1) | 0.861 | 0.936 |

The isolated Python repro (#1/#2) was faster because the baseline there did a **real**
`[16,1792]→[16,1,1792]` input reshape and a **real** Q re-pack `[16,12,1,128]→[1,16,12,128]`,
both of which the decode op removes. **In GLM at opt_level=1 those reshapes are S=1 free
relabels**, so the fusion removes ~nothing — while `nlp_create_qkv_heads_decode` *forces*
`HEIGHT_SHARDED` output and GLM's interleaved norm/RoPE path then pays for new
`sharded→interleaved` `to_memory_config` conversions. Net: added cost, ~no saving → regression.

Takeaway: the win in #1/#2 came from the **tile-fill / layout** change (`[1,B,heads,hd]` packs
`heads` 12→32 over B=16 blocks instead of padding seq 1→32 over B*heads=192 blocks), **not** from
the op swap itself. The op swap only helps when its forced sharded layout is also what downstream
wants. **Next:** chase the tile-fill win directly, on opt_level=1, independent of the decode op.

> Aside: an opt_level=2 experiment for GLM (vs the pinned opt_level=1) was started to see if the
> layout optimizer makes the sharded↔interleaved round-trips free, but was parked before
> completing.

---

## #2 — full SDPA block: fold the decode split + decode-layout RMSNorm/RoPE into the block

**Status:** validated (PCC 0.999 vs a torch SDPA reference), **~2.9x less total device kernel time**.
**Repro:** `scripts/ttnn_sdpa_block_repro.py --variant {baseline,decode}`
**Reports:** `perf/sdpa_block/{baseline,decode}_tt_perf_report.txt`
**Block boundary (fixed):** in `[16,1792]` fused-QKV bf16 → out `[16,1536]` bf16. See
`perf/sdpa_block/baseline_perf_summary.md`.

### The code change

This generalizes entry #1 from the isolated split to the whole attention block. The decode op
emits Q/K/V already in the `[1, B, heads, head_dim]` decode layout, so we keep RMSNorm + partial
RoPE in that layout instead of the IR's `[B, heads, 1, head_dim]` layout.

```python
# baseline front half: input reshape [16,1792]->[16,1,1792] (REAL), generic split,
# RMSNorm + partial RoPE in [B,heads,1,hd], then a REAL q re-pack [16,12,1,128]->[1,16,12,128].

# decode front half:
x = ttnn.reshape(inp["qkv"], (1, 1, batch, hidden))          # FREE view
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(x, num_heads=12, num_kv_heads=1)
q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)        # un-shard for norm (layernorm rejects HEIGHT_SHARDED)
k = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
q = ttnn.rms_norm(q, weight=inp["wq"], ...)                  # in [1,B,heads,hd]
q = _partial_rope_decode(q, cos_rep, sin_rep, ...)          # in [1,B,heads,hd]
k = ttnn.rms_norm(k, weight=inp["wk"], ...)
k = _partial_rope_decode(k, cos_rep, sin_rep, ...)
# v stays height-sharded -> feeds paged_update_cache directly (no reshape, no extra shard);
# q is already [1,B,heads,hd] -> feeds SDPA directly (no q re-pack reshape).
```

**The RoPE subtlety (and the fix).** In the decode layout dim 2 = `heads` is exactly the axis
`ttnn.experimental.rotary_embedding` treats as its "sequence". The IR cos/sin are `[1,1,1,64]`
(one position), so naively the op only rotates head 0 correctly (PCC drops to ~0.41–0.62).
GLM applies the **same** single-position rotation to every head, so the fix is to feed cos/sin
**repeated across the heads axis** (tile-padded to 32 → `[1,1,32,64]`). These are constants
(computed once per decode step, shared across layers), so we build them host-side in
`make_inputs` — **no extra op in the measured loop**. With that, decode-layout RoPE is
bit-identical to the IR layout (isolated PCC 1.0).

### Correctness (PCC)

Validated against a pure-torch decode-SDPA reference built from the exact q / kv-cache tensors
fed to the device op (`_torch_sdpa_ref`, `--variant decode --check`):

| Variant | vs torch SDPA reference | max\|diff\| |
|---------|------------------------:|-----------:|
| `decode` | **PCC 0.998709** | 0.088 (bf16-level) |
| `baseline` | PCC 0.745748 | 3.089 |

Note: the **decode** path is the one that matches the math reference. The `baseline` repro's
SDPA is numerically off because it approximates the IR's exact `to_memory_config` for the SDPA
query (the decode path produces the layout SDPA actually wants, natively). This is a
repro-fidelity detail and does **not** affect op timings — the op sequence was already validated
against the real 32-device IR report (`attention_report.txt`), and both variants run the same
SDPA/cache ops.

### Perf (Tracy, 30 iters + 10 warmup, sum of DEVICE KERNEL DURATION over 30 iters)

| Op | Baseline (30 iters) | Decode (30 iters) | Δ |
|----|--------------------:|------------------:|---|
| RotaryEmbedding | 1870.95 μs | 474.00 μs | **~4.0x** |
| ReshapeView | 1774.73 μs (3/iter) | 438.02 μs (1/iter) | **~4.1x** (only output reshape left) |
| LayerNorm (rms_norm) | 1239.53 μs | 467.42 μs | **~2.7x** |
| Slice | 1204.56 μs | 326.88 μs | **~3.7x** |
| split_qkv | 1132.49 μs (NlpCreateHeads) | 224.56 μs (NlpCreateHeadsDecode) | **~5.0x** |
| Concat | 520.47 μs | 130.41 μs | **~4.0x** |
| SdpaDecode | 472.57 μs | 474.11 μs | ~1.0x |
| PagedUpdateCache | 240.27 μs | 240.37 μs | ~1.0x |
| InterleavedToSharded | 78.34 μs (2/iter) | 39.00 μs (1/iter) | **~2.0x** (v stays sharded) |
| ShardedToInterleaved | — | 105.61 μs (new, q+k un-shard) | added |
| **Total device kernel** | **8533.9 μs (≈285 μs/iter)** | **2920.4 μs (≈97 μs/iter)** | **~2.9x** |

### Why the whole block (not just the reshapes) got faster

The big surprise: it isn't only the removed reshapes. The IR runs the per-head ops in
`[B, heads, 1, hd]` where the seq=1 axis is tile-padded 1→32, wasting 31/32 of every tile across
all `B*heads` = 192 tile-blocks. The decode layout `[1, B, heads, hd]` instead pads `heads`
12→32 over just `B` = 16 tile-blocks, so RotaryEmbedding / RMSNorm / Slice / Concat each process
far fewer padded tiles → 3–4x cheaper. Removing the input reshape + q re-pack (3 reshapes → 1)
and keeping V height-sharded (no v reshape / one fewer shard) are on top of that.

Cost added: 2 `ShardedToInterleaved` (q,k un-shard so layernorm accepts them) ≈ 106 μs/30 iters —
small relative to the savings. A follow-up could try a sharded-input RMSNorm program config to
drop even those.

---

## #1 — split_qkv: `split_query_key_value_and_split_heads` → `nlp_create_qkv_heads_decode`

**Status:** validated (PCC 1.0), ~11x faster on the patch.
**Repro:** `scripts/ttnn_split_qkv_repro.py --variant {general,decode}`
**Detail doc:** `perf/split_qkv/comparison_general_vs_decode.md`

### The code change

Baseline (`general`) — input reshape + generic split + a reshape on each output:

```python
x = ttnn.reshape(linear_out, (batch, seq, hidden))            # [16,1792] -> [16,1,1792]  (REAL reshape, re-tiles 16 rows)
q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
    x, num_heads=12, num_kv_heads=1, transpose_key=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
q = ttnn.reshape(q, (seq, batch, num_heads, head_dim))         # [16,12,1,128] -> [1,16,12,128]  (REAL reshape)
k = ttnn.reshape(k, (seq, batch, num_kv_heads, head_dim))      # free view
v = ttnn.reshape(v, (seq, batch, num_kv_heads, head_dim))      # free view
```

Optimized (`decode`) — decode-specialized op that emits Q/K/V already in attention layout:

```python
x = ttnn.reshape(linear_out, (1, seq, batch, hidden))          # [16,1792] -> [1,1,16,1792]  (FREE view)
q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
    x, num_heads=12, num_kv_heads=1)                           # outputs already [1,B,heads,head_dim], height-sharded
# no output reshapes needed
```

Why it's faster: the decode op returns Q/K/V directly in the `[1, B, heads, head_dim]`
layout that the baseline only reaches *after* an expensive query reshape. The input
reshape becomes a free view, and all output reshapes disappear.

### Correctness (PCC)

Same input `[16,1792]` bf16/TILE/DRAM, identical output shapes (Q `[1,16,12,128]`, K/V `[1,16,1,128]`):

| Output | max\|diff\| | PCC |
|--------|-----------:|----:|
| query | 0 | 1.000000 |
| key   | 0 | 1.000000 |
| value | 0 | 1.000000 |

Value-identical. (Memory layout differs — decode outputs are L1 height-sharded vs DRAM
interleaved — which is the idiomatic form for the downstream decode SDPA.)

### Perf (200 iters, median DEVICE KERNEL DURATION)

| Variant | Op(s) | Device ops/iter | Patch kernel total | vs baseline |
|---------|-------|----------------:|-------------------:|-------------|
| `general` (baseline) | input reshape + split_qkv + q reshape | 3 | **82,195 ns** | 1.0x |
| `decode` | `nlp_create_qkv_heads_decode` | 1 | **7,505 ns** | **~11x faster** |

Baseline breakdown: input reshape 32,178 ns + split_qkv 37,742 ns + query reshape 12,275 ns
(k/v reshapes are free). Decode: single op 7,505 ns; input reshape free; no output reshapes.

### Applicability to the full SDPA block

This is the `NlpCreateHeads`/reshape portion of the attention block (~13% split + part of the
~21% reshape time in `perf/sdpa_block/baseline_perf_summary.md`). Folding this into the block
repro is a candidate next step, but the decode op emits height-sharded outputs, so the RMSNorm
+ RoPE that follow would need to consume that layout — to be evaluated in a block-level entry.
