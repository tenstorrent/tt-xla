# split_qkv patch: general vs decode — perf comparison

Same input (`[16,1792]` bf16, TILE, DRAM interleaved) and same output shapes
(Q `[1,16,12,128]`, K/V `[1,16,1,128]`), value-identical outputs (verified `max|diff|=0`).
Repro: `scripts/ttnn_split_qkv_repro.py --variant {general,decode}`, 200 iters, wormhole_b0.

## TL;DR

| Variant | TTNN op | Device ops / iter | Patch kernel total (median) | vs baseline |
|---------|---------|------------------:|----------------------------:|-------------|
| `general` (baseline) | `split_query_key_value_and_split_heads` + reshapes | 3 | **82,195 ns** | 1.0x |
| `decode` | `experimental.nlp_create_qkv_heads_decode` | 1 | **7,505 ns** | **~11x faster** |

The decode op returns Q/K/V already in the `[1, B, heads, head_dim]` attention layout, so:
- the input reshape `[16,1792] -> [1,1,16,1792]` is a **free view** (last-2-dim tiling unchanged),
  whereas the baseline input reshape `[16,1792] -> [16,1,1792]` re-tiles the 16 rows (~32 µs);
- **no output reshapes** are needed (baseline needs a real ~12 µs query reshape).

## Per-op breakdown (median DEVICE KERNEL DURATION)

### general (run `2026_06_11_13_09_17`)
| Op | Output shape | Cores | Kernel median (ns) |
|----|--------------|------:|-------------------:|
| input reshape `[16,1792]->[16,1,1792]` | `[1,16,1,1792]` | 72 | 32,178 |
| split_qkv (`NlpCreateHeadsDeviceOperation`) | `[16,12,32,128]` | 16 | 37,742 |
| query reshape `[16,12,1,128]->[1,16,12,128]` | `[1,16,12,128]` | 64 | 12,275 |
| key/value reshape | — | — | free view (0 ops) |
| **total** | | | **82,195** |

### decode (run `2026_06_11_13_18_16`)
| Op | Output shape | Cores | Kernel median (ns) |
|----|--------------|------:|-------------------:|
| `NLPCreateQKVHeadsDecodeDeviceOperation` | `[1,16,12,128]` | 16 | 7,505 |
| input reshape `[16,1792]->[1,1,16,1792]` | — | — | free view (0 ops) |
| output reshapes | — | — | none needed |
| **total** | | | **7,505** |

tt-perf-report stacked totals: general reshapes 8,894 µs + split 7,549 µs;
decode single op 1,508 µs (over 200 iters).

## Notes / caveats

- Decode op constraints (satisfied here): input TILE, shape `[1,1,B,hidden]`, B<=32; outputs are
  HEIGHT_SHARDED in L1 (baseline outputs were DRAM interleaved). Shapes + values match; the memory
  layout differs, which is the expected/idiomatic form for decode SDPA downstream.
- This op is the decode-phase specialization (S=1). It is the right replacement for this IR, which is
  a bs64/isl128 decode-style graph (B per device = 16, seq = 1).

## How captured

```bash
export TT_METAL_HOME=/home/mvasiljev/tt-metal
export PATH=/home/mvasiljev/tt-metal/python_env/bin:$PATH
# baseline
python -m tracy -r -v -o perf/split_qkv/initial scripts/ttnn_split_qkv_repro.py --variant general --iters 200 --warmup 10
# decode
python -m tracy -r -v -o perf/split_qkv/decode   scripts/ttnn_split_qkv_repro.py --variant decode  --iters 200 --warmup 10
# reports
tt-perf-report <ops_perf_results.csv> --start-signpost split_qkv_patch_start --end-signpost split_qkv_patch_end
```

Artifacts: `perf/split_qkv/decode/...`, `perf/split_qkv/decode_tt_perf_report.{txt,csv}`,
baseline in `perf/split_qkv/initial/...` and `perf/split_qkv/initial_tt_perf_report.*`.
