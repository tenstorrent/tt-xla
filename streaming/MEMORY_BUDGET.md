# Memory Budget — DeepSeek-V4-Flash Streaming

All numbers are **BF16 (2 bytes/elem)** unless noted. Computed from the
real config in [`weight_loader.load_config_args`](../tests/torch/models/deepseek_v4/weight_loader.py).

## Model dimensions

| Field | Value |
|---|---|
| `n_layers` | 43 |
| `n_routed_experts` | 256 |
| `n_activated_experts` | 6 |
| `n_shared_experts` | 1 |
| `dim` | 4096 |
| `moe_inter_dim` | 2048 |
| `n_heads` | 64 |
| `head_dim` | 512 |
| `q_lora_rank` | 1024 |
| `o_lora_rank` | 1024 |
| `o_groups` | 8 |
| `vocab_size` | 129280 |

## Per-block weight size

### Routed experts (compound `StackedExperts` after `enable_sparse_mlp`)

| Param | Shape | Bytes |
|---|---|---|
| `experts.gate_proj` | `(256, 4096, 2048)` | 4.29 GB |
| `experts.up_proj`   | `(256, 4096, 2048)` | 4.29 GB |
| `experts.down_proj` | `(256, 2048, 4096)` | 4.29 GB |
| **routed total** | | **12.88 GB** |

### Shared expert + router

| Param | Shape | Bytes |
|---|---|---|
| `shared_experts.w1.weight` | `(4096, 2048)` | 16.78 MB |
| `shared_experts.w2.weight` | `(2048, 4096)` | 16.78 MB |
| `shared_experts.w3.weight` | `(4096, 2048)` | 16.78 MB |
| `mlp.router.gate.weight`   | `(256, 4096)` | 2.10 MB |
| **MoE non-routed total** | | **52.44 MB** |

### Attention (MLA)

| Param | Shape | Bytes |
|---|---|---|
| `wq_a.weight` | `(1024, 4096)` | 8.39 MB |
| `wq_b.weight` | `(32768, 1024)` | 67.11 MB |
| `wkv.weight`  | `(512, 4096)` | 4.19 MB |
| `wo_a.weight` | `(8192, 4096)` | 67.11 MB |
| `wo_b.weight` | `(4096, 8192)` | 67.11 MB |
| **attn total** | | **213.91 MB** |

### Norms + HC params (small)

~1 MB / block aggregate (RMSNorm scales, hc mixes).

### Compressor / Indexer (only on layers with `compress_ratio > 0`)

Adds ~50–100 MB when present. Fold into per-block budget conservatively.

### Per-block total (worst case, MoE block with compressor)

**~13.2 GB BF16 / block**

The expert weights dominate (~97 % of per-block cost).

## Whole-model totals

| Block class | Count | Per-block | Total |
|---|---|---|---|
| MoE block | 43 | 13.2 GB | **~568 GB** |

Wait — that's much higher than the ~280 GB referenced elsewhere.
Reconciliation: the `weight_loader` returns BF16 dequant of FP4-packed
expert weights. The on-disk HF cache is the FP4 form (~140 GB). The
BF16-dequant in-memory form roughly doubles to ~280-560 GB depending on
whether shared experts are stored separately or fused, and whether the
loader holds both old (per-expert) and new (stacked) forms during the
sparse-MLP rewrite.

⚠️ **Action item for the implementer**: instrument the existing e2e
loader with `tracemalloc` or `psutil.Process().memory_info().rss` to
capture the actual peak. The numbers here are an upper bound.

## Top-level params

| Param | Shape | Bytes |
|---|---|---|
| `embed.weight`     | `(129280, 4096)` BF16 | 1.06 GB |
| `head.weight`      | `(129280, 4096)` FP32 | 2.12 GB |
| `norm.weight`      | `(4096,)` FP32 | 16 KB |
| `hc_head_fn`       | `(4, 16384)` FP32 | 256 KB |
| `hc_head_base`     | `(4,)` FP32 | 16 B |
| `hc_head_scale`    | `(1,)` FP32 | 4 B |
| **top-level total** | | **~3.18 GB** |

Manageable to load all-at-once before the streaming block loop.

## Per-device after SPMD sharding (4×8 mesh = 32 devices)

| Param | Sharding | Per-device |
|---|---|---|
| `experts.{gate,up,down}_proj` | compound `(_axis_0, _axis_1)` 32-way | ~134 MB ea |
| `attn.wq_b.weight` | `(_axis_1, None)` 8-way | 8.4 MB |
| `attn.wo_a.weight` | `(_axis_1, None)` 8-way | 8.4 MB |
| `attn.wo_b.weight` | `(None, _axis_1)` 8-way | 8.4 MB |
| `embed.weight` | replicated | 1.06 GB ea |
| `head.weight` | replicated | 2.12 GB ea |

**Per-device per-block: ~420 MB** (mostly compound experts).
**Per-device total: ~18 GB** (43 blocks + top-level).

Each TT device has ~12 GB DRAM. Need bfp4/bfp8 quantization to fit (the
existing test does this via `apply_weight_dtype_overrides`).

After bfp4 expert + bfp8 attention:
- `experts.{gate,up,down}_proj`: bfp4 ≈ 0.5 byte/elem ÷ 4 vs BF16's 2 byte/elem → **¼ of BF16**, ~33 MB per device per block
- `attn.*.weight`: bfp8 ≈ 1 byte/elem ÷ 2 vs BF16 → **½**, ~4 MB per device per block

**Per-device total post-quant: ~5 GB**. Fits.

## Peak host RAM target for streaming

| Phase | Held on host |
|---|---|
| Top-level load | ~3.2 GB transient |
| Per-block streaming | ~13.2 GB transient (one block) |
| Sparse-MLP rewrite | another ~13 GB temporarily (StackedExperts construction stacks 256 originals — see verification TODO below) |
| Worst-case overlap | ~26 GB transient + ~3 GB top-level + Python ≈ **~30 GB peak** |

vs ~280 GB for the all-at-once loader.

⚠️ **Verification TODO**: confirm `enable_sparse_mlp` doesn't double-hold
the stacked + per-expert tensors. If it does, peak = 2 × 13 GB. If it
streams the stack and frees originals, peak = 13 GB.

## Tracing memory (recipe)

In `streaming_loader.py` add:

```python
import psutil, os
def rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e9

# at every phase boundary:
print(f"[mem] after block {layer_id}: {rss_gb():.2f} GB", flush=True)
```

If a phase exceeds expectation, that's the next bottleneck to attack.
