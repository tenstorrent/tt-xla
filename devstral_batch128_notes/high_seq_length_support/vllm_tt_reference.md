# vLLM TT Plugin — Reference (chunked prefill / high-context, Devstral-2-123B BH galaxy)

> Compiled by the exploration agent 2026-07-16. Line numbers from the current working tree (incl. uncommitted edits).
> Package: `integrations/vllm_plugin/vllm_tt/`.

## Directory inventory — `integrations/vllm_plugin/vllm_tt/`
| File | Role |
|---|---|
| `__init__.py` | Plugin entry. Registers TT attention backends; `register()`→`vllm_tt.platform.TTPlatform`; forces spawn; monkeypatches `Attention`→`TTAttention`, loads `TTFusedMoE`. |
| `platform.py` (619) | `TTConfig` (all `additional_config` knobs) + `TTPlatform`. `check_and_update_config()` (L336): installs fp8 hook, resolves prefill knobs, selects `AscendScheduler`, forces `block_size=32`, enables chunked prefill from `prefill_chunk_size`. MLA path (L502) DISABLES chunked prefill. |
| `worker.py` (417) | `TTWorker`: SPMD setup, `init_device()`, `determine_available_memory()` sizes KV from PJRT `dram_size_bytes`×`gpu_memory_utilization`. |
| `model_runner.py` (4086) | `TTModelRunner` — mesh construction, chunked-prefill budgeting, `_chunked_sdpa_active` gating, KV spec, `_prepare_inputs`/`_pin_input_shardings`, `capture_model`, `execute_model`. The heart. |
| `vllm_distributed_utils.py` (454) | SPMD sharding: `safe_mark_sharding`, `ParallelismMode`, per-layer partition fns, `MODULE_TYPE_TO_WRAPPING_FUNC`, `shard_model()`. **Has embedding fix.** |
| `fp8_dequant.py` (263) | `TTFp8DequantLinearMethod` — fp8→bf16 at load. **Has version-skew fix.** |
| `metadata.py` (416) | `XLASupportedSamplingMetadata` — fixed-shape CPU sampling tensors. |
| `vllm_utils.py` (164) | `determine_mesh_shape()`, `apply_hidden_layer_override()` (the `num_hidden_layers` knob), `prev_power_of_2`. |
| `input_batch.py` (798) | `InputBatch` persistent state; chunked prefill advances `num_computed_tokens`. |
| `sampler.py` (418) | On-device sampler; bypassed via `cpu_sampling=True` (#4387/#4440). |
| `scheduler/ascend_scheduler.py` | `AscendScheduler` — prefill-first; chunk admission, same-stage batching, b1-prefill cap, KV watermark. |
| `attention_impls/attention.py` (904) | `TTAttentionBackend`, `TTMetadata`, `TTAttentionBackendImpl` (prefill/chunked-SDPA/paged decode), KV write helpers. |
| `attention_impls/attention_mla.py` | MLA — OFF-PATH (Devstral is dense MHA/GQA). |
| `layers/fused_moe.py`, `layers/{mm_embeddings,mrope,multimodal_attention}.py` | OFF-PATH (Devstral dense, text-only). |
| `layers/{rmsnorm,rotary_embedding}.py` | TT RMSNorm / RoPE. |
| `pooling_runner.py`, `overrides.py`, `logger.py` | Pooling runner (off-path), override shims, logger. |

## Chunked-prefill flow
`AscendScheduler.schedule` → `platform.check_and_update_config` → `TTModelRunner` (budget + `_chunked_sdpa_active` + `chunk_start_idx`) → `TTAttentionBackendImpl.chunked_scaled_dot_product_attention`.

- **Config (platform.py:336):** installs fp8 hook (346); `block_size=32` forced (478); chunked opt-in (513-545): `prefill_chunk_size>0` ⇒ `per_seq_chunk=max(chunk_size, block_size)`, budget=`per_seq_chunk*max_num_seqs`, sets `enable_chunked_prefill`, `tt_chunked_prefill_enabled`, `tt_prefill_chunk_size`.
- **Scheduler (ascend_scheduler.py:66):** prefill-first, one-at-a-time; b1-prefill cap (104-142) routes small batches to batch-1 graph; same-stage batching (233-241) only batches prefills sharing `num_computed_tokens`.
- **Runner budgeting (model_runner.py:260-465):** mesh (289-324) `("batch","model")`, `dp_size=mesh_shape[0]`, `use_2d_mesh = 1 not in mesh_shape`. KV dtype (357-367): `bfp_bf8`⇒spec `uint8` accounting. **`_chunked_sdpa_active` (432-434)** = `prefill_chunk_budget < max_model_len AND max_num_blocks_per_req % 8 == 0`. Hard alignment (439-449): chunking on requires `max_num_blocks_per_req % 8 == 0` ⇒ **`max_model_len % 256 == 0`** (256 = 8×block_size 32).
- **Inputs (_prepare_inputs ~1340-1500):** page_table + fill_page_table; `safe_mark_sharding(page_table, mesh, ("batch", None))` (1455). **`chunk_start_idx`** (1461-1495) set only for a cached-prefix chunk (`num_computed>0` AND `_chunked_sdpa_active`); decode & first-chunk keep it None.
- **Attention (attention.py):** `TTMetadata` (177) carries `chunk_start_idx`/`fill_page_table`/`dp_size`/`batch_idx`. Prefill KV write `_handle_paged_attention` (461): `paged_fill_cache` with `batch_idx % local_batch` rebase when dp_size>1 (500). Cached-prefix chunked `_compute_full_attention` (517-560): `chunked_scaled_dot_product_attention(q, k_cache, v_cache, page_table, chunk_start_idx, scale)`. Decode: `paged_scaled_dot_product_attention_decode` (658). Precompile `_precompile_backbone` builds `prefix_chunk ∈ [False, True]` when `_chunked_sdpa_active` (2563-2592).

## Uncommitted fixes (detail)
- **fp8_dequant.py:** `__init__` (87-94) sets `activation_quant_key`/`weight_quant_key`/`input_dtype` placeholders; `create_weights()` (103-120) no-ops `init_fp8_linear_kernel` around `super().create_weights()` (vLLM≥0.20 moved it there; KeyError('OOT') on OOT platform).
- **vllm_distributed_utils.py:** `partition_vocab_parallel_embedding` (339-357) forward-hook `(None,None,None)`→`("batch",None,None)`. Kills the per-forward DP-axis all_gather(32→128)+mesh_partition(128→32) round-trip; keeps the legit TP hidden gather (cluster_axis=1, 1536→12288). `partition_parallel_lm_head` (325-336) unchanged `("model",None)`; commented experimental variants left below.

## Sharding (SPMD via shard_model)
Mesh `("batch","model")`=(DP,TP). `MODULE_TYPE_TO_WRAPPING_FUNC` (390-405): QKV/MergedColumn→`("model",batch_axis)`; ColumnParallel→`("model",None)`; RowParallel→`(batch_axis,"model")`; `ParallelLMHead`→`("model",None)`; `VocabParallelEmbedding` weight `(None,"model")` + output hook `("batch",None,None)`. Vocab dim unsharded (tt-mlir #3370). `_pin_input_shardings` (1757) pins input_ids/inputs_embeds/positions to batch_axis in DP modes.

## Test files
- **test_data_tensor_parallel_generation.py::test_dptp_devstral (~292-398):** markers nightly/data_parallel/tensor_parallel/bh_galaxy. Full production knobs, `num_hidden_layers=2`. Env: `TT_DEVSTRAL_MAX_MODEL_LEN` (default 1024; sweep 4096/8192), `TT_DEVSTRAL_TRACE` (default 1; =0 bypasses trace). `cpu_sampling=True` REQUIRED. Asserts output coherence + host memory.
- **test_prefill.py (NEW):** DP+TP chunked-hang reproducer. `test_prefill_single_device`/`_chunked` (single-chip controls, no CCL). `test_prefill_dptp_chunked_repro` (155): 4-cell {chunked}×{trace} matrix, Devstral 4×8 + Qwen3 8×4. `test_prefill_dptp_chunked_smallmesh` (236): [2,4] 8-chip. (Field expectation docs PRE-DATE the D50/D58 fixes — treat as stale where trace-on now captures.)
- **test_vllm_benchmarks.py:** module knobs (30-43): `_BENCH_MAX_MODEL_LEN` 128→1024, `TT_BENCHMARK_PREFILL_CHUNK_SIZE`/`_TRACE`. `_tp_config()` (99-130): defaults `optimization_level 2→1`, `use_2d_mesh False→True`, `min_context_len 32→128`. `TP_CONFIGS` (236) consumed by `test_vllm_tp_benchmark` (418). Galaxy entries: `devstral-123b-galaxy-tp` (281) mesh[4,8] gpu_util=0.3 opt1 bfp8 trace prefill_chunk=128 num_hidden_layers=2; `qwen3-32b-galaxy-tp` (312) mesh[8,4] gpu_util=0.45 max_num_seqs=256.
  - **num_hidden_layers is set here** (and in the test's additional_config) — this is the knob to run 2-3 layer versions.

## Key architectural gates
1. **`max_model_len % 256 == 0`** required when chunked prefill on (else `_chunked_sdpa_active=False` / NotImplementedError). Valid: 256, 512, 1024, 4096, 8192.
2. `prefill_chunk_size > 0` is the single switch for chunked prefill.
3. Chunked SDPA = its own traced graph (extra precompiled graph per bucket, `prefix_chunk=True`).
4. `cpu_sampling=True` mandatory on 2D DP+TP mesh (#4387, #4440).
5. KV under bfp8: `uint8` accounting spec + bf16 staged buffer; bfp_bf4 unsupported (#5011).
6. `num_hidden_layers=2` for fast graph-coverage; full depth (~88) needs `gpu_memory_utilization` re-tune.
7. 32 devices default mesh (4,8); explicit `mesh_shape` must multiply to device count.
