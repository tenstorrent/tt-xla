loader_path: third_party.tt_forge_models.dream_omni_2_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_dream_omni_2_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "device initialization failed: UMD NOC address mismatch in SiliconSysmemManager (silicon_sysmem_manager.cpp:391) — Expected NOC address 0x1000000000000000 but got 0x1000000040000000 → RuntimeError: Proceeding could lead to undefined behavior"

# DreamOmni2 GGUF — Q4_K_M_GGUF — p150 Benchmark Attempt

## Model

| Field | Value |
|---|---|
| Loader | `third_party.tt_forge_models.dream_omni_2_gguf.causal_lm.pytorch.loader` |
| Variant | `Q4_K_M_GGUF` (`DREAM_OMNI_2_7_6B_Q4_K_M_GGUF`) |
| HF repo | `xiabs/DreamOmni2`, subfolder `vlm-model` |
| Architecture | `Qwen2_5_VLForConditionalGeneration` (VLM, text-only inference) |
| Parameters | ~8.29B (declared 7.6B) |
| Hardware | Blackhole p300c → arch `p150` |

## Status: DONE_FAIL

### Failure

The test failed at device initialization during the bring-up run
(`--num-layers 1 --max-output-tokens 3`), after ~464 seconds.

```
WARNING - Expected NOC address: 0x1000000000000000, but got 0x1000000040000000
RuntimeError: Proceeding could lead to undefined behavior
  File "silicon_sysmem_manager.cpp", line 391
    SiliconSysmemManager::pin_or_map_sysmem_to_device()
```

The full traceback originates in UMD's `SiliconSysmemManager` during
`LocalChip::start_device()` → `Cluster::start_driver()` → tt-metal
`MetalContext` initialization. This is a hardware/UMD infrastructure error
and is not fixable within the benchmark skill scope.

### Steps completed

- [x] Build probe passed (pjrt_plugin_tt importable)
- [x] TT_MESH_GRAPH_DESC_PATH set for p150 Blackhole
- [x] Variant `DREAM_OMNI_2_7_6B_Q4_K_M_GGUF` confirmed present at submodule HEAD
- [x] Size check passed: 7.6B declared ≤ 25B p150 cap
- [x] Test function `test_dream_omni_2_gguf` added to `tests/benchmark/test_llms.py`
- [x] Model weights fully cached (~15.6 GB, 4 safetensors shards)
- [x] Bring-up run attempted with `--num-layers 1 --max-output-tokens 3`
- [ ] Device initialization succeeded — **FAILED** (UMD NOC address mismatch)

### Configuration hard-coded in test

| Setting | Value |
|---|---|
| `optimization_level` | `DEFAULT_OPTIMIZATION_LEVEL` (2) |
| `trace_enabled` | default (True) |
| `experimental_weight_dtype` | `"bfp_bf8"` (via `DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE`) |
| `batch_size` | 32 |

## Action required

File a UMD/tt-metal issue for the NOC address mismatch on Blackhole p300c:
`SiliconSysmemManager::pin_or_map_sysmem_to_device` — Expected
`0x1000000000000000`, got `0x1000000040000000`.
