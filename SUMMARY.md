loader_path: third_party.tt_forge_models.gemma_3_12b_it_max_horror_gguf.causal_lm.pytorch.loader
variant_id: 12B_IT_MAX_HORROR_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma_3_12b_it_max_horror_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "loader GGUF_FILE 'Gemma-3-12b-it-MAX-HORROR-Imatrix-Q4_K_M.gguf' does not exist in HF repo DavidAU/Gemma-3-12b-it-MAX-HORROR-Imatrix-GGUF (HTTP 404); repo was reorganized, available Q4_K_M file is now 'Gemma-3-12b-it-MAX-HORROR-D_AU-Q4_K_M-imat.gguf' — fix requires editing loader.py in tt-forge-models"

# Benchmark added: test_gemma_3_12b_it_max_horror_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_3_12b_it_max_horror_gguf

## Model
- HF name:    DavidAU/Gemma-3-12b-it-MAX-HORROR-Imatrix-GGUF
- Loader:     third_party.tt_forge_models.gemma_3_12b_it_max_horror_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_IT_MAX_HORROR_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The loader's hardcoded `GGUF_FILE = "Gemma-3-12b-it-MAX-HORROR-Imatrix-Q4_K_M.gguf"` returned
HTTP 404 from HuggingFace. The repository `DavidAU/Gemma-3-12b-it-MAX-HORROR-Imatrix-GGUF`
was reorganized and all file names changed. The equivalent Q4_K_M file is now
`Gemma-3-12b-it-MAX-HORROR-D_AU-Q4_K_M-imat.gguf`.

Fix required: update `GGUF_FILE` in
`third_party/tt_forge_models/gemma_3_12b_it_max_horror_gguf/causal_lm/pytorch/loader.py`
to the new filename. This change belongs in the tt-forge-models repo.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md (this file)

## tt-forge-models submodule
no change
