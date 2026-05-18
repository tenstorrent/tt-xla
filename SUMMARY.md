loader_path: third_party.tt_forge_models.gemma3_singlish_sinhala_merged.causal_lm.pytorch.loader
variant_id: Merged
arch: p150
status: DONE_FAIL
test_function: test_gemma3_singlish_sinhala_merged
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
failure_reason: "loader bug: Gemma3ForCausalLM.__init__() does not accept 'use_cache' kwarg passed by loader.py; fix needed in tt-forge-models"

# Benchmark added: test_gemma3_singlish_sinhala_merged

## Test
tests/benchmark/test_llms.py::test_gemma3_singlish_sinhala_merged

## Model
- HF name:    savinugunarathna/Gemma3-Singlish-Sinhala-Merged
- Loader:     third_party.tt_forge_models.gemma3_singlish_sinhala_merged.causal_lm.pytorch.loader
- Variant:    ModelVariant.MERGED ("Merged")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The bring-up run (--num-layers 1 --max-output-tokens 3) failed immediately at model
loading with:

    TypeError: Gemma3ForCausalLM.__init__() got an unexpected keyword argument 'use_cache'

The loader (`loader.py`) sets `model_kwargs = {"use_cache": False}` and passes those
kwargs to `AutoModelForCausalLM.from_pretrained(...)`, which forwards them to
`Gemma3ForCausalLM.__init__()`. The Gemma3 architecture (model_type: gemma3_text,
hidden_size=640, num_hidden_layers=18 — approx 0.5B params) does not accept
`use_cache` as a constructor argument in the version of transformers present in the
venv. This is a loader bug; no fix is possible under this skill (editing
`third_party/tt_forge_models/` is out of scope). The fix belongs in tt-forge-models.

## Measured (full model, defaults)
- Sample per second:  N/A (model failed to load)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (no perf metrics generated)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A

## Files changed
- tests/benchmark/test_llms.py (added test_gemma3_singlish_sinhala_merged)

## tt-forge-models submodule
no change
