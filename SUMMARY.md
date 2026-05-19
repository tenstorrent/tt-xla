loader_path: third_party.tt_forge_models.llama_3_1_8b_instruct_openbookqa_sft_para.causal_lm.pytorch.loader
variant_id: 8B_Instruct_OpenbookQA_SFT_Para
arch: p150
status: DONE_PASS
test_function: test_llama_3_1_8b_instruct_openbookqa_sft_para
samples_per_second: 22.369783254696088
ttft_ms: 474.441179
prefill_pcc: 0.999041
first_decode_pcc: 0.998514
top_perf_samples_per_sec: 41.6804
pct_of_target: 53.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_llama_3_1_8b_instruct_openbookqa_sft_para

## Test
tests/benchmark/test_llms.py::test_llama_3_1_8b_instruct_openbookqa_sft_para

## Model
- HF name:    qiaw99/Llama3.1-8B-Instruct-OpenbookQA-SFT-para
- Loader:     third_party.tt_forge_models.llama_3_1_8b_instruct_openbookqa_sft_para.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_1_8B_INSTRUCT_OPENBOOKQA_SFT_PARA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  22.369783254696088
- TTFT (ms):          474.441179
- Prefill PCC:        0.999041
- First decode PCC:   0.998514
- Wall clock:         0:16:48
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_3_1_8b_instruct_openbookqa_sft_para_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 53.7% (22.37 / 41.68)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             491035558016
- breakdown.matmul:        491035558016
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8198033604
- effective_count:        7672697028
- memory_bytes:           9203163914
- memory_gb:              8.571114310994744
- effective_memory_bytes: 8152490762
- effective_memory_gb:    7.592598685994744
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 41.6804
- top_perf_time_ms:         23.9921
- dram_time_ms:             15.9947
- compute_time_ms_lofi:     0.5580
- compute_time_ms_hifi2:    1.1160
- compute_time_ms_hifi3:    1.6740
- compute_time_ms_hifi4:    2.2320

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: use hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
