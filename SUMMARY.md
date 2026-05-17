loader_path: third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader
variant_id: 029_SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_q4_k_m_gguf
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
failure_reason: "ETH training timed out (900000ms, eth core 22,16): device initialization fails on every attempt; device cannot be reset (tt-smi not available, PCI reset requires root)"

# Benchmark added: test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/029-shisa-gamma-7b-v1-v2new-dpo405b-i1-GGUF
- Loader:     third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF (value: "029_SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device init failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         0:21:09 (15 min ETH timeout + model download)
- Hardware:           n150 (wormhole_b0, device 0x401e)

## Failure Details
Both test runs (num_layers=1, max_output_tokens=3) failed at device initialization:
```
RuntimeError: ETH training timed out after 900000 ms, on eth core 22, 16
Location: wormhole_tt_device.cpp:255
  tt::umd::TopologyDiscovery::wait_eth_cores_training
  tt::umd::TopologyDiscovery::discover_remote_devices
  ...
  torch_xla::bridge::GetCurrentAtenDevice()
```

The device at `/dev/tenstorrent/3` (PCI 0000:c1:00.0, Wormhole B0 0x401e) is one of 4
TT devices in the system. The ETH training timeout suggests this device expects Ethernet
connectivity to a peer (n300 pair) that is not accessible in this container environment.
Reset requires tt-smi or root PCI reset access — neither is available here.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test never reached device execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
All fields: N/A (device not initialized)

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
