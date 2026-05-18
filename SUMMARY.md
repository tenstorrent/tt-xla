loader_path: third_party.tt_forge_models.exaone_3_5_awq.causal_lm.pytorch.loader
variant_id: 3.5_7.8B_Instruct_AWQ
arch: p150
status: DONE_FAIL
test_function: test_exaone_3_5_7_8b_instruct_awq
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
failure_reason: "transformers 5.2.0 incompatibility: EXAONE-3.5-7.8B-Instruct-AWQ remote code (modeling_exaone.py) requires `check_model_inputs` from `transformers.utils.generic`, which is not present in transformers==5.2.0"

# Benchmark added: exaone_3_5_7_8b_instruct_awq

## Test
tests/benchmark/test_llms.py::test_exaone_3_5_7_8b_instruct_awq

## Model
- HF name:    LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ
- Loader:     third_party.tt_forge_models.exaone_3_5_awq.causal_lm.pytorch.loader
- Variant:    3.5_7.8B_Instruct_AWQ

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
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

## Failure details
The test fails immediately during model loading with:

    ImportError: cannot import name 'check_model_inputs' from 'transformers.utils.generic'
    (/home/ttuser/tt-xla/venv/lib/python3.12/site-packages/transformers/utils/generic.py)

Traceback (abbreviated):
    third_party/tt_forge_models/exaone_3_5_awq/causal_lm/pytorch/loader.py:91: in load_model
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
    modeling_exaone.py:40: in <module>
        from transformers.utils.generic import check_model_inputs, maybe_autocast

The model uses `trust_remote_code=True`, which downloads `modeling_exaone.py` from
LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ. That file requires `check_model_inputs`
from transformers, which was not added until a version later than 5.2.0. This is
a model-code / transformers-version incompatibility; it cannot be fixed within this
skill's scope (which forbids editing third_party/tt_forge_models/ or upgrading the
global transformers pin).

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or execution

## Files changed
- tests/benchmark/test_llms.py (test function added)
- .github/workflows/perf-bench-matrix.json (entry added)

## tt-forge-models submodule
no change
