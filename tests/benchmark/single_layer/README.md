# Single-layer LLM/encoder perf benchmarks

Everything related to single-layer LLM/encoder benchmarks lives here:

| File / dir         | Role                                                          |
| ------------------ | ------------------------------------------------------------- |
| `subsets.sh`       | Source of truth: which tests run, grouped by mesh tier.       |
| `run_benchmarks.py`| Runner: invokes pytest, collects MLIRs + per-test JSON.       |
| `generated/ttir/`  | Output: TTIR MLIRs + aggregated `measured_<device>.json`.     |
| `generated/ttnn/`  | Output: TTNN MLIRs.                                           |
| `README.md`        | This file: rationale for the subsets + JSON schema.           |

The pipeline entry point is [`regen.sh`](regen.sh), which sources
`subsets.sh`, invokes `run_benchmarks.py`, and handles tt-mlir SHA pinning and
device-reset retries.

The actual subset lists live in [`subsets.sh`](subsets.sh). The tables below
explain *why* each test is in its tier.

The subsets are non-overlapping; you compose them, e.g. `SUBSET=single,llmbox`
on an 8-device host or `SUBSET=single,galaxy` on a 32-device host. `llmbox`
and `galaxy` cannot be combined (different mesh shapes; the wrapper rejects).

| Subset   | Devices needed | Combine with                          |
| -------- | -------------- | ------------------------------------- |
| `single` | 1              | (alone, or with `llmbox` or `galaxy`) |
| `llmbox` | 8 (1×8 mesh)   | usually `single,llmbox`               |
| `galaxy` | 32 (4×8 mesh)  | usually `single,galaxy`               |

## Single chip

Aligned 1:1 with tt-mlir's `single_block_layer_perf_tests/` lit tests. Add or
remove entries here in lockstep with that directory.

| Test                | Why                                                                                                                                                       |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_llama_3_2_1b` | Canonical small Llama; only model with both prefill+decode lit tests.                                                                                     |
| `test_llama_3_1_8b` | Larger Llama 3.1 (different scale + GQA pattern).                                                                                                         |
| `test_phi2`         | Phi-2 (2.7B); Phi family coverage.                                                                                                                        |
| `test_falcon3_1b`   | Tuple `read_logits_fn = output[0]` path; stand-in for falcon3_3b/7b.                                                                                      |
| `test_gemma_2_2b`   | Gemma 2 family (SWA, soft-cap, different RMSNorm placement). Gated.                                                                                       |
| `test_mistral_7b`   | Canonical Mistral; sentencepiece+protobuf path. Distinct from Ministral-8B (covered in `llmbox`).                                                         |
| `test_qwen_3_0_6b`  | Qwen 3 family; stand-in for qwen_3_* and qwen_2_5_*.                                                                                                      |
| `test_bert`         | Encoder; only `_g0_` graph; stand-in for other encoders.                                                                                                  |

## llmbox (8 devices, 1×8 mesh)

TP code-path coverage. Run on top of the single-chip set.

| Test                              | Why                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------ |
| `test_llama_3_1_8b_instruct_tp`   | Default TP, `trace_enabled=True`, `optimization_level=2`. Baseline TP path.                      |
| `test_falcon3_7b_tp`              | TP + tuple read_logits_fn. Frequent regression target (tt-xla#4573).                             |
| `test_ministral_8b_tp`            | `trace_enabled=False` + `optimization_level=1`. Frequent perf regressions (tt-xla#4474).         |
| `test_gpt_oss_20b_tp`             | MoE `mesh_config_fn` + `shard_spec_fn` (1×8). Most-regressed TP LLM (tt-xla#4473, #4064, #3907). |
| `test_llama_3_1_70b_tp`           | Only model using `weight_dtype_overrides` (mlp.gate_proj/up_proj → bfp_bf4). Must run on llmbox. |

## Galaxy (32 devices, 4×8 mesh)

Single-chip set + galaxy-only TP tests. Does not include llmbox tests.

| Test                                            | Why                                                                              |
| ----------------------------------------------- | -------------------------------------------------------------------------------- |
| `test_llama_3_1_70b_tp_galaxy`                  | 32-way TP variant; mesh_shard at galaxy scale.                                   |
| `test_gpt_oss_120b_tp_dp_galaxy_batch_size_128` | Only test combining tensor + data parallelism. MoE custom shard at galaxy scale. |

## What's deliberately not here

- `*_accuracy` variants — different code path (TOP1/TOP5 token accuracy).
- Vision tests (resnet, vit, …) — don't accept `--num-layers`.
- `vllm_*` — different runner.
- Open phi/qwen_2_5 PCC issues (#4478, #4479, #4436, #4437) — they trip at full
  32-token decode, but regen runs at `--max-output-tokens 2`. Use the regular
  benchmark CI for those.

## Direct runner usage

`regen.sh` is the supported entry point. To call the runner directly without
the pin/build/hang-recovery scaffolding:

```bash
source tests/benchmark/single_layer/subsets.sh
python tests/benchmark/single_layer/run_benchmarks.py \
    --test "${SUBSET_SINGLE},${SUBSET_LLMBOX}"
```

## measured_<device>.json schema

Written by the runner after every completed test (so a Ctrl-C / device hang mid-sweep
leaves a partial-but-valid file).

```json
{
  "device_type": "llmbox",
  "arch": "wormhole",
  "device_count": 8,
  "tests": {
    "test_llama_3_1_8b_instruct_tp": {
      "group": "llm",
      "status": "ok",
      "error": null,
      "samples_per_sec": 12.34,
      "ttft_ms": 1234.5,
      "prefill_pcc": 0.997,
      "prefill_pcc_target": 0.99,
      "decode_pcc": 0.985,
      "decode_pcc_target": 0.99
    }
  }
}
```

`status="failed"` means either pytest crashed OR the measured PCC was below
its target (`error` field describes which).
