# GPT-OSS 120B QB2 Chatbot — Context

## Machine
- **QuietBox2 (QB2)**: 2x Blackhole p300c cards, 4 chips total
- **Mesh**: `(1, 4)` — batch axis = 1 device, model axis = 4 devices
- **Memory**: ~64-96 GB total
- **Container**: Docker on shared `ttuser` account, SSH via `qb2_17`
- **Slurm**: QB2s are now in the `bh_qb2` Slurm partition (`qb2-120-p01t[01,08]`). Reserve a node with `srun -p bh_qb2 --nodelist=qb2-120-p01t08 --pty sleep infinity` inside a tmux session on the slurm login node (`ssh exabox`, username `ssalice`). This blocks other Slurm jobs but NOT direct SSH users.
- **Model weights**: Already downloaded at `/home/ttuser/.cache/huggingface/hub/models--openai--gpt-oss-120b`

## What This File Does
`examples/pytorch/gpt_oss_120b.py` — interactive chatbot demo for GPT-OSS 120B on QB2.
Run with:
```bash
python examples/pytorch/gpt_oss_120b.py --interactive --sparse-moe --batch-size 8 |& tee gpt_oss.log
```

For fast iteration (1-layer model, compiles much faster):
```bash
python examples/pytorch/gpt_oss_120b.py --interactive --sparse-moe --batch-size 8 |& tee layer_1.log
```
Enable 1-layer mode by setting `config.num_hidden_layers = 1` in `setup_model_and_tokenizer()` or via the loader (`self.num_layers = 1` in `loader.py:70`). Note: `trust_remote_code=True` (capital T) — lowercase `true` is a Python NameError.

## Key Constraints
- **batch_size=8 minimum** with `--sparse-moe`: sparse MoE dispatch requires `batch_size × dispatch_devices(4) = multiple of 32`. Min is 8.
- **PREFILL_PAD_LEN**: Fixed prefill length so XLA cache always hits the same compiled graph. Currently set to **96** (a multiple of 32, which uses the `split_seq` dispatch path in sparse MoE). Previously 100 (non-multiple of 32, `split_bd` path). The `split_seq` path was previously thought to hang on QB2 but now works with the rebased `ssalice/sparse_bfp4` tt-mlir branch (confirmed 2026-04-09).

## Mixed Precision — How It Actually Works
Two independent mechanisms exist for weight dtype conversion:

### 1. Global `experimental_weight_dtype` (compiler config)
Set via `compiler_config.experimental_weight_dtype = "bfp_bf8"` or the YAML flag `enable_weight_bfp8_conversion: true`. Converts **all** weights to the specified dtype. This is what the test runner uses.

### 2. Per-tensor `apply_weight_dtype_overrides()` (parametrize-based)
Uses `torch.nn.utils.parametrize` to annotate individual weights. Supports a `"default"` key plus glob-pattern overrides. This is what the examples script and the standalone tests use.

**The chatbot example uses method 2** with this config (matching `gpt-oss-120b.json`):
```python
apply_weight_dtype_overrides(model, {
    "default": "bfp_bf8",                                    # all weights → bfp8
    "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",   # expert weights → bfp4
    "model.layers.*.mlp.experts.down_proj": "bfp_bf4",      # expert weights → bfp4
})
```

**What ends up in bfp8 (per layer):** q_proj, k_proj, v_proj, o_proj, router, input_layernorm, post_attention_layernorm, embed_tokens, model.norm, lm_head
**What ends up in bfp4:** gate_up_proj, down_proj (expert MLP weights only)
**What stays bf16:** All activations, attention compute (QK^T, softmax, attn×V), MoE dispatch/combine. bfp8/bf4 only applies to weight storage — the actual matmul computation runs in bf16.

### Test runner vs examples path discrepancy
The test runner (`test_models.py`) uses `enable_weight_bfp8_conversion` (method 1) which is global-only — it cannot do mixed bf8+bf4. The JSON auto-discovery in `dynamic_torch_model_tester.py` handles per-tensor overrides, but if `enable_weight_bfp8_conversion` is also set, they can conflict. The standalone test we created (`tests/torch/models/gpt_oss/test_gpt_oss_120b_qb2.py`) uses method 2 directly to avoid this issue.

## Standalone Tests
`tests/torch/models/gpt_oss/test_gpt_oss_120b_qb2.py` — created to replicate the examples path exactly, bypassing the test runner infra.

Three tests:
- **`test_gpt_oss_120b_prefill`** — full prefill step (100 tokens, batch 8)
- **`test_gpt_oss_120b_decode`** — prefill + one decode step
- **`test_gpt_oss_120b_decode_only`** — decode-only (single token, no prefill, empty cache). Produces the same compiled graph as the chatbot's decode step.

All three use:
- Model via `ModelLoader(variant=GPT_OSS_120B)` (MXFP4, bf16, eager attention)
- `apply_weight_dtype_overrides` with bf8 default + bf4 experts (method 2)
- Sparse MoE with `cluster_axis=1`
- StaticCache, full sharding matching the examples path

Run with:
```bash
pytest -svv tests/torch/models/gpt_oss/test_gpt_oss_120b_qb2.py::test_gpt_oss_120b_decode_only |& tee 1layer_test.log
```

**Status (2026-04-09):** `test_gpt_oss_120b_decode_only` passes on 1-layer model with bf8+bf4 mixed precision and DRAM monitoring enabled. Verified the ttnn graph has correct bf8 (attention/lm_head weights) and bf4 (expert MLP weights).

## Changes Made Relative to Galaxy Demo (Brian Tsoi / btsoi/gpt_oss_120b, PR ~3799)

### From the original draft (`4b12197e0`)
The original only supported Galaxy (32 devices) and llmbox (8 devices).

| Change | What | Why |
|--------|------|-----|
| `create_device_mesh` | Added `elif num_devices == 4: mesh_shape = (1, 4)` | QB2 has 4 chips |
| `enable_sparse_mlp` | Added `cluster_axis=1` | QB2's model axis is axis 1; dispatch/combine must happen there |
| `apply_weight_dtype_overrides` | Added bfp4 for `gate_up_proj` + `down_proj` across all layers | Expert weights are the biggest tensors in MoE, compress to fit QB2 memory |
| `mark_sharding_on_inputs_and_model` | Simplified to model-axis-only sharding, no batch-axis sharding, biases/norms replicated | QB2 batch axis = 1 device so sharding along it is a no-op |
| `xr.initialize_cache` | Added compilation cache at `~/.cache/tt_xla/gpt_oss_120b` | Avoid recompiling on every run |
| `PREFILL_PAD_LEN=100` | Fixed prefill token length | Cache hit every prompt; avoids split_seq dispatch path |
| Interactive UX | Single `You:` prompt, silently pads to batch of 8 | Chatbot feel |
| Channel filter | Only streams `final` channel tokens | GPT-OSS uses analysis/commentary/final channels; only final is the actual response |
| Performance stats | Prefill time, decode latency, tokens/s | Demo metrics |

### New changes (ssalice, 2026-04-08/09)
| Change | What | Why |
|--------|------|-----|
| `mark_sharding_on_inputs_and_model` | When `sparse_moe=True` and layer is `A2aSparseMLP`, expert weights use `(("batch", "model"), None, None)` instead of `("model", None, None)` | Matches `get_moe_shard_specs()` in the runner (`inject_custom_moe`). Required for sparse MoE correctness. |
| Compilation cache status | Prints `[WARM (N entries)]` or `[COLD (will compile)]` at startup | Easy visibility into whether cache will be used |
| `torch.compile()` timing | Times the wrapper call separately to expose pre-forward overhead | The ~29s gap between "torch.compile starts" and prefill done was hidden; torch.compile wrapper itself takes significant time for 120B |
| `tests/conftest.py` | Catch `AttributeError` alongside `OSError` in `TeeCapture.start()` | The venv Python 3.12 (Clang-built) lacks `os.memfd_create`, which crashes test setup. This lets tests fall back gracefully. |
| `test_config_inference_tensor_parallel.yaml` | Added `enable_weight_bfp4_conversion: true` to 120B entry | Enables global bfp4 alongside bfp8 in test runner (note: bfp4 wins if both set) |
| `tests/runner/test_config/constants.py` | Added `enable_weight_bfp4_conversion` to allowed config keys | Analogous to bfp8 flag |
| `tests/runner/test_utils.py` | Resolves `enable_weight_bfp4_conversion` from YAML config | Analogous to bfp8 |
| `tests/runner/test_models.py` | Sets `experimental_weight_dtype = "bfp_bf4"` when flag is set | Analogous to bfp8 |
| `tests/runner/conftest.py` | Reports `"bfp4"` in weights_dtype tag | Analogous to bfp8 |
| `examples/pytorch/gpt_oss_120b_og.py` | Backup of original Galaxy demo script before QB2 modifications | Reference for what the upstream code looked like |
| Standalone test | `tests/torch/models/gpt_oss/test_gpt_oss_120b_qb2.py` | Replicates examples path with proper bf8+bf4 mixed precision, bypasses test runner limitations |

### Uncommitted local changes (as of 2026-04-09)
- `examples/pytorch/gpt_oss_120b.py`: `PREFILL_PAD_LEN` changed from 100 to **96** (WARNING: 96 is a multiple of 32, which triggers the `split_seq` dispatch path. This was likely for testing — revert to 100 for production.)
- `tests/conftest.py`: `TeeCapture.start()` catches `AttributeError` alongside `OSError` (fixes `os.memfd_create` crash in Docker)
- `CONTEXT.md`: this file

## Changes in Branch Commits (by others)

### `209e15760` — "gpt oss 120b works on qb2" (mmilosevic)
- `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`: added `enable_weight_bfp8_conversion: true` for 120B entry
- `tests/runner/testers/torch/dynamic_torch_model_tester.py`: `enable_sparse_mlp(..., cluster_axis=1)`
- `third_party/CMakeLists.txt`: pinned tt-mlir to `aknezevic/sparse_bfp4` branch (now resolved to `7ae5e1f36d4a013e965439b5decd8af6e0561c39`)

### `e185405317` in `tt_forge_models` — "4b expert weights" (mmilosevic)
- `gpt_oss/pytorch/mixed_precision_configs/gpt-oss-120b.json`: new file, sets `gate_up_proj` + `down_proj` to `bfp_bf4`
- `gpt_oss/pytorch/loader.py`: added `get_weight_dtype_config_path()`, QB2 mesh `(1,4)`, and new `load_shard_spec` (model-axis-only, no batch sharding)

### `a6777bb62` in `tt_forge_models` — "added default bfp8" (ssalice)
- `gpt_oss/pytorch/mixed_precision_configs/gpt-oss-120b.json`: added `"default": "bfp_bf8"` so all non-expert weights get bfp8 via JSON auto-discovery

## Performance on QB2 (stable, bf4-only — no bfp8)
| Metric | Value |
|--------|-------|
| Model load | ~3 min |
| Prefill cold compile | ~3 min (first run only) |
| Decode cold compile | ~55s (first run only) |
| Prefill execution (warm) | ~73s per prompt |
| Decode latency | ~512ms/token |
| Throughput (8 users) | ~15 tokens/s |
| Throughput (per user) | ~2 tokens/s |

Note: reported "Prefill (incl. compile)" time only measures the forward pass. The `torch.compile()` wrapper call adds ~29s of overhead before `step_start` that is not counted. True wall-clock cost is ~102s on first prompt.

## Known Issues / TODO
1. **Full-model prefill hangs with bf8+bf4** — The full 36-layer model with bf8+bf4 mixed precision hangs during prefill execution on QB2 after the MLIR module is emitted. The 1-layer decode-only test passes fine. Investigating whether this is prefill-specific, layer-count-dependent, or machine-specific.
2. **73s prefill** — slow for chatbot UX. May improve with bfp8 (lower precision = faster matmuls). Also worth testing smaller `PREFILL_PAD_LEN` (e.g. 75) to reduce compute.
3. **~~split_seq path unvalidated~~** — RESOLVED. `PREFILL_PAD_LEN=96` (multiple of 32, `split_seq` path) now works on QB2 with the rebased `ssalice/sparse_bfp4` tt-mlir branch. Previously hung at seq_len=128 on the older tt-mlir pin.
4. **No multi-turn memory** — KV cache is reset between prompts, model has no memory of previous turns.
5. **Decode output runs to max cache** — generation stops at EOS or when 256-100=156 tokens are generated. Could add `--max-new-tokens` flag.

## Hang Debugging — 2026-04-08/09
- **Symptom**: prefill always hangs after `END OF MLIR MODULE` is printed in the log. Process stays at ~400% CPU but no new output and no progress.
- **Tested**: bfp8 enabled → hangs. bf16 (no precision override) → also hangs on same machine (qb2-120-p01t08). Small unrelated model ran fine on same machine.
- **Hypothesis**: may be a machine-specific hardware/IOMMU state issue (earlier in the session we saw `Expected NOC address: 0x1000000000000000, but got 0x1000000040000000` bus errors). tt-smi warm reset was done but may not have fully recovered state. Full reboot needed to confirm.
- **Parallel investigation**: running 120B through test runner (`test_models.py`) on a different machine to see if hang reproduces there.
- **DRAM monitoring branch**: `ssalice/sparse_bfp4` in tt-mlir, rebased onto latest. Contains DRAM monitoring hack that prints each op + samples device DRAM before execution. Currently built and active in the tt-xla tree.
- **1-layer decode with DRAM monitoring**: PASSED (2026-04-09). The decode-only test with bf8+bf4 mixed precision completes successfully with DRAM monitoring active. Confirmed the ttnn graph has correct dtype assignments. This narrows the hang to either prefill-specific behavior or full-model (36-layer) scale.

## tt-mlir Branch for Debugging
- **Branch**: `ssalice/sparse_bfp4` on `github.com/tenstorrent/tt-mlir`
- **Base**: `7ae5e1f36d4a013e965439b5decd8af6e0561c39`
- **Rebased** onto latest (2026-04-09), force-pushed
- **Contains**: DRAM monitoring hack — prints each op + samples device DRAM before execution
- **Currently active**: `third_party/CMakeLists.txt` `TT_MLIR_VERSION` is set to `ssalice/sparse_bfp4` (branch name, not commit hash). The original pin `0cd797d81d449aaacc2bf38eea2fe02a0b9fc56e` is commented out.

## Compilation Cache
Location: `~/.cache/tt_xla/gpt_oss_120b/`
- 4 files currently cached:
  - 2 for bf16+bf4 (prefill ~21MB + decode ~15MB)
  - 2 for bf8+bf4 (prefill ~21MB + decode ~15MB)
- Clear with `rm -rf ~/.cache/tt_xla/gpt_oss_120b` to force recompile
- Cache invalidates automatically if graph changes (model, shapes, compiler options)

## Log Files (as of 2026-04-09)
| File | Description |
|------|-------------|
| `ttnn_1layer_decode.log` | Clean ttnn graph for 1-layer decode with bf8+bf4 (156KB). Safe to send externally. |
| `1layer_test.log` | Full test output for 1-layer decode-only test with bf8+bf4 and DRAM monitoring (1.2MB) |
| `decode.log` | Full 36-layer bf8+bf4 decode graph extracted from compilation cache (15MB) |

## Checking Machine Health
```bash
who                          # who is logged in
lsof /dev/tenstorrent/*      # who has TT devices open (run with sudo if empty looks wrong)
lsmod | grep tenstorrent     # use count — 0 means devices are free
ps aux | grep python         # any model jobs running
tt-smi -s                    # device telemetry (healthy if it returns JSON)
tt-smi -r all                # warm reset all devices
```
