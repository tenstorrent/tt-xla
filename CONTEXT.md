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

For fast iteration (3-layer model, compiles much faster):
```bash
python examples/pytorch/gpt_oss_120b.py --interactive --sparse-moe --batch-size 8 |& tee layer_3.log
```
Enable 3-layer mode by setting `config.num_hidden_layers = 3` in `setup_model_and_tokenizer()`. Note: `trust_remote_code=True` (capital T) — lowercase `true` is a Python NameError.

## Key Constraints
- **batch_size=8 minimum** with `--sparse-moe`: sparse MoE dispatch requires `batch_size × dispatch_devices(4) = multiple of 32`. Min is 8.
- **PREFILL_PAD_LEN=100**: Fixed prefill length so XLA cache always hits the same compiled graph. Must NOT be a multiple of 32 — seq_len % 32 == 0 triggers the `split_seq` dispatch path in sparse MoE which hangs on QB2. The validated path is `split_bd` (BD=32 always divisible by 32).
- **bfp8 disabled**: `experimental_weight_dtype: bfp_bf8` is commented out — global bfp8 causes prefill execution to hang on QB2. Expert weights still use bfp4 per-tensor override (72 overrides across 36 layers). Re-enable and test once QB2 bfp8 execution is validated.

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

### New changes (ssalice, 2026-04-08)
| Change | What | Why |
|--------|------|-----|
| `mark_sharding_on_inputs_and_model` | When `sparse_moe=True` and layer is `A2aSparseMLP`, expert weights use `(("batch", "model"), None, None)` instead of `("model", None, None)` | Matches `get_moe_shard_specs()` in the runner (`inject_custom_moe`). Required for sparse MoE correctness. |
| Compilation cache status | Prints `[WARM (N entries)]` or `[COLD (will compile)]` at startup | Easy visibility into whether cache will be used |
| `torch.compile()` timing | Times the wrapper call separately to expose pre-forward overhead | The ~29s gap between "torch.compile starts" and prefill done was hidden; torch.compile wrapper itself takes significant time for 120B |

## Changes in Branch Commits (by others)

### `209e15760` — "gpt oss 120b works on qb2" (mmilosevic)
- `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`: added `enable_weight_bfp8_conversion: true` for 120B entry
- `tests/runner/testers/torch/dynamic_torch_model_tester.py`: `enable_sparse_mlp(..., cluster_axis=1)`
- `third_party/CMakeLists.txt`: pinned tt-mlir to `aknezevic/sparse_bfp4` branch (now resolved to `7ae5e1f36d4a013e965439b5decd8af6e0561c39`)

### `e185405317` in `tt_forge_models` — "4b expert weights" (mmilosevic)
- `gpt_oss/pytorch/mixed_precision_configs/gpt-oss-120b.json`: new file, sets `gate_up_proj` + `down_proj` to `bfp_bf4`
- `gpt_oss/pytorch/loader.py`: added `get_weight_dtype_config_path()`, QB2 mesh `(1,4)`, and new `load_shard_spec` (model-axis-only, no batch sharding)

## Performance on QB2 (stable, no bfp8)
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
1. **bfp8 hangs** — global bfp8 causes prefill to hang during hardware execution on QB2. Tested both with bfp8 enabled and in pure bf16 — both hang on qb2-120-p01t08 after the MLIR module is emitted. May be a machine-level issue rather than precision-specific. Testing on a second machine and through the test runner to isolate.
2. **73s prefill** — slow for chatbot UX. May improve with bfp8 (lower precision = faster matmuls). Also worth testing smaller `PREFILL_PAD_LEN` (e.g. 75) to reduce compute.
3. **split_seq path unvalidated** — any `PREFILL_PAD_LEN` that is a multiple of 32 triggers the `split_seq` dispatch path in `A2aSparseMLP`. This hung at seq_len=128. Do not use multiples of 32 until this is debugged.
4. **No multi-turn memory** — KV cache is reset between prompts, model has no memory of previous turns.
5. **Decode output runs to max cache** — generation stops at EOS or when 256-100=156 tokens are generated. Could add `--max-new-tokens` flag.

## Hang Debugging — 2026-04-08
- **Symptom**: prefill always hangs after `END OF MLIR MODULE` is printed in the log. Process stays at ~400% CPU but no new output and no progress.
- **Tested**: bfp8 enabled → hangs. bf16 (no precision override) → also hangs on same machine (qb2-120-p01t08). Small unrelated model ran fine on same machine.
- **Hypothesis**: may be a machine-specific hardware/IOMMU state issue (earlier in the session we saw `Expected NOC address: 0x1000000000000000, but got 0x1000000040000000` bus errors). tt-smi warm reset was done but may not have fully recovered state. Full reboot needed to confirm.
- **Parallel investigation**: running 120B through test runner (`test_models.py`) on a different machine to see if hang reproduces there.
- **DRAM monitoring branch**: created `ssalice/sparse_bfp4` in `~/ssalice/tt-mlir` — cherry-picks two commits from `ssalice/phi4-DRAM` on top of `7ae5e1f36d4a013e965439b5decd8af6e0561c39` (the current tt-xla pinned commit). Contains DRAM monitoring hack that prints each op as it executes, so we can see exactly which op the hang occurs at. Plan: update `third_party/CMakeLists.txt` to point to this branch and rebuild.

## tt-mlir Branch for Debugging
- **Branch**: `ssalice/sparse_bfp4` on `github.com/tenstorrent/tt-mlir`
- **Base**: `7ae5e1f36d4a013e965439b5decd8af6e0561c39` (same as current tt-xla pin)
- **Commits on top**:
  1. `9b01f635` — "use top down traversal for pattern rewriting"
  2. `6c0ada44` / `f43595e79` (cherry-picked) — "DRAM Monitoring Hack" — prints each op + samples device DRAM before execution
- **To use**: update `TT_MLIR_VERSION` in `third_party/CMakeLists.txt` to the tip of `ssalice/sparse_bfp4`, rebuild

## Compilation Cache
Location: `~/.cache/tt_xla/gpt_oss_120b/`
- 2 files when fully warm: prefill graph + decode graph
- Clear with `rm -rf ~/.cache/tt_xla/gpt_oss_120b` to force recompile
- Cache invalidates automatically if graph changes (model, shapes, compiler options)

## Checking Machine Health
```bash
who                          # who is logged in
lsof /dev/tenstorrent/*      # who has TT devices open (run with sudo if empty looks wrong)
lsmod | grep tenstorrent     # use count — 0 means devices are free
ps aux | grep python         # any model jobs running
tt-smi -s                    # device telemetry (healthy if it returns JSON)
tt-smi -r all                # warm reset all devices
```
