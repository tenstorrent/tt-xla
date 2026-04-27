# GPT-OSS-120B bfp4 CPU-Bypass Accuracy Report

## Goal

Measure, for `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64`
with MoE experts quantized to `bfp_bf4`, the end-to-end PCC of:

1. the **device** run (all ops on TT hardware), and
2. the **CPU-bypass** run (same on-device `ttnn.typecast(..., BFLOAT4_B)`
   preserved, but each bfp4 matmul replaced by `torch.matmul` on CPU),

both compared against a **pure-CPU bf16 non-compiled reference** — the
standard PCC methodology the other tests use.

## Methodology

1. Compile the model with `tt_torch.codegen_py` (`backend="codegen_py"`,
   `export_tensors=True`), producing a standalone `main.py` with real HF
   weights serialized to `tensors/*.tensorbin`.
2. Parse the generated `main.py` for bfp4 matmul triplets
   (`ttnn.typecast(..., BFLOAT4_B) → ttnn.matmul → ttnn.typecast(..., BFLOAT16)`).
3. Emit `main_cpu_bypass.py`: each bfp4 matmul replaced by
   `typecast → BFLOAT16 → per-shard gather to host → torch.matmul → ttnn.from_torch`.
4. Run `main.py` and `main_cpu_bypass.py` in isolated subprocesses (they
   share `_cached__main` globals), saving the per-chip-stacked outputs to
   `device_outputs.pt` and `bypass_outputs.pt`.
5. Run a separate **pure-CPU reference** forward pass (bf16 weights, *no*
   bfp4, no TT, no SPMD) on the same inputs, save to
   `cpu_reference_outputs.pt`.
6. Reconstruct the full logical tensor from the 32-chip stack (target-shape-
   driven: dim `d` is row-sharded if `per_chip[d]*4==target[d]`, col-sharded
   if `*8`). Compute PCC in **float64 with chunked accumulation**.

### Op-level schema: what is replaced, what is preserved

In the generated `main.py` every bfp4 matmul shows up as a three-op
triplet — *quantize weight → matmul → dequantize output*:

```python
# 1. WEIGHT QUANTIZATION (preserved in bypass)
var_Wq = ttnn.typecast(var_W, ttnn.DataType.BFLOAT4_B, memory_config=...)

# 2. MATMUL (replaced by CPU torch.matmul in bypass)
var_Y  = ttnn.matmul(var_A, var_Wq, memory_config=..., ...)

# 3. OUTPUT CAST BACK TO BF16 (preserved as-is)
var_Yb = ttnn.typecast(var_Y, ttnn.DataType.BFLOAT16, memory_config=...)
```

`main_cpu_bypass.py` keeps (1) and (3) untouched; only step (2) is
substituted:

```python
# === CPU BYPASS #N (was ttnn.matmul) ===
_torch_act_N  = _bypass_to_host_bf16(var_A)      # activation: gather + host bf16
_torch_wt_N   = _bypass_to_host_bf16(var_Wq)     # weight: on-device bfp4->bf16, gather, host
_cpu_result_N = torch.matmul(_torch_act_N, _torch_wt_N)
var_Y         = _bypass_from_host_bf16(_cpu_result_N, var_A)   # push bf16 back to device
```

Helpers injected at module scope:

```python
def _bypass_to_host_bf16(t):
    if t.dtype == ttnn.DataType.BFLOAT4_B:
        t = ttnn.typecast(t, ttnn.DataType.BFLOAT16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host = ttnn.from_device(t)
    shards = ttnn.get_device_tensors(host)
    # reshape to (mesh_rows, mesh_cols, *shard), detect shard dims by shape-diff
    # between neighbors, then torch.cat along those dims.
    ...

def _bypass_from_host_bf16(cpu_tensor, reference_device_tensor):
    return ttnn.from_torch(cpu_tensor, dtype=ttnn.DataType.BFLOAT16,
                           layout=ttnn.Layout.TILE,
                           device=reference_device_tensor.device(),
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
```

**Op-count delta per bfp4 matmul** (device run → bypass run):

| op | device run | bypass run |
|---|---|---|
| `ttnn.typecast(W, BFLOAT4_B)` (weight quant) | 1 | 1 (preserved) |
| `ttnn.typecast(Wq, BFLOAT16)` (weight dequant for CPU) | 0 | 1 (added) |
| per-chip shard gather (`get_device_tensors` + `torch.cat`) | 0 | 1 per operand |
| `ttnn.from_device` + `ttnn.to_torch` | 0 | 2 (added — activation + weight) |
| **core op** | `ttnn.matmul` on TT | `torch.matmul` on CPU |
| `ttnn.from_torch` | 0 | 1 (result back to device) |
| `ttnn.typecast(Y, BFLOAT16)` (output cast) | 1 | 1 (preserved) |

### Weight overrides under test

| pattern | dtype |
|---|---|
| `model.layers.*.mlp.router.weight` | `bf16` |
| `model.layers.*.mlp.experts.gate_up_proj` | `bfp_bf4` |
| `model.layers.*.mlp.experts.down_proj` | `bfp_bf4` |
| default | `bfp_bf8` |

Mesh: Galaxy 4×8 (`batch`, `model`). MoE expert sharding:
`gate_up_proj → (model, batch, None)`, `down_proj → (model, None, batch)`.

## Phase 1 — Layer 18 only

RAM-efficient setup: build a 1-layer GPT-OSS config, stream only the
safetensor shards that contain `model.layers.18.*`, dequantize mxfp4
`_blocks`/`_scales` pairs via
`transformers.integrations.mxfp4.convert_moe_packed_tensors`, remap keys
to `model.layers.0.*`, load into slot 0. Layers 0–17 and 19–35 are never
materialized.

Driver: `tests/benchmark/debug_bfp4_layer18.py`.

### Layer 18 — PCC vs CPU reference

Scripts:
- `tests/benchmark/run_cpu_reference_layer18.py` — bf16 forward pass on CPU.
- `tests/benchmark/pcc_vs_cpu_layer18.py` — target-shape-driven
  reconstruction of the 4×8 mesh-sharded outputs + fp64 chunked PCC.

_Note: only run tests against the real prompt
(`DEFAULT_INPUT_PROMPT`); zero-token input is meaningless for
PCC-vs-CPU (the model's output on zero tokens is dominated by
quantization noise, not real signal) and is **not** used in these
measurements._

With **tokenized `DEFAULT_INPUT_PROMPT`** (same prompt `llm_benchmark` uses) —
`tests/benchmark/bfp4_layer18_output_prompt/`:

| CPU reference entry | Shape | Device vs CPU | Bypass vs CPU |
|---|---|---|---|
| `logits` (full) | (64, 128, 201088) | 0.980842 | 0.978762 |
| `kv_keys_layer0` | (64, 8, 128, 64) | 0.999955 | 0.999955 |
| `kv_values_layer0` | (64, 8, 128, 64) | 0.999958 | 0.999958 |

Further narrowed to the **exact slices pytest checks**
(`logits[:, -1]` → last-token logits; `output_logits[0][0]` → first batch,
last token only):

| Slice | Shape | Device vs CPU | Bypass vs CPU |
|---|---|---|---|
| all batch, last token | (64, 201088) | 0.994985 | 0.993395 |
| pytest exact (batch[0], last token) | (201088,) | 0.994960 | 0.993412 |

With **pytest-matching inputs** (natural-length 17-token prefill,
`max_cache_len=128` — identical to what `construct_inputs` produces) —
`tests/benchmark/bfp4_layer18_output_pytest_match/`:

| CPU reference entry | Shape | Device vs CPU | Bypass vs CPU |
|---|---|---|---|
| `logits` (full) | (64, 17, 201088) | 0.917515 | 0.916364 |
| `kv_keys_layer0` | (64, 8, 128, 64) | 0.972123 | 0.972123 |
| `kv_values_layer0` | (64, 8, 128, 64) | 0.999319 | 0.999319 |

| Slice | Shape | Device vs CPU | Bypass vs CPU |
|---|---|---|---|
| all batch, last token | (64, 201088) | 0.998742 | 0.997456 |
| pytest exact (batch[0], last token) | (201088,) | 0.998744 | 0.997455 |

The pytest-exact slice at 0.998744 is still ~0.107 above the value
pytest itself reports (0.891743) for the same configuration. This gap
holds after matching input shape, input tokens, layer, weight
quantization, mesh, and sharding spec — so the codegen_py standalone
path and the live `_call_experimental_compile` path are numerically
different on the MoE graph. The codegen path yields higher PCC against
the same CPU reference than the pytest path does.

(Note the KV-cache PCC drop to 0.972 for keys is an artifact of
un-used cache slots 17..127 holding slightly different values on
device vs CPU — prefill only writes positions 0..16.)

### Layer 18 — pytest reference (`test_gpt_oss_120b_tp_galaxy_batch_size_64` with `layer_index=18`)

**PCC = 0.891743** (required_pcc for this test is 0.93, so the test reports FAILED).

Command:
```
pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 \
       --layer-index 18
```



Attempted:

```
pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 \
       --layer-index 18
```

Required changes (applied in this repo):

1. `tests/benchmark/test_llms.py` — added `layer_index` parameter to
   `test_gpt_oss_120b_tp_galaxy_batch_size_64` (threaded through
   `test_llm_tp` → `test_llm` → `create_model_loader` → loader kwarg).
2. `third_party/tt_forge_models/gpt_oss/pytorch/loader.py` (pinned
   submodule doesn't have `layer_index` upstream) — added `layer_index`
   kwarg to `ModelLoader.__init__`. When set, the loader builds a
   1-layer config (`num_hidden_layers = 1`), streams only the
   safetensor shards that contain `model.layers.<target_layer>.*`,
   dequantizes mxfp4 `_blocks`/`_scales` pairs via
   `transformers.integrations.mxfp4.convert_moe_packed_tensors`, and
   remaps the keys onto slot 0. Avoids materializing the other 35
   layers' weights (~18 GB bf16 per layer). Matches the load profile
   verified in the pytest log:
   `Loading weights: 100%|██████████| 20/20` = lm_head + embed_tokens +
   norm + 17 params for layer 0.

Baseline sanity run (no `--layer-index`, `--num-layers 1`): the same
test with layer 0's weights completes with **PCC = 0.998588** against
the CPU reference. This confirms the test infrastructure works end-to-end
on this host; the 3-stage compile pipeline ( torch_xla →
`_call_experimental_compile` → tt-mlir ) is functional. The `layer_index`
parameter and streaming loader patch are the only things that need to
work for us to capture the layer-18 number.

Initial `--layer-index 18` attempts crashed inside `bridge.extract_compiled_graph`
with `llvm::raw_fd_ostream preferred_buffer_size(): FD >= 0 && "File not
yet open!"`. The cause was my first loader patch going through a
different build path (`config=one_layer_config` set inside `load_model`
instead of via `load_config`'s `num_layers` hook, plus an `assign=False`
state-dict load that left remapped tensors in non-contiguous
non-matching-dtype shape). Rewrote the patch so the load is identical to
`--num-layers 1` (which works) and the layer-18 weights are copied in
after the standard `from_pretrained` returns, with dtype + contiguity
matched to the target parameters. See loader.py diff in the submodule.

Layer-18 PCC number from the pytest path:

| Run | PCC vs CPU reference | Required PCC | Result |
|---|---|---|---|
| `--num-layers 1` (layer 0, bfp4 MLPs) | **0.998588** | 0.93 | PASSED |
| `--layer-index 18` (bfp4 MLPs) | **0.891743** | 0.93 | FAILED |

## Phase 2 — Whole model (36 layers)

### Full model — pytest reference (`test_gpt_oss_120b_tp_galaxy_batch_size_64`)

**PCC = 0.949253** (required_pcc = 0.93, test PASSED).

Command:
```
pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64
```

### Full model — CPU bypass driver

Driver: `tests/benchmark/debug_bfp4_full_model.py`.

Same pipeline as layer 18 but builds the full 36-layer model via
`llm_benchmark.setup_model_and_tokenizer`. Every bfp4 matmul
(72 = 2 per layer × 36) is replaced by a CPU torch.matmul.

Codegen artefacts already produced:
- `bfp4_full_output/main.py` (1.9 MB)
- `bfp4_full_output/main_cpu_bypass.py` (1.9 MB) — 72 bfp4 matmul groups
- `bfp4_full_output/tensors/` — 219 GB of real HF weights

### Results — pytest `test_gpt_oss_120b_tp_galaxy_batch_size_64` (no `--layer-index`, full 36 layers)

| Run | PCC vs CPU reference | Required PCC | Result |
|---|---|---|---|
| Full model, bfp4 MoE experts | **0.949253** | 0.93 | PASSED |

### Results — generated `main.py` + `main_cpu_bypass.py` vs CPU reference

Running the full-model codegen artefacts (`bfp4_full_output/main.py`
and `bfp4_full_output/main_cpu_bypass.py`) in a subprocess and comparing
against a pure-CPU reference (same inputs, bf16 weights, no bfp4, no TT):

**Device run (`main.py`)** — **OOM on device**:

```
TT_FATAL: Out of Memory: Not enough space to allocate 4246732800 B DRAM
buffer across 12 banks, where each bank needs to store 353894400 B,
but bank size is 1071821792 B (allocated: 721704256 B, free: 350117536 B,
largest free block: 303610688 B)
  bank_manager.cpp:439
```

The codegen_py-generated `main.py` tries to materialise a ~4.2 GB
DRAM tensor when per-chip DRAM is already 721 MB full, and it can't
fit. The same 36-layer configuration runs successfully via the pytest
compile path (0.949253 above), so this OOM is specific to the codegen
path — it is **not** memory-equivalent to the compiled model.

**Retries** trying to fit the codegen path in DRAM:

| Attempt | Config | Outcome |
|---|---|---|
| 1 | `batch=64`, `optimization_level=1` (matches pytest) | OOM at 4.2 GB DRAM allocation |
| 2 | `batch=16`, `optimization_level=1` | **Same OOM, same exact 4.2 GB allocation size** |
| 3 | `batch=16`, `optimization_level=2` | **Same OOM, same exact 4.2 GB allocation size** |

The identical allocation across all three attempts shows the failing
buffer is unrelated to batch (activations scale with batch, this
doesn't) and unaffected by the compiler's optimisation level. It's a
fixed per-chip allocation the codegen emits — likely during
mesh-partition of the replicated `lm_head`/`embed_tokens` weights
(201088 × 2880 bf16 ≈ 1.16 GB each, replicated across all 32 chips,
and the intermediate gather expands the live set further).

**Bypass run (`main_cpu_bypass.py`)** — not run; same
`load_inputs_for__main()` path would OOM identically.

The pytest compile path at `batch=64` fits because its runtime
streams / frees DRAM op-by-op; the codegen_py-emitted `main.py`
allocates everything upfront. The two paths are therefore **not
memory-equivalent**, and a PCC-vs-CPU comparison on the full-model
codegen artefacts cannot be produced without reworking the generated
`main.py`'s weight-load strategy (or regenerating with a different
sharding that shards the embedding/lm_head weights instead of
replicating them).

## Tweaks / patches applied

Each issue encountered and the fix.

### P1.1 — `_experts_implementation = "dense"` rejected before `from_pretrained`

**Symptom:** `ValueError: Specified 'experts_implementation="dense"' is not
supported. The only possible arguments are 'eager', 'grouped_mm',
'batched_mm'.` Raised from `transformers/modeling_utils.py:1903` during
`AutoModelForCausalLM.from_pretrained(config=config, …)`.

**Cause:** newer `transformers` validates the config string during
`__init__`; `"dense"` is an internal TT-side value that must be applied
*after* the model is constructed. `llm_benchmark.setup_model_and_tokenizer`
does this post-load.

**Fix:** set `model.config._experts_implementation = "dense"` **after**
`from_pretrained` returns.

### P1.2 — Root-owned `generated/watcher/kernel_names.txt` blocks fopen

**Symptom:** `TT_THROW: Watcher failed to create kernel name file` from
`watcher_server.cpp:330`, followed by the atexit crash
`Check failed: !g_computation_client_initialized`.

**Cause:** tt-metal writes `<cwd>/generated/watcher/kernel_names.txt` at
program start. The Docker image ships a root-owned `generated/` tree.
`fopen` fails for uid 2036; the exception surfaces during module init, and
the torch_xla shutdown path re-enters `GetComputationClient()` after the
client was killed, tripping the "can only be initialized once" DCHECK.

**Fix:** `chown -R dgolubovic:users /home/dgolubovic/repos/tt-xla/generated`
(from outside the container). Also relaxed perms on the in-tree
`tt-metal/generated/` and `tt-metal/.watcher/`. Side note: setting
`TT_METAL_WATCHER=disabled` actually *enables* the watcher — the parser
unconditionally sets `enabled=true` whenever the var is set. Leave
unset to keep watcher off.

### P1.3 — Codegen inlines `main_const_eval_N` into `main.py`, no `consteval.py`

**Symptom:** `bfp4 matmul groups found: 0`.

**Cause:** `run_alchemist_cpu_bypass` expected two files (`consteval.py`
for `typecast → BFLOAT4_B`, `main.py` for the forward pass). The current
`tt_torch.codegen_py` emits everything inlined into `main.py`; with
`consteval.py` missing, `find_bfp4_consteval_functions("")` returned the
empty set.

**Fix:** in `run_alchemist_cpu_bypass`, when `consteval.py` is missing,
pass `main_code` itself as `consteval_code`. After the fix the parser
finds 2 bfp4 matmul groups for layer 18 and 72 for the full model.

### P1.4 — Generated `utils.DeviceGetter` hard-codes `FABRIC_1D`, breaks on 4×8 mesh

**Symptom:** `TT_FATAL: Could not find any forwarding direction from src
(M0, D0) to dst (M0, D28)` from `tt_metal/fabric/fabric.cpp:153`.

**Cause:** tt-alchemist emits a generic `utils.py` whose `DeviceGetter`
always calls `ttnn.set_fabric_config(FabricConfig.FABRIC_1D)`. On the 4×8
Galaxy mesh, multi-hop all_gather requires a richer topology.

**Fix:** patch the generated `utils.py` to use
`FabricConfig.FABRIC_1D_RING` on multi-chip meshes. Matches what the
tt-mlir PJRT runtime auto-selects
(`runtime/lib/common/mesh_fabric_config.cpp::classifyAxis` picks
`FABRIC_1D_RING` when the axis's physical chip channels form a ring —
the Galaxy does). `FABRIC_2D` caused a deadlock on this codegen path.

### P1.5 — `ttnn.to_torch` on multi-device sharded tensor needs a mesh_composer

**Symptom:** `TT_FATAL: Can't convert a tensor distributed on
MeshShape([4, 8]) mesh to row-major logical tensor. Supply a mesh
composer to concatenate multi-device shards.`

**Cause:** `ttnn.to_torch(host_tensor)` on a multi-buffer host tensor
cannot infer the shard layout. The generated `main.py` never had to
gather outputs (its `main()` just calls `_main(...)` and discards the
result), so no composer was ever emitted.

**Fix (driver side):** use `ttnn.get_device_tensors(host_tensor)` to get
per-chip shards and `torch.stack([...], dim=0)`.

**Fix (bypass helpers):** use `ttnn.get_device_tensors` + target-shape-
driven reconstruction inside `_bypass_to_host_bf16` (detect row/col
shard dims by checking `per_chip[d] * {4, 8} == target[d]`; `torch.cat`
along the sharded dim; take any replica on non-sharded axes).
Earlier variants that used in-graph `ttnn.all_gather` on both cluster
axes OOM'd because `dim=-1` duplicated data along non-shard dims (17 GB
per chip observed for the MoE `gate_up_proj`).

### P1.6 — Stale device state → `Read unexpected run_mailbox value from core`

**Symptom:** Hundreds of `TT_FATAL: Read unexpected run_mailbox value
from core (x=25,y=17)` before the usual init.

**Cause:** previous crashed run left firmware on one of the 32 chips
in an inconsistent state.

**Fix:** `/home/dgolubovic/.local/bin/tt-smi -glx_reset_auto` from the
host — resets all 32 Galaxy ASICs and re-initialises (~35 s). Safe
under exclusive SLURM allocation.
