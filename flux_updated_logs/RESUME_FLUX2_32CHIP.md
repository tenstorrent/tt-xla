# FLUX.2 transformer bring-up — RESUME HANDOFF (32-chip Galaxy)

Branch: `akannan/bringup_flux2`. Submodule `third_party/tt_forge_models` @ `5e64767ff3`
(branch `akannan/fix_flux2_encoder_oom`). Machine when paused: 32-chip Galaxy.
Paused: 2026-06-16. State at pause: all jobs killed, `tt-smi -r` done, 32 devices idle & healthy.

## Where we are (read these in order)
1. `HANDOFF.md` — original bring-up plan + env setup (caches off /home, HF token, tt-smi).
2. `FIX_RESULTS.md` — mesh `(1,32)` fix made text_encoder PASS; transformer still failed (then OOM).
3. `TRANSFORMER_PCC_DIAGNOSIS.md` — AdaLN modulation shard-spec was mis-sharded → PCC collapse;
   fixed (submodule `5e64767ff3`). Concluded 8-chip can't reach 0.99 → fall back to 32-chip bf16.
4. `CHIP32_DEADLOCK_FINDINGS.md` — **THE CURRENT BLOCKER (read this).**

## The current blocker (NEW, 2026-06-16)
The 32-chip transformer **deadlocks in device execution** — host spins forever in
`FDMeshCommandQueue::read_completion_queue → completion_queue_wait_front`. Reproduces even at
**NL=2/NS=2**, so it is NOT depth / bfp8 / OOM / shard-spec. It's a **CCL ring deadlock**: the
`(1,32)` logical mesh emits a 32-way `reduce_scatter` 1D ring that can't map to the physical
`(4,8)` Galaxy (startup warns `client_instance.cc:417 ... 1D {1,N}->(4,8) reshape fails in the
MGD solver`, tt-metal#43210).

Both 32-chip mesh configs are upstream-blocked:
- `MESH_SHAPES[32]=(1,32)` (current): compiles, **runtime deadlock** (tt-metal#43210).
- `MESH_SHAPES[32]=(8,4)` (old): **compile fail** `CollectivePermuteOp not implemented` (tt-mlir#3370).
`MESH_SHAPES` lives in submodule `flux2/pytorch/src/model_utils.py:57`. Text encoder PASSES on `(1,32)`.

## Component status snapshot
| Component | 32-chip status |
|-----------|----------------|
| text_encoder_sharded | PASS on `(1,32)` (fixed by submodule `ac2212ffdf`) |
| vae_decoder | single-chip, separate |
| transformer_sharded | **BLOCKED — CCL deadlock (this doc)** |

## Open options (decision pending — none done yet)
1. **Fabric/topology config**: make a `(1,32)` ring valid on Galaxy (tt-metal fabric config:
   1D line/ring vs 2D torus; mesh device ordering). Cheap but speculative.
2. **2D sharding**: mesh `(4,8)`/`(8,4)`, shard weight dim over BOTH axes
   (`(("batch","model"), None)`) for 32-way via per-axis collectives. Check SHLO for
   collective_permute (tt-mlir#3370) BEFORE running on HW.
3. **Escalate upstream**: attach repros to tt-metal#43210 + tt-mlir#3370, pause until fixed.

## Repro / isolation harnesses (committed on this branch)
- `tests/torch/models/flux2/test_transformer_realwt_isolate.py` — REAL weights, truncated depth.
  Env: `FLUX_NL`, `FLUX_NS`, `FLUX_WDTYPE` (""=bf16 | "bfp_bf8"), `FLUX_SHARDED`, `FLUX_SHARD_MODE`
  (all|blocks_only|mod_only).
- `tests/torch/models/flux2/test_transformer_depth_isolate.py` — random weights, depth sweep.
- `tests/torch/models/flux2/test_transformer_tiny_isolate.py` — minimal.

## EXACT resume recipe (on the Galaxy)
```bash
cd /proj_sw/user_dev/ctr-akannan/15_june_yyz/tt-xla
source .flux_env.sh && source venv/activate     # .flux_env.sh has HF token; gitignored — recreate if missing (see HANDOFF.md)
tt-smi -r                                        # ALWAYS reset first; on this Galaxy prints a CPLD warning but works (fallback: tt-smi -glx_reset)
python -c "import torch_xla.runtime as xr; xr.set_device_type('TT'); print(xr.global_runtime_device_count())"  # expect 32
# minimal deadlock repro (hangs today — ALWAYS use timeout + tee; never leave a bare pytest detached):
rm -rf generated/inspector
timeout 1200 env FLUX_NL=2 FLUX_NS=2 FLUX_WDTYPE="" FLUX_SHARDED=1 FLUX_SHARD_MODE=all \
  python -m pytest -svv tests/torch/models/flux2/test_transformer_realwt_isolate.py::test_realwt \
  2>&1 | tee flux_updated_logs/realwt_32chip_nl2_ns2_bf16.log
```
## Hang-detection cheatsheet (learned this session)
- High CPU + frozen `generated/inspector/*.yaml` mtimes = device hang, not slow compute.
- `gdb -p <pid> -batch -ex "thread apply all bt 4" | grep completion_queue_wait_front` → if it hits,
  it's the CCL deadlock. Kill + `tt-smi -r`.
- Find the python pid (pytest may be a child of a bash wrapper): `pgrep -f "pytest.*realwt"`.

## Gotchas
- `.flux_env.sh` contains an HF token — NEVER commit it (now in .gitignore).
- Keep caches off `/home` (9.4G mount); `.flux_env.sh` redirects TT_METAL_CACHE/HF_HOME to /proj_sw.
- `tmp/`, `generated/`, `build_log.log`, `tt_snap*.json` are local junk (now gitignored).
