# FLUX.2 Bringup — Work Handoff (paste into new chat after reserving a proper machine)

## Goal
On branch `akannan/bringup_flux2`, comment out all `@pytest.mark.xfail` (and `@pytest.mark.skip`)
markers on the FLUX.2 component tests, run every component against latest `main` to see its true
status, store run logs under `flux_updated_logs/`, run them **sequentially** (single-chip and
multi-chip), then list the failures and proceed with fixes.

## ⚠️ Why work stopped — WRONG HARDWARE
The reserved machine (`wh-lb-44-special-ctr-akannan-...`) is **4 standalone n300 cards (8 chips
total) that are NOT cabled together into a T3000 mesh**. Evidence:
- `tt-smi` snapshot shows all 4 "L" chips with identical mesh coords `(0,0,0,0)` and all 4 "R"
  chips with `(1,0,0,0)` → each n300 is an independent 1x2 mesh, not a 2x4 T3K.
- Fabric/control-plane init across all 8 chips fails on EVERY run (even single-device tests,
  because the PJRT plugin initializes the full fabric at startup):
  `TT_FATAL ... Physical chip id N not found in control plane chip mapping ...
   control_plane.cpp:1264` for chips 0–7.
- **Proof it's a topology problem, not corruption:** `TT_VISIBLE_DEVICES=0` (restrict to ONE
  n300 card) → `global_runtime_device_count() == 2` succeeds. Spanning all 8 fails.

**Action: reserve a true T3000 (8 interconnected chips) or Galaxy.** The FLUX.2 sharded tests
need 8 chips in ONE mesh. This branch already contains logs proving the tests run on such HW
(`flux2_dev_32b_logs/`, `flux2_dev_32b_logs_galaxy/`).

## Tests in scope (`tests/torch/models/flux2/`)
| File | Test | Marker on branch | Needs |
|------|------|------------------|-------|
| test_text_encoder.py | `test_text_encoder` | `skip` (24B exceeds single-chip DRAM) | n/a |
| test_text_encoder.py | `test_text_encoder_sharded` | `xfail` (PCC 0.9684 < 0.99) | 8 chips, tensor_parallel |
| test_transformer.py | `test_transformer` | `skip` (32B exceeds single-chip DRAM) | n/a |
| test_transformer.py | `test_transformer_sharded` | `xfail` (OOM DRAM, bank_manager.cpp:462) | 8 chips, tensor_parallel |
| test_vae_decoder.py | `test_vae_decoder` | none (single_device, nightly) | 1 chip |

The xfail reasons to re-verify:
- `test_text_encoder_sharded`: "PCC comparison failed: pcc=0.9684 vs required 0.99" — compiles/runs e2e on 8 chips, numerical accuracy gap.
- `test_transformer_sharded`: "Out of Memory: cannot allocate 56623104 B DRAM buffer across 12 banks ... TT_FATAL bank_manager.cpp:462" — 32B transformer still DRAM-bound when sharded over 8 chips.

## Environment setup ALREADY DONE on this checkout (re-do on new machine if it's a fresh workspace)
Working dir: `/proj_sw/user_dev/ctr-akannan/15_june_yyz/tt-xla`

1. **Branch**: `git checkout akannan/bringup_flux2` (done). Submodule
   `third_party/tt_forge_models` @ `09239ae` (heads/main).
2. **Home dir was 100% full (9.4G).** Caches redirected to `/proj_sw` (2.1T free) via
   `.flux_env.sh` (created at repo root, gitignored-by-intent — see below). Moved old uv cache
   off home.
3. **venv was incomplete** (only numpy + plugin). Installed both requirement sets:
   `uv pip install -r python_package/requirements.txt`
   `uv pip install -r venv/requirements-dev.txt`  (torch 2.10.0+cpu, torch-xla 2.9.0+git5c82b10,
   jax 0.7.1, transformers 5.2.0, diffusers, etc. — all installed OK)
4. **tt-smi installed** into venv: `uv pip install git+https://github.com/tenstorrent/tt-smi.git`
   (system `~/.local/bin/tt-smi` was broken — missing module). Use `tt-smi -r` to reset.
5. **HF gated access**: token set in `.flux_env.sh` (FLUX.2 is gated; download confirmed working).

### `.flux_env.sh` contents (recreate at repo root on new machine)
```bash
export UV_CACHE_DIR=/proj_sw/user_dev/ctr-akannan/.cache/uv
export PIP_CACHE_DIR=/proj_sw/user_dev/ctr-akannan/.cache/pip
export HF_HOME=/proj_sw/user_dev/ctr-akannan/.cache/huggingface
export HF_TOKEN=<YOUR_HF_TOKEN>            # redacted — set your own gated-access token
export HUGGING_FACE_HUB_TOKEN=<YOUR_HF_TOKEN>
export UV_INDEX_STRATEGY=unsafe-best-match
export TMPDIR=/proj_sw/user_dev/ctr-akannan/.cache/tmp
mkdir -p "$TMPDIR"
```
(Keep caches off `/home` — it is a 9.4G weka mount that fills instantly.)

## RESUME PLAN on the new (true T3000/Galaxy) machine
1. Verify topology is a real mesh BEFORE anything else:
   ```bash
   source .flux_env.sh; source venv/activate
   tt-smi -s > /tmp/snap.json
   python3 -c "import json;[print(i,x['board_info']['coords']) for i,x in enumerate(json.load(open('/tmp/snap.json'))['device_info'])]"
   # On a real T3K the 8 chips have DISTINCT mesh coords, not all (0,0,0,0)/(1,0,0,0).
   tt-smi -r   # reset before runs
   python -c "import torch_xla.runtime as xr; xr.set_device_type('TT'); print(xr.global_runtime_device_count())"
   # Expect 8 with no 'chip id not found in control plane' error.
   ```
2. Comment out the `@pytest.mark.skip` and `@pytest.mark.xfail` decorators on all 5 tests in
   `tests/torch/models/flux2/` so true status is observed (NOT YET DONE — edits were not made
   because the HW couldn't run anything).
3. Run sequentially, one log per test, into `flux_updated_logs/`. Reset (`tt-smi -r`) between
   tests to avoid stale `run_mailbox` state (we observed crashes leave chips dirty):
   ```bash
   # single chip
   tt-smi -r; python -m pytest -svv tests/torch/models/flux2/test_vae_decoder.py 2>&1 | tee flux_updated_logs/test_vae_decoder.log
   tt-smi -r; python -m pytest -svv tests/torch/models/flux2/test_text_encoder.py::test_text_encoder 2>&1 | tee flux_updated_logs/test_text_encoder_single.log
   tt-smi -r; python -m pytest -svv tests/torch/models/flux2/test_transformer.py::test_transformer 2>&1 | tee flux_updated_logs/test_transformer_single.log
   # multi chip (8)
   tt-smi -r; python -m pytest -svv tests/torch/models/flux2/test_text_encoder.py::test_text_encoder_sharded 2>&1 | tee flux_updated_logs/test_text_encoder_sharded.log
   tt-smi -r; python -m pytest -svv tests/torch/models/flux2/test_transformer.py::test_transformer_sharded 2>&1 | tee flux_updated_logs/test_transformer_sharded.log
   ```
4. Collect PASSED/FAILED + root-cause error per test, then fix (text_encoder_sharded = PCC gap;
   transformer_sharded = DRAM OOM).

## Notes / gotchas observed
- A crashing python run leaves tensix cores dirty → `Read unexpected run_mailbox value from core
  25-17 (assert.hpp:104)`. Always `tt-smi -r` before the next run.
- Plugin prints `InitializeComputationClient() can only be called once` at process teardown after
  a failure — harmless, it's during finalize.
- `flux_updated_logs/test_vae_decoder.log` here contains the FAILED run from the wrong machine
  (control-plane error) — overwrite it on the new machine.
