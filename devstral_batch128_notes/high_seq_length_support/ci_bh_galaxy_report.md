# Running `test_dptp_devstral` on Blackhole Galaxy in CI

**Question:** Can we run
`tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral`
on **Blackhole (BH) galaxy** hardware in CI as a plain "run test" job, and is BH-galaxy CI even available?

## TL;DR

**Yes — BH-galaxy CI exists and the job that would run this test is already on `main`. The only reason it isn't running today is that the test itself is not yet merged to `main` (it lives on this WIP branch). No marker change is needed.**

- The BH-galaxy runner label is **`galaxy-bh`**. It is actively used across workflows (nightly-experimental test job, perf benchmarks, and the manual single-test dispatch menu). **All of the CI plumbing below is confirmed present on `origin/main`.**
- There is already a nightly CI job on `main` named **`run_vllm_bh_galaxy`** that runs the entire `tests/integrations/vllm_plugin` directory on `galaxy-bh` with the mark filter **`nightly and tensor_parallel and bh_galaxy`**.
- `test_dptp_devstral` carries exactly the markers that filter selects (`nightly`, `tensor_parallel`, and `bh_galaxy` on its `mesh_shape=[4,8]` param). **No marker needs to be added or changed.**
- **BUT the test file `test_data_tensor_parallel_generation.py` (all three DP+TP tests, including `test_dptp_devstral`) is NOT on `origin/main` yet** — it is uncommitted/WIP on branch `ssalice/devstral-qwen-wip-07-13-2026`. Because the scheduled nightly runs `main`, it does **not** currently collect this test. Two consequences:
  - **Once this branch merges to `main`, the existing `run_vllm_bh_galaxy` job picks the test up automatically** (no workflow or marker change required).
  - **To run it now, dispatch `manual-test-single.yml` against this WIP branch** (workflow_dispatch runs whatever ref you select) — see §6.
- The remaining caveat is **not** CI plumbing but **the 123B Devstral model**: HF token + shared HF cache are wired, and the model exists on HF (`mistralai/Devstral-2-123B-Instruct-2512`, 125B params, **`license:other` → gated**, fp8). But it is large/gated and `num_hidden_layers=2` does **not** shrink the HF *download* (it only truncates compilation). Feasibility depends on the model being pre-cached and the HF token having accepted-license access.

Important distinction: **`galaxy-bh` (Blackhole galaxy) ≠ `galaxy-wh-6u` (Wormhole galaxy 6U) ≠ `qb2-blackhole` (Blackhole quiet-box, 4-chip p300c).** The *production* nightly runs vLLM on `galaxy-wh-6u` and `qb2-blackhole`, but the BH-galaxy vLLM job lives only in the **experimental** nightly.

---

## 1. CI workflows and hardware/runner selection

Tests run through the reusable workflow **`.github/workflows/call-test.yml`**, driven by "preset" JSON files in `.github/workflows/test-matrix-presets/`. Each preset entry declares a `runs-on` (runner label), `dir` (pytest path), and `test-mark` (pytest `-m` expression). The matrix is expanded by `.github/scripts/generate_test_matrix.py`.

Runner labels that appear in the repo (the `runs_on` choice list in `manual-test-single.yml:26-36` is the authoritative menu):
- `n150`, `n300`, `p150` — single-/dual-chip Wormhole/Blackhole cards
- `n300-llmbox`, `llmbox-1`, `llmbox-2` — Wormhole llmbox
- **`galaxy-wh-6u`** — Wormhole galaxy (6U)
- **`galaxy-bh`** — **Blackhole galaxy** (the target hardware)
- **`qb2-blackhole`** — Blackhole quiet-box (4-chip p300c)
- `lb-blackhole` — Blackhole loudbox
- `cpu` — CPU-only shared runner

Workflows that dispatch tests to **`galaxy-bh`**:
- `.github/workflows/schedule-nightly-experimental.yml:52-62` — job **`test_bh_galaxy`**, cron `0 4 * * *` + `workflow_dispatch`, uses preset `basic-test-nightly-experimental.json`.
- `.github/workflows/manual-test-single.yml:34` — `galaxy-bh` is a selectable `runs_on` option (the "run one test" path).
- `.github/workflows/manual-benchmark.yml:23,25` and `.github/workflows/perf-bench-matrix.json:762,772` — perf/benchmark jobs on `galaxy-bh`.

**BH galaxy IS available in CI.** The `galaxy-bh` label is wired into scheduled, manual, and benchmark workflows. (Note: this repo cannot prove the runner is *online/healthy* right now — only that the label is provisioned and referenced; that must be confirmed in the GitHub Actions runner pool.)

---

## 2. How tests are selected / routed to a board

- **Marker definitions** live in `pytest.ini` (not `pyproject.toml`). Relevant markers:
  - `push`, `nightly` — pipeline selectors
  - `single_device`, `data_parallel`, `tensor_parallel` — parallelism
  - `dual_chip` (n300), `llmbox`, `galaxy`, **`bh_galaxy`** ("marks test for blackhole galaxy (galaxy-bh)", `pytest.ini:41`), `bhqb` (BH quiet-box)
- **Routing = preset entry pairs a `runs-on` label with a `test-mark` expression.** A test reaches a given board when (a) it has the markers matching that entry's `-m` expression and (b) the entry's `runs-on` points at that board. There is no separate manifest — the marker + preset pair is the routing.
- `call-test.yml` builds the pytest command from the entry: mark expression at `call-test.yml:317-320` (`-m '<mark>'`), pytest run at `call-test.yml:414`. `extra-wheel: "vllm"` triggers download+install of the vLLM wheel (`call-test.yml:189-201`).
- The BH-galaxy vLLM entry (`basic-test-nightly-experimental.json:3`):
  ```json
  { "runs-on": "galaxy-bh", "name": "run_vllm_bh_galaxy",
    "dir": "./tests/integrations/vllm_plugin",
    "test-mark": "nightly and tensor_parallel and bh_galaxy", "extra-wheel": "vllm" }
  ```
  So **to be picked up on BH galaxy, a vLLM test needs markers `nightly` + `tensor_parallel` + `bh_galaxy`.**

---

## 3. The target test's current markers

`tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py`, `test_dptp_devstral` (def at line 308), decorators at lines 293-307:

```python
@pytest.mark.nightly                 # 293
@pytest.mark.data_parallel           # 294
@pytest.mark.tensor_parallel         # 295
@pytest.mark.parametrize(["enable_const_eval","experimental_weight_dtype"],
                         [pytest.param(True, "bfp_bf8")])          # 296-301
@pytest.mark.parametrize("mesh_shape",
    [pytest.param([4, 8], marks=pytest.mark.bh_galaxy)])           # 302-306
```

The single `mesh_shape` param `[4, 8]` is marked `bh_galaxy`, so the collected instance carries **`nightly`, `data_parallel`, `tensor_parallel`, `bh_galaxy`**. There is **no** `record_test_properties` on this test (unlike model tests).

**This already matches `nightly and tensor_parallel and bh_galaxy`.** No marker needs to be added or changed. The existing `run_vllm_bh_galaxy` job (on `main`) *will* select `test_dptp_devstral` — **once the test is merged to `main`**. It is not on `main` today (see §6), so the scheduled job does not collect it yet; it is runnable now only via a manual dispatch pointed at this branch.

(Note it is *not* selected by the *production* nightly, which routes vLLM only to `galaxy-wh-6u`/`qb2-blackhole` — see §4/§6.)

---

## 4. vLLM plugin CI

Yes — vLLM plugin CI exists and installs the vLLM wheel.

- Install path: preset entries set `"extra-wheel": "vllm"`; `call-test.yml:189-201` downloads the `vllm-tt-whl-release` artifact and installs it. Requirements hashed/cached from `integrations/vllm_plugin/requirements-vllm-plugin.txt` (`call-test.yml:133`). (Note: CI installs the *built vLLM wheel*, not `pip install -e integrations/vllm_plugin`.)
- Presets that run `./tests/integrations/vllm_plugin`:
  - `test-matrix-presets/vllm-model-tests.json` (push): n150, p150, n300, n300-llmbox
  - `test-matrix-presets/vllm-model-tests-nightly.json` (nightly): adds `galaxy-wh-6u` (mark `galaxy_wh_6u`) and `qb2-blackhole` (mark `bhqb`)
  - `test-matrix-presets/basic-test-nightly.json:17,20` (production nightly): vLLM on `qb2-blackhole` (`bhqb`) and `galaxy-wh-6u` (`galaxy_wh_6u`)
  - **`test-matrix-presets/basic-test-nightly-experimental.json:3`**: vLLM on **`galaxy-bh`** (`bh_galaxy`) — the only BH-galaxy vLLM entry.

So the vLLM integration is fully wired into CI; the BH-galaxy variant is in the **experimental** nightly (`schedule-nightly-experimental.yml`), not the production nightly (`schedule-nightly.yml:37`).

---

## 5. Model download / size (the real gate)

- **HF access is wired:** `call-test.yml:405-408` sets `HF_HOME=/mnt/dockercache/huggingface` (shared, persistent cache), `TORCH_HOME=/mnt/dockercache/torchhub`, and `HF_TOKEN=${{ secrets.HF_TOKEN }}`.
- **`num_hidden_layers=2` does NOT reduce the HF download.** `apply_hidden_layer_override` (`integrations/vllm_plugin/vllm_tt/vllm_utils.py:117-164`) only mutates the in-memory config so compilation builds 2 layers; the vLLM weight loader still resolves the full HF snapshot (all safetensors shards of `mistralai/Devstral-2-123B-Instruct-2512`). Only 2 layers' tensors are loaded into memory, but the ~123B repo must be present in the HF cache. So feasibility hinges on the model already being cached at `/mnt/dockercache/huggingface`, or a large one-time download.
- **Gating:** the model exists and is public-but-license-gated. HF confirms `mistralai/Devstral-2-123B-Instruct-2512` — **125B params (125026M), `license:other` (gated → requires license acceptance + token), fp8, arch `ministral3`, updated 15 Jul 2026, ~295K downloads**. The CI `HF_TOKEN` must have accepted the license for this repo. (For comparison, the sibling test's `Qwen/Qwen3-32B` is apache-2.0, ungated.) Note the model is natively **fp8** on HF — consistent with the plugin's fp8→bf16 dequant hook.
- **Host-memory assertion is a non-blocker:** `check_host_memory` (`tests/integrations/vllm_plugin/generative/conftest.py:115-151`) only asserts for models present in its `model_rss_limits_gb` dict; Devstral is not listed, so the threshold is `None` and the assert is skipped.

---

## 6. Verdict and concrete gaps

**Can we run `test_dptp_devstral` on BH galaxy in CI?** Yes — the BH-galaxy vLLM job exists on `main` and the test's markers already match it. The only thing standing between "it exists" and "it runs in the scheduled nightly" is that **the test file is not yet merged to `main`**.

**Path A — run it now (no merge needed):** dispatch `.github/workflows/manual-test-single.yml` **with the branch/ref set to `ssalice/devstral-qwen-wip-07-13-2026`** (workflow_dispatch runs the selected ref, and the test only exists there):
   - `runs_on = galaxy-bh`
   - `dir = ./tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral`
   - `mark = bh_galaxy` (or leave blank; the node id already pins the test)
   - `install_vllm_wheel = true`

**Path B — scheduled nightly (after merge):** once this branch merges to `main`, the existing `run_vllm_bh_galaxy` job in `schedule-nightly-experimental.yml` (whole `vllm_plugin` dir, `nightly and tensor_parallel and bh_galaxy`) will collect it automatically. **No workflow or marker edit is required — just merge.**

**Gaps / things to confirm before it passes green:**

1. **Merge to `main`** — the test file `test_data_tensor_parallel_generation.py` (incl. `test_dptp_devstral`) is WIP on this branch and absent from `origin/main`; the scheduled job can't run what isn't on `main`. *(Confirmed via `git show origin/main:...` — file not found.)* The CI plumbing (`run_vllm_bh_galaxy`, `test_bh_galaxy`, `galaxy-bh` preset entry) **is** already on `main`.
2. **`galaxy-bh` runner online/healthy** in the Actions runner pool (repo only proves the label is wired).
3. **Model access:** `mistralai/Devstral-2-123B-Instruct-2512` exists but is `license:other` (gated). The CI `HF_TOKEN` must have accepted the license, and the ~125B repo should ideally be **pre-populated in the shared HF cache** `/mnt/dockercache/huggingface` so the job doesn't attempt a large cold download. `num_hidden_layers=2` will not save the download.
4. **If you want it in the *production* nightly** (not just experimental), add a `galaxy-bh` vLLM entry to `basic-test-nightly.json` mirroring `basic-test-nightly-experimental.json:3`, and add a `test_bh_galaxy` job to `schedule-nightly.yml` — or promote the experimental job. As-is, the production nightly only covers `galaxy-wh-6u` and `qb2-blackhole` for vLLM.

---

## Key file:line citations

- Marker defs incl. `bh_galaxy`: `pytest.ini` (markers block; `bh_galaxy` = "blackhole galaxy (galaxy-bh)")
- Target test markers: `tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py:293-308`
- BH-galaxy vLLM preset entry: `.github/workflows/test-matrix-presets/basic-test-nightly-experimental.json:3`
- BH-galaxy scheduled job: `.github/workflows/schedule-nightly-experimental.yml:52-62`
- `galaxy-bh` in manual single-test menu: `.github/workflows/manual-test-single.yml:34`
- vLLM wheel install in CI: `.github/workflows/call-test.yml:189-201`
- pytest mark + run: `.github/workflows/call-test.yml:317-320,414`
- HF token/cache env: `.github/workflows/call-test.yml:405-408`
- Production nightly preset (no galaxy-bh vLLM): `.github/workflows/schedule-nightly.yml:37` + `.github/workflows/test-matrix-presets/basic-test-nightly.json:17,20`
- `num_hidden_layers` override (compile-only, not download): `integrations/vllm_plugin/vllm_tt/vllm_utils.py:117-164`
- host-memory check (no Devstral threshold): `tests/integrations/vllm_plugin/generative/conftest.py:115-151`
