# Falcon3-7B-Instruct: spec_tests / VLLMParamConformanceTest on forge (Aug 2026)

Debug notes for the first `spec_tests` run of a forge-backend LLM. Context: Andrija Cicovic
reported that `VLLMParamConformanceTest` fails for Mistral on forge and that no forge model had a
successful run in recent CI history — either the model doesn't declare the test, or it was already
red for another reason. This documents adding it for Falcon3-7B-Instruct and what actually fails.

SDPA numerics work is in `FALCON3_SDPA_NUMERICS_DEBUG.md`. Raw logs live in
`falcon3_sdpa_ccfg_2026-08-02/`, untracked.

**Related commits.** tt-inference-server (`kmabee/falcon3_hang_accuracy_debug`, local only, not
pushed): `b61aaf583` enables the test for Falcon3, `158d997cc` adds the MATH_APPROX_MODE
passthrough. Both were committed with `--no-verify` -- that repo's `pytest` pre-commit hook fails
with 8 collection errors on a clean tree, unrelated to these changes.

---

## What these tests are

`spec_tests` is a workflow type alongside `benchmarks` / `evals`, part of the release workflow.
A model only gets one if it declares a test in `test_module/test_suites/llm.json`.

`VLLMParamConformanceTest` checks the server honours its documented OpenAI-compatible request
parameters — nothing about model quality or speed. It's the only release stage that can catch a
server silently misbehaving while still emitting plausible numbers: evals and benchmarks are
purely quantitative, so a corrupted prompt or ignored output cap surfaces as a slightly low score
rather than a defect. (That is how the Mistral-Small chat-template bug was caught: evals returned
32.5 vs 45.96 published, which read as ordinary bring-up drift.)

Implementation: `tt-inference-server-v2/test_module/llm_tests/vllm_param_conformance_test.py`
wraps a child pytest over `tt-inference-server-v2/llm_module/test_vllm_chat_completions.py`
(9 test functions, 22 parametrized assertions), endpoint `/v1/chat/completions`.

## Wiring Falcon3 in — two additions, not one

1. **`tt-inference-server-v2/test_module/server_tests_config.json`** — Falcon3 had **no
   `model_configs` entry at all** (27 models, zero falcon). Added `falcon3_7b` with
   `id_name: falcon3-7b`, `weights: ["Falcon3-7B-Instruct"]`, `category: LLM`, and the same
   `compatible_devices` list as `llama_3_1_8b`.
2. **`tt-inference-server-v2/test_module/test_suites/llm.json`** — added `falcon3_7b` to the
   `VLLMParamConformanceTest` matrix (previously `qwen3_32b`, `llama_3_1_8b`,
   `llama_70b_family`, `gpt_oss_20b`).

Order matters: adding only (2) raises
`ValueError: Model 'falcon3_7b' referenced in test_matrix but not found in model_configs`
(`test_categorization_system/suite_loader.py:78`). So for any model that has never had spec tests,
"not declared in llm.json" understates the work.

Verify expansion without touching hardware:

```bash
cd /localdev/kmabee/tt-inference-server/tt-inference-server-v2
env -u PYTHONPATH /localdev/kmabee/tt-xla/venv/bin/python -c "
import json,sys; sys.path.insert(0,'.')
from test_module.test_categorization_system.suite_loader import expand_test_matrices, load_server_tests_config
cfg=load_server_tests_config()
mats=json.load(open('test_module/test_suites/llm.json'))['test_matrices']
for s in expand_test_matrices(mats, cfg['model_configs']):
    if 'falcon' in s['id']: print(s['id'], s['weights'], [t['template'] for t in s['test_cases']])
"
```

Expect `falcon3-7b-p150 ['Falcon3-7B-Instruct'] ['VLLMParamConformanceTest']`.

**Do not** round-trip these JSON files through `json.dump` — it reformats ~900 lines. Edit as text.

## How to run

### 1. Start the server

```bash
cd /localdev/kmabee/tt-xla            # MUST cd here first -- venv/activate is $(pwd)-relative
source venv/activate
TT_INFERENCE_SERVER_ROOT=/localdev/kmabee/tt-inference-server \
TT_METAL_HOME=/localdev/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal \
  /localdev/kmabee/tt-inference-server/tt-media-server/launch_falcon3_bf16.sh |& tee server.log
```

Compile takes ~17-20 min at 32K ctx. Ready when the log shows `VLLM model load` and
`curl localhost:8019/health` returns 200. Watch for it with:

```bash
until grep -q "VLLM model load" server.log; do sleep 15; done; echo READY
```

Optional env knobs (none are required for conformance testing — they affect numerics, not the
API surface):

| var | effect |
|:--|:--|
| `CPU_SAMPLING=true` | host-side sampling instead of on-device (see results below) |
| `MATH_FIDELITY=hifi4` / `FP32_DEST_ACC_EN=true` / `MATH_APPROX_MODE=false` | compute-kernel-config overrides |
| `PORT=8019` | server port (default 8019) |

`launch_falcon3_baseline.sh` is the bfp8-weights variant; `launch_falcon3_bf16.sh` is bf16.

The launch script sets `API_KEY=your-secret-key` by default — you need the same value in the
client environment below, or every test 401s.

### 2. Run the conformance suite

```bash
cd /localdev/kmabee/tt-inference-server
env -u PYTHONPATH API_KEY=your-secret-key /localdev/kmabee/tt-xla/venv/bin/python run.py \
  --model Falcon3-7B-Instruct --tt-device p150 --engine forge \
  --impl forge-vllm-plugin --workflow spec_tests --service-port 8019 \
  --dev-mode --skip-system-sw-validation
```

Takes ~10-17 min (see wall-clock note below). Results are in the `VLLMParamConformanceTest -> fail (data={...})` log line:
`parameter_conformance_summary` for per-case pass/fail, `detailed_test_results` for messages. It is
one Python dict literal on a single line, so parse it rather than grepping —
`~/scripts/parse_spec_tests.sh LOG` prints the scoreboard, the failing assertions, and the suite
wall-clock.

### 3. Running only selected tests (skip run.py entirely)

The workflow shells out to a child pytest, so for iteration you can drive that pytest directly
against an already-running server — no workflow scaffolding, no model_config validation, no
18-minute full suite. Set up once per shell:

```bash
cd /localdev/kmabee/tt-inference-server/tt-inference-server-v2
export API_KEY=your-secret-key
PY=/localdev/kmabee/tt-inference-server/.workflow_venvs/.venv_v2_run_script/bin/python
PP=/localdev/kmabee/tt-inference-server:/localdev/kmabee/tt-inference-server/tt-inference-server-v2
```

Then the four failing tests only:

```bash
env -u PYTHONPATH PYTHONPATH=$PP $PY -m pytest \
  llm_module/test_vllm_chat_completions.py \
  -k "test_logprobs or test_n or test_non_uniform_seeding or test_seed_reproducibility" \
  --output-path /tmp/spec_sel --task-name sel \
  --endpoint-url http://127.0.0.1:8019/v1/chat/completions \
  --model-name tiiuae/Falcon3-7B-Instruct -v
```

Collects 5/22, 17 deselected (`test_n[2]`, `test_n[3]`, `test_seed_reproducibility`,
`test_non_uniform_seeding`, `test_logprobs`). Swap the `-k` expression for a single test to
isolate one. Add `-x` to stop at first failure, `-s` to see the raw responses the tests print.

Four gotchas:

- **`-k` is a substring match**: `-k test_n` also selects `test_non_uniform_seeding` (3 tests, not
  2). To isolate, use node IDs instead — `"llm_module/test_vllm_chat_completions.py::test_n"`, or
  `"...::test_n[3]"` for a single parametrization. Quote them; brackets are glob chars.
- **`--output-path` is required.** The `output_path` fixture does
  `Path(request.config.getoption("--output-path"))` with no default, so omitting it dies on
  `Path(None)`. Any writable dir works.
- **`PYTHONPATH` must carry both roots.** `llm_module/conftest.py` imports from `server_tests`
  (repo root) and `report_module` (v2 root), and the child interpreter does not inherit `run.py`'s
  in-process `sys.path` additions (see the comment at `vllm_param_conformance_test.py:118-124`).
  Run from the v2 root as cwd, matching what the workflow does.
- **`API_KEY` still applies** — without it every selected test 401s, same as the full suite.

Output is ordinary pytest, so assertion diffs land in the terminal and `parse_spec_tests.sh` is not
useful on it. Structured results still land in `/tmp/spec_sel/parameter_report_sel.json`.

### Gotcha: missing API_KEY looks like a total backend failure

Without `API_KEY` (or `OPENAI_API_KEY`) in the environment, **all 22 assertions fail** with
`401 Client Error: Unauthorized ... {'detail': 'Not authenticated'}`, and the summary rows just
read `0/N passed`. You have to open `detailed_test_results` to see every message is a 401.

Auth resolves at `server_tests/conftest.py:115` (`OPENAI_API_KEY` or `API_KEY`, else `JWT_SECRET`).
The child pytest inherits the parent env (`env = dict(os.environ)` in
`vllm_param_conformance_test.py`), so setting it on the `run.py` invocation is enough.

Note the eval workflow does *not* need this — lm-eval hits `/v1/completions`. Only the
conformance suite hits `/v1/chat/completions`, which requires auth.

**If Andrija's Mistral CI failure shows a uniform 401 pattern, part of it may be environmental
rather than backend conformance.** Worth checking before attributing it to forge.

---

## Results: Falcon3-7B on forge, P150

Server config: bf16 weights + KV, `math_fidelity=hifi4`, `fp32_dest_acc_en=true`,
`math_approx_mode=false`, opt 1, trace on. Log: `spec_tests_falcon3_devicesampling.log`.

### Run 1 — device sampling (`cpu_sampling=False`): 5 pass / 4 fail

| test | result | failure |
|:--|:--|:--|
| test_coherence_verbatim_echo | PASS 1/1 | |
| test_determinism_parameters | PASS 3/3 | |
| test_max_tokens | PASS 2/2 | |
| test_penalties | PASS 9/9 | |
| test_stop | PASS 2/2 | |
| test_n | **FAIL 0/2** | `n=2` returns 1 choice — `assert 1 == 2` |
| test_logprobs | **FAIL 0/1** | `'logprobs'` absent from the response choice |
| test_seed_reproducibility | **FAIL 0/1** | same seed -> `'The capital of France is Paris.'` vs `'Paris.'` |
| test_non_uniform_seeding | **FAIL 0/1** | seed=0, expected 1 unique output, got **15** |

Most of the suite passes — stop sequences, max_tokens, penalties, and pinned
temperature/top_k/top_p determinism all work.

**Two are clear unimplemented-parameter gaps:**
- `n` — the server ignores it and always returns a single choice.
- `logprobs` — the field is never present in the response, even when requested.

**Two seed failures — initially suspected to be config, but they are not.**
`tt-media-server/tt_model_runners/vllm_runner.py:53-54` documents that device sampling
(`cpu_sampling=False`) "can't honor `SamplingParams(seed=...)`", so the obvious hypothesis was that
`test_seed_reproducibility` / `test_non_uniform_seeding` fail only because of that config. **Run 2
disproves it** — see below.

### Run 2 — host sampling (`CPU_SAMPLING=true`): identical, 5 pass / 4 fail

Log: `spec_tests_falcon3_cpusampling.log`. Confirmed `cpu_sampling: True` in `additional_config`,
and zero `Sampling on xla:0` lines in the server log (device sampling genuinely off).

The result is the **same set of four failures**, with the same messages:

- `test_seed_reproducibility` — same seed still gives `'The capital of France is Paris.'` vs `'Paris.'`
- `test_non_uniform_seeding` — seed=0 gives **16** unique outputs (15 under device sampling)

So the seed handling gap is **not** attributable to device sampling. All four failures look like
genuine forge-backend gaps:

| gap | evidence |
|:--|:--|
| `n` ignored | always returns 1 choice regardless of `n` |
| `logprobs` never returned | field absent from the response even when requested |
| `seed` not honored | non-reproducible under both device and host sampling |

This supports Andrija's hypothesis fairly directly: the forge backend does not implement these
OpenAI-defined parameters, independent of sampling location. Note the suite still mostly passes —
stop sequences, `max_tokens`, penalties, and pinned temperature/top_k/top_p determinism all work —
so it is a specific set of unimplemented parameters, not a broadly non-conformant server.

### Run 3 — host sampling again, same server instance: 6 pass / 3 fail (FLAKY)

Log: `spec_tests_falcon3_cpusampling_run3.log`. Run against the **same server process** as run 2,
same `cpu_sampling: True` config, ~1h20m later.

| test | run 2 (14:22) | run 3 (15:47) |
|:--|:--|:--|
| `test_seed_reproducibility` | FAIL | **PASS** |
| all others | identical | identical |

**`test_seed_reproducibility` is flaky.** Same server, same config, opposite result. It issues two
sequential same-seed requests, so it can pass by chance when nothing else is in flight.
`test_non_uniform_seeding` — 32 *concurrent* requests via `asyncio.gather` — has failed in every
run so far.

That pattern fits tt-xla #4539's description exactly: a single shared seed across all 32 cores
that ignores per-row `q_samples` will break per-row determinism under concurrency while
low-concurrency sequential requests can still line up. It also means the seed conclusion needs
care:

- **`test_non_uniform_seeding` (concurrent): consistently broken** — the reliable signal.
- **`test_seed_reproducibility` (sequential): flaky** — do not treat a single pass or fail as
  evidence either way.

So the earlier claim in run 2 that "the `cpu_sampling` workaround does not hold" is too strong on
the basis of one run. What holds is the *concurrent* case failing regardless of sampling mode.
The sequential case needs repeated runs before anything is concluded from it.

### Wall-clock note

Measured from `Running vLLM parameter suite` to the `VLLMParamConformanceTest ->` verdict:

| run | duration |
|:--|--:|
| device sampling (`cpu_sampling=False`) | 10m30s |
| host sampling (`CPU_SAMPLING=true`) | 17m22s |

Host sampling is ~65% slower here. Note this is a *generation-heavy* suite --
`test_non_uniform_seeding` alone fires 15+ full generations -- so it is not representative of
steady-state decode throughput, where host sampling has measured *faster* on this model.

(An earlier draft of this file claimed ~20s for the device-sampling run. That was the aborted
401 run, where every request was rejected before any generation happened.)

---

## Existing tt-xla coverage for the same parameters

`tt-xla/tests/integrations/vllm_plugin/sampling/` already covers most of this ground, which makes
the spec-test failures easier to interpret:

| spec failure | tt-xla coverage | status |
|:--|:--|:--|
| `seed` | `test_sampling_params.py::test_seed` | **`@pytest.mark.xfail(strict=True)`**, cites tt-xla **#4539** |
| `logprobs` | `test_logprobs_correctness.py` (8 tests), `test_trace_logprobs.py` | not xfailed |
| penalties | `test_penalties_correctness.py` | not xfailed |
| **`n`** | **none** | no coverage anywhere in the tree |

The existing xfail reason on `test_seed`:

> Device sampler does not honor per-row seeds: the `tt::sampling` kernel uses a single shared seed
> across all 32 cores and ignores per-row `q_samples`, so seeded sampling is no longer
> deterministic. Tracked in tt-xla #4539. **Workaround: set
> `additional_config={'cpu_sampling': True}`.** Remove this xfail once the kernel grows per-row
> seed support.

### The workaround did not hold over HTTP

Run 2 used `cpu_sampling: True` (verified in `additional_config`, zero `Sampling on xla:0` lines)
and both seed tests failed -- but run 3 on the same server passed `test_seed_reproducibility`, so
only the *concurrent* test (`test_non_uniform_seeding`) is a reliable failure. For that one, two
readings:

1. the documented workaround is stale, or
2. the seed is dropped somewhere in the **HTTP -> SamplingParams** path, which would be a
   different bug from #4539's kernel-level one.

(2) is more likely: tt-xla's `test_seed` drives the offline `LLM` API in-process, while the spec
test goes over `/v1/chat/completions`. Note also that `test_cpu_sampling.py` contains **no seed
test at all**, so the workaround has never actually been verified in tt-xla's own suite.

The same shape holds for `logprobs`: it passes tt-xla's own correctness tests but is absent from
the HTTP response. So across the reliable failures the pattern is **"works in-process, broken over HTTP"** — pointing at the serving/entrypoint layer rather than the sampler or the kernels.
That is a sharper hypothesis than "forge doesn't implement the OpenAI API" and should be the first
thing the next session tests.

**Check first:** whether #4539 is still open and whether its scope already covers the HTTP path.
If it does, this is evidence the stated workaround is wrong; if not, this is a separate
serving-layer issue.

## A pure tt-xla repro is feasible (no tt-inference-server dependency)

`tt-xla/tests/integrations/vllm_plugin/generative/test_responses_api.py` already does exactly what
is needed: it spawns `python -m vllm.entrypoints.openai.api_server` as a subprocess with the tt
plugin, waits for health, and exercises HTTP endpoints. Its fixture uses a deliberately tiny
config for fast startup:

```
--max-model-len 128 --max-num-batched-tokens 128 --max-num-seqs 1 --gpu-memory-utilization 0.001
```

`falcon3_serve_no_fix.log` also confirms stock `vllm serve` works standalone on tt-xla.

So a tt-xla-only ticket can carry a self-contained test that starts its own server and asserts
`n` / `logprobs` / `seed` over HTTP, runnable in tt-xla CI with no tt-inference-server checkout.
Adjustments needed vs the `test_responses_api.py` fixture:

- `--max-num-seqs 32` (the seeding test fires 32 concurrent requests), not 1
- a small model rather than Falcon3-7B, for speed

This is probably the better bug report: it isolates the serving layer, runs in tt-xla CI, and
sidesteps whether the failure is Falcon3-specific.

## Faster server for iterating

Nothing in the suite needs long context — largest generation is 1024 tokens (`test_stop`), all
prompts are one-liners, and no test references the `max_context` fixture. So
`MAX_MODEL_LENGTH=4096` should give identical results with a much shorter compile. **Not yet
verified** — worth one run to confirm before trusting it.

Keep `MAX_NUM_SEQS=32`: `test_non_uniform_seeding` fires 32 concurrent requests via
`asyncio.gather`, and a smaller batch would serialize them.

Speculative extras, unverified: `MIN_NUM_SEQS=32` (the default of 1 compiles a second b1-prefill
graph) and `ENABLE_TRACE=false`. Change one variable at a time if the result matters.

## Open items

- File `n`, `logprobs` and `seed` against the forge vLLM plugin — all three are independent of
  Falcon3 and will hit every forge model that declares spec tests.
- Decide whether the Falcon3 `llm.json` / `server_tests_config.json` additions should be upstreamed
  as-is (they will make Falcon3 CI red until `n` / `logprobs` / `seed` are fixed or xfailed).
- Cross-check Andrija's Mistral CI failure for the 401 pattern before attributing it to forge.
- Test the "works in-process, broken over HTTP" hypothesis — the single highest-value next step.
- Re-run `test_seed_reproducibility` several times to characterise its flake rate; a flaky test in
  a release-gating suite is its own problem regardless of the underlying seed bug.
- Confirm `MAX_MODEL_LENGTH=4096` reproduces 5 pass / 4 fail.
- Add `n` coverage to tt-xla's vllm_plugin tests; it has none today.
