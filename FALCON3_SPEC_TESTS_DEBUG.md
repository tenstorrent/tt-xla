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

So the seed handling gap is **not** attributable to device sampling.

> **Superseded.** This section originally concluded that all four failures were "genuine
> forge-backend gaps" supporting Andrija's hypothesis. That is wrong. Two of the three parameters
> never reach forge at all — they are dropped by tt-media-server's own OpenAI shim. See
> [Root causes](#root-causes-verified-on-hardware) below, which was established by running the
> same three parameters against a stock `vllm serve` with no tt-inference-server in the picture.

Note the suite still mostly passes — stop sequences, `max_tokens`, penalties, and pinned
temperature/top_k/top_p determinism all work — so it is a specific set of parameters, not a
broadly non-conformant server.

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

### Why none of these tests catch the real bugs

Every one of them stops short of the layer where the defects live:

| test | what it actually drives | why it misses |
|:--|:--|:--|
| `test_seed_correctness.py` | `Sampler.random_sample()` on CPU with hand-built `q_samples` | validates the math; never touches the engine, so broken wiring upstream is invisible |
| `test_sampling_params.py::test_seed` | offline `LLM`, device sampling | `xfail(strict=True)` — the failure is expected and green, and nobody re-checks it under the workaround it recommends |
| `test_cpu_sampling.py` | offline `LLM`, host sampling | contains **no seed test at all** — the workaround was never verified |
| `test_logprobs_correctness.py` | `Sampler().gather_logprobs()` directly | all 14 call sites pass `num_logprobs >= 1`; `logprobs=0` appears nowhere in the tree |
| `n` | — | no coverage anywhere |

The common thread: the suite unit-tests the sampler and offline-tests the engine, but nothing
exercises the **OpenAI request -> `SamplingParams` translation**, which is where all of these live.

## Root causes (verified on hardware)

Established by driving `n` / `logprobs` / `seed` against a stock
`python -m vllm.entrypoints.openai.api_server` with opt-125m — no tt-inference-server, no Falcon3.
Results:

```
n=2 / n=3                        HTTP 200, choices=2 / 3      <- works
logprobs=True                    HTTP 500  IndexError         <- tt-xla bug
logprobs=True, top_logprobs=3    HTTP 200, logprobs present   <- works
seed=42 x2                       identical=False              <- tt-xla bug
seed=0 / seed=7, x16 concurrent  8/16 unique each             <- seed not honored
```

This splits the original four failures across **two different repos**.

### tt-media-server, not forge

The Falcon3 server under test was `uvicorn main:app` — tt-media-server's hand-rolled OpenAI shim,
not vLLM's `api_server`. Three of the conformance failures are caused there and never reach the
plugin:

| failure | cause |
|:--|:--|
| `n` ignored | `open_ai_api/chat.py:143` — `choices` is a hardcoded 1-element list. `n` *is* forwarded to `SamplingParams`; the extra outputs are discarded when the response dict is built. |
| `logprobs` absent | `domain/chat_completion_request.py` — the pydantic model has **no `logprobs` field**, so `"logprobs": true` is dropped before vLLM sees it. `CompletionRequest` has it; only the chat model is missing it. |
| `seed` ignored | `utils/sampling_params_builder.py:48` — **`seed = None` is hardcoded**, deliberately, citing #4539 and a ~5x slowdown on the seeded path. |

The hardcoded `seed = None` fully explains why run 2's `CPU_SAMPLING=true` changed nothing: the
seed was discarded host-side before sampling location could matter.

### tt-xla, underneath

1. **`logprobs: true` without `top_logprobs` -> HTTP 500.** It maps to `SamplingParams(logprobs=0)`.
   `metadata.py:220` computes `needs_logprobs = max_num_logprobs > 0 if max_num_logprobs else False`,
   so a `0` takes the `else False` branch, no logprobs tensors are produced, and vLLM's
   `_create_chat_logprobs` then dies on `top_logprobs[i]` with `IndexError`. `logprobs=0` is a valid
   request meaning "the sampled token's logprob, no alternatives".
2. **logprobs + `cpu_sampling` kills EngineCore.** `sampler.py:253`
   `logprobs.gather(-1, token_ids)` receives a host `LongTensor` for `selected_token_ids` and raises
   `Input tensor is not an XLA tensor`. Every subsequent request on that server 500s.
3. **`seed` is not honored, and `cpu_sampling: True` does not fix it.** Reproduced over HTTP
   (16/16 unique at seed=0) *and* via the in-process `LLM` API (8/8 unique in a single batch,
   seed=42 non-reproducible across calls).

Point 3 retires the "works in-process, broken over HTTP" hypothesis for seed: it is broken on both
paths, so #4539's stated kernel-level cause cannot be the whole story and **the workaround
documented on `test_seed` is simply wrong** — not stale, not HTTP-specific.

## The tt-xla repro test

`tests/integrations/vllm_plugin/sampling/test_openai_sampling_params.py` — self-contained, spawns
its own `api_server`, no tt-inference-server checkout needed.

```bash
cd /localdev/kmabee/tt-xla
source venv/activate
pytest tests/integrations/vllm_plugin/sampling/test_openai_sampling_params.py -v
```

Measured: **3 passed, 5 xfailed, 12m13s**, most of it the two server startups (device-sampling and
`cpu_sampling` fixtures). All xfails are `strict=True`, so a fix turns them into red XPASS rather
than silently passing.

| test | result |
|:--|:--|
| `test_chat_n[2]`, `test_chat_n[3]` | PASS — regression guard, `n` works here |
| `test_chat_logprobs_with_top_logprobs` | PASS — the contrast case |
| `test_chat_logprobs_without_top_logprobs` | XFAIL — the `logprobs=0` 500 |
| `test_chat_logprobs_cpu_sampling` | XFAIL — EngineCore death |
| `test_chat_seed_reproducibility` | XFAIL |
| `test_chat_seed_reproducibility_cpu_sampling` | XFAIL — the workaround |
| `test_chat_seed_under_concurrency` | XFAIL — 16 concurrent, seed=0 |

Fixture gotcha worth knowing before copying `test_responses_api.py`'s config: the plugin asserts
`max_num_batched_tokens >= max_model_len * max_num_seqs`, so raising `--max-num-seqs` for the
concurrency test forces `--max-num-batched-tokens` up too (128 -> 4096 at `max_model_len=128`).
Leaving it at 128 fails at engine init, not at request time.

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

Against **tt-media-server** (three separate few-line fixes, all independent of Falcon3 and forge):

- `n` — build `choices` from all outputs instead of a hardcoded 1-element list (`open_ai_api/chat.py:143`).
- `logprobs` — add the field to `ChatCompletionRequest` and forward it; it is already plumbed for `CompletionRequest`.
- `seed` — the hardcoded `seed = None` needs revisiting once tt-xla honors seeds; until then it should
  at least be documented in the API surface rather than silently ignored.

Against **tt-xla**:

- File `logprobs=0` -> 500 (`metadata.py:220`). Cheapest real fix of the set.
- File logprobs + `cpu_sampling` -> EngineCore death (`sampler.py:253`).
- Correct the `test_seed` xfail reason: the `cpu_sampling` workaround it recommends does not work,
  verified both in-process and over HTTP. Check whether #4539 is still open and whether its scope
  covers the non-kernel path.

Process / open questions:

- Decide whether the Falcon3 `llm.json` / `server_tests_config.json` additions should be upstreamed
  as-is (they will make Falcon3 CI red until the tt-media-server side is fixed).
- Cross-check Andrija's Mistral CI failure — if it also ran through tt-media-server, its `n` /
  `logprobs` / `seed` failures have the same three non-forge causes and "forge is non-conformant"
  is the wrong conclusion for them. Check the 401 pattern too.
- Decide whether `test_openai_sampling_params.py` belongs in CI at 12 min for one device. Currently
  marked `nightly`.
- Re-run `test_seed_reproducibility` several times to characterise its flake rate; a flaky test in
  a release-gating suite is its own problem regardless of the underlying seed bug.
- Confirm `MAX_MODEL_LENGTH=4096` reproduces the same pass/fail split.
