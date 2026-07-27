# Handoff: Falcon3-7B-Instruct hang hunt (tt-xla standalone repro)

**Status as of 2026-07-27.** Written to move this investigation to another
machine/session. Everything below is verified against a live run on the
original box unless explicitly marked as unverified.

---

## 1. What we are chasing

Falcon3-7B-Instruct, forge LLM on a single P150, hangs during eval runs: the
engine stops producing tokens, in-flight requests never complete, and the
server has to be killed. The goal on the new machine is to **reproduce it
reliably and, ideally, reproduce it without tt-inference-server in the loop**,
so it can be filed against tt-xla with a standalone reproducer.

### This is NOT the already-fixed #4521 hang

A hang with the *same outward fingerprint* was chased in July and fixed. Do not
re-derive it:

- **tt-inference-server#4521** — "Forge single-chip LLMs intermittently hang
  under concurrent load"
- **tt-xla#5664** — the real root cause
- **tt-xla#5672** — the fix (merged 2026-07-17, `9d838be6d`)
- Prior handoff doc:
  `HANDOFF_conc32_device_read_stall.md` on tt-inference-server branch
  `kmabee/issue_4496_forge_llm_production_settings.testing`

**Root cause of #4521:** `AscendScheduler.schedule()` gates its decode path on
`len(self.scheduled_req_ids) == 0`. A request given a *partial* prefill chunk
had its id added to `scheduled_req_ids` but was deliberately kept out of
`self.running` until it finished. `update_from_output()` cleared ids only by
iterating `self.running`, so the partial's id could never be cleared — decode
was blocked for *every* running request, permanently, with no error. Fix:

```python
# integrations/vllm_plugin/vllm_tt/scheduler/ascend_scheduler.py
for req_id in num_scheduled_tokens:
    self.scheduled_req_ids.discard(req_id)
```

**The fix is present in the branch this doc ships with** — see
`ascend_scheduler.py` around line 807 (`(tt-xla #5664)` comment). So a new hang
here is *something else*.

### The trap: the gdb backtrace lies

At hang time the only interesting thread is blocked in:

```
from_device -> Tensor::cpu -> enqueue_read_tensor -> enqueue_read_shards_nolock
 -> FDMeshCommandQueue::finish_nolock -> read_completion_queue
  -> copy_completion_queue_data_into_user_space   <-- looks like the smoking gun
```

For #4521 this was a **red herring** — it is the normal, healthy *idle* state
once the scheduler stops feeding work. Weeks went into tt-metal fast-dispatch
before that was understood. Expect to see this exact backtrace again; it does
not by itself mean the device is at fault.

### The two moves that actually cracked it — do these first

1. **Remove any client-side timeout** and watch the server directly. This
   proved the wedge never self-resolves, killing the "device race that
   eventually clears" theory.
2. **`py-spy dump` on the EngineCore process.** The main thread was spinning in
   vLLM core's *idle-loop* path, which only fires when the scheduler returns a
   **completely empty schedule**. That pointed at `AscendScheduler`, not the
   device. gdb's C++ backtrace could not show this; py-spy's Python view could.

`py-spy` is **not installed** on the original box — install it first:
`pip install py-spy` (needs `ptrace` perms; run as the same user or with sudo).

---

## 2. Environment to recreate

| Component | Value |
|---|---|
| Device | single P150 (Blackhole), chip 0 |
| tt-xla branch | `kmabee/falcon3_7b_hang_debug` (pushed to origin) |
| tt-xla HEAD | this commit; base = upstream `7fe709421` |
| tt-xla local-only commit | `0f3cde69b` "Expose math_fidelity knob in vLLM plugin TTConfig" |
| tt-xla install | **editable local build**, `pjrt-plugin-tt 0.1.260723+dev.0f3cde69b` from `<tt-xla>/python_package` |
| vllm | `0.22.1` |
| vllm_tt | `0.1` (in-tree, `integrations/vllm_plugin`) |
| torch-xla | `2.9.0+git56f8e61` |
| tt-lang | `1.1.5.dev20260704+light` |
| tt-inference-server branch | `kmabee/falcon3_hang_accuracy_debug` @ `3e7af8ec8` (pushed) |
| helper scripts repo | `git@github.com:kmabeeTT/scripts.git` @ `feea715` |

The tt-xla plugin is a **local editable build**, so the new machine must build
tt-xla at this branch — a CI wheel will not carry `0f3cde69b`.

`TT_METAL_HOME` in this setup is **only a writable JIT/kernel cache root**
(it contains just `built/`), not a real tt-metal source tree. The real tt-metal
lives under `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal`.

---

## 3. The model config under test

Forge P150 spec, from `workflows/model_specs/dev/cnn.yaml` in tt-inference-server.
Verified byte-exact against a live run (`Device 0: additional_config=` log line):

```json
{"enable_const_eval": true, "min_context_len": 128,
 "experimental_weight_dtype": "bfp_bf8", "experimental_kv_cache_dtype": "bfp_bf8",
 "cpu_sampling": false, "optimization_level": 1, "enable_trace": true,
 "prefill_chunk_size": 1024, "min_num_seqs": 1, "prefill_batch_threshold": 16,
 "num_hidden_layers": 1}
```

Engine args: `max_model_len=32768`, `max_num_seqs=32`,
`max_num_batched_tokens=32768` (`max(batch*chunk, max_model_len)`),
`gpu_memory_utilization=0.35`, `enable_chunked_prefill=False` as passed (the
plugin platform flips it back to `True` internally).

`num_hidden_layers: 1` is the **single-layer debug hack** — compiles in ~2 min
instead of ~20 and isolates serving/concurrency behavior from model depth. Note
from the #4521 work: single-layer ran 200 clean conc=64 waves, i.e. that stall
was **model-depth sensitive**. If the current hang also refuses to reproduce at
one layer, that is a data point, not a dead end — retry at full depth
(`NUM_HIDDEN_LAYERS=""`).

**Accuracy at one layer is meaningless** (observed: ifeval 8.09%, gpqa 2.00%).
Single-layer is a hang instrument only.

---

## 4. Scripts shipped with this branch

Both live at the tt-xla repo root and are self-contained (no tt-inference-server
code executes).

### `serve_falcon3_7b_forge.sh` — stock `vllm serve`, no tt-media-server

```bash
cd <tt-xla> && source venv/activate && ./serve_falcon3_7b_forge.sh
```

Reproduces the config above exactly, including key order in
`additional_config` (vLLM hashes that dict), the env the runner sets
(`TT_VISIBLE_DEVICES`, `TT_METAL_CACHE=$TT_METAL_HOME/built/<dev>`,
`OMP/MKL/TORCH` thread caps), and a **background warmup POST** mirroring
`VLLMForgeRunner.warmup()` — `vllm serve` has no warmup of its own, so without
it the first eval request eats the one-time compile and looks like a hang.
Wait for `[warmup] WARMUP COMPLETE` before driving evals.

Knobs: `PORT DEVICE_IDS NUM_HIDDEN_LAYERS MAX_MODEL_LENGTH MAX_NUM_SEQS
GPU_MEMORY_UTILIZATION PREFILL_CHUNK_SIZE MIN_NUM_SEQS PREFILL_BATCH_THRESHOLD
OPTIMIZATION_LEVEL CPU_SAMPLING ENABLE_TRACE KV_CACHE_DTYPE WEIGHT_DTYPE
MATH_FIDELITY FP32_DEST_ACC_EN API_KEY WARMUP TT_METAL_HOME`.

### `run_falcon3_7b_evals.sh` — the two evals, no run.py

```bash
./run_falcon3_7b_evals.sh                       # both tasks, 0.75 each
./run_falcon3_7b_evals.sh --limit 0.05          # both, 5%
./run_falcon3_7b_evals.sh --limit 20            # both, 20 docs
./run_falcon3_7b_evals.sh --tasks ifeval --limit 1.0
./run_falcon3_7b_evals.sh --concurrent 8        # narrow the admission burst
```

Inlines the two `lm_eval` commands that
`tt-inference-server-v2/llm_module/eval_command.py:build_eval_command()` emits
for this model. **Verified argv-identical against a live `run.py` eval process**
(`ps` on the running invocation), including `max_retries=1` from the P150
spec's `eval_max_retries`, `num_concurrent=32` clamped to `max_concurrency`,
`--seed 42` *plus* the injected `seed=42` inside `--gen_kwargs`,
`--apply_chat_template`, and the `EVALS_COMMON` `--trust_remote_code
--confirm_run_unsafe_code`.

Ends with a summary scoring the same keys against the same references
(`prompt_level_strict_acc` vs 72.64, `exact_match,flexible-extract` vs 43.43).

---

## 5. lm_eval on a fresh machine

`uv` handles this; no tt-inference-server checkout needed. ~2 min, ~1.5 GB:

```bash
uv venv --python 3.10 <tt-xla>/.venv_lm_eval
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu \
uv pip install --python <tt-xla>/.venv_lm_eval \
  "lm-eval[api,ifeval,math,sentencepiece,r1_evals,ruler,longbench,hf] @ git+https://github.com/tstescoTT/lm-evaluation-harness.git@f3d35ab3c8d74b90548b3198b2337c32431a8e06" \
  datasets==3.1.0
```

`run_falcon3_7b_evals.sh` auto-detects `$ROOT/.venv_lm_eval`, else
tt-inference-server's `.workflow_venvs/.venv_evals_common`; `LM_EVAL_BIN`
overrides.

**Do not substitute PyPI `lm-eval`.** Both tasks exist upstream, so it looks
safe, but the TT fork (96 ahead / **108 behind** upstream) patches
`lm_eval/tasks/ifeval/instructions.py`, `lm_eval/tasks/ifeval/utils.py`, and
`lm_eval/filters/extraction.py` — the last being the `flexible-extract` filter
gpqa is scored on. Upstream silently scores **both** of these tasks
differently. Pin the sha; `@evals-common` is a moving branch.

`UV_EXTRA_INDEX_URL` is load-bearing — without the CPU torch index you pull
multi-GB CUDA wheels for a client that never touches a GPU.

gpqa needs HF access (`Idavidrein/gpqa` is gated) unless the dataset is cached;
set `HF_TOKEN`. ifeval is open.

---

## 6. The tt-inference-server path (still the best repro today)

The hang has so far only been *observed* through tt-inference-server. Loop mode
was added to the launcher for exactly this hunt:

```bash
~/scripts/model_servers/run_evals_forge.sh --model Falcon3-7B-Instruct --port 8019 \
  --loops 20 --dir falcon3_7b_instruct_evals_loop |& tee falcon3_loop_master.log
```

Master log carries one `START` and one result line per iteration (timestamp,
rc, elapsed); each iteration's full output goes to `<dir>/iter_NN.log`.
A hang shows up as a bare `START` with no matching result line.

`--timeout SECS` exists but **think before using it**: it SIGTERMs `run.py`,
which disconnects the eval client, and per tt-xla#5664 a client disconnect
aborting in-flight requests is exactly what force-clears the wedged scheduler
state. A timeout can destroy the evidence. Run without it to catch a hang in
progress.

Server launcher (tt-media-server path, single layer):
`~/scripts/model_servers/launch_falcon3_7b_instruct_uvicorn.sh`, run from a
tt-xla venv. Set `TT_INFERENCE_SERVER_ROOT` to select the checkout.

---

## 7. When it hangs — diagnostic runbook

```bash
# 0. FIRST, rule out the boring causes
grep -iE "TT_FATAL|Out of Memory" <serve log>       # OOM
#    server warm? a cold server's first-request compile is not a hang

# 1. py-spy FIRST — this is the check that distinguished cause from symptom
pip install py-spy
py-spy dump --pid <VLLM::EngineCore pid>
#    main thread in vLLM core's idle loop  -> scheduler returned an EMPTY
#                                             schedule -> scheduler-side bug
#    main thread genuinely blocked in a device call -> device-side

# 2. gdb second, for the native picture
gdb -p <VLLM::EngineCore pid> -batch -ex "thread apply all bt"
#    compile threadpools idle (tf::Executor::_wait_for_task, libTTMLIRCompiler)
#    confirms it is NOT a compile
#    read_completion_queue on its own means little — see section 1

# 3. chip alive?
tt-smi -s      # ARCCLK ticking, DDR_STATUS 0x55555555, heartbeat advancing

# 4. recover
pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "uvicorn main:app"
fuser -k 8019/tcp; tt-smi -r
```

Pick the **non-zombie** `VLLM::EngineCore` pid — defunct ones linger.

---

## 8. Known non-parity: standalone vs tt-media-server

This matters for interpreting a *clean* standalone run.

| | tt-media-server | `serve_falcon3_7b_forge.sh` |
|---|---|---|
| Admission driver | `device_worker_dynamic_batch`: `get_many(32)` → burst of `generate()` → keeps pulling while those run = continuous overlapping bursts | vLLM-native, smooth |
| Per-request seed | force-dropped (#4338 / tt-xla#4539) | honored by vLLM |
| HTTP layer | tt-media-server FastAPI | vLLM OpenAI server |
| Scheduler | `AscendScheduler` | **same** — `platform.py` forces `scheduler_cls="vllm_tt.scheduler.AscendScheduler"` for TT regardless of entry point |

In the #4521 hunt this difference was **decisive**: tt-media-server hung on the
first conc=64 batch in 2/2 runs, while stock `vllm serve` ran 60 clean conc=64
rounds and in-process `AsyncLLMEngine` ran 20,000 requests clean. So:

> **A clean standalone run is evidence, not proof.** If the hang reproduces
> only through tt-media-server, that is itself the finding — it points at the
> driver/admission pattern, and reducing *that* to a minimal harness becomes
> the next step.

Both paths still run greedy (lm-eval sends `temperature=0`), so the seed
difference is a speed/code-path difference, not an expected accuracy
difference. `EVAL_GEN_SEED=""` drops the seed if you want the closer match.

---

## 9. Eval limits — a live gotcha

`limit_samples_map[CI_NIGHTLY]` for both tasks is **0.75** on the
tt-inference-server debug branch (commit `3e2c2b4df` "Downsample to 0.75",
2026-07-23), *not* the 0.25 that appears in older notes. Confirmed by `ps` on a
live `run.py` eval, which passes `--limit 0.75`.
`run_falcon3_7b_evals.sh` defaults to 0.75 to match. Re-check this value on the
branch you land on — it has moved more than once during this debug work.

Higher limit = more docs = more concurrent load = presumably better odds of
tripping the hang. Worth keeping at 0.75+ while hunting.

---

## 10. Open questions for the new machine

1. **Does it reproduce at all under stock `vllm serve`?** Nothing has been run
   through `serve_falcon3_7b_forge.sh` yet — it is written and config-verified,
   but **never executed against hardware**. That is step one.
2. **Single layer vs full depth.** #4521 was depth-sensitive. If one layer
   stays clean, go to `NUM_HIDDEN_LAYERS=""`.
3. **Concurrency scaling.** #4521's probability scaled hard with concurrency
   (~1/20 at conc=32, near-deterministic at conc=64). `--concurrent` on the
   eval script and `MAX_NUM_SEQS` on the serve script are the knobs; consider
   pushing past 32.
4. **Is it even the same class of bug?** py-spy first. If the scheduler is
   returning empty schedules again, look for another `scheduled_req_ids`-style
   leak; if not, the device-side theory gets its first real evidence.

---

## 11. Related, separate thread: accuracy

There is a parallel accuracy investigation on the same model — P150 scores
below the L4 reference on both tasks, with the odd finding that **bf16
weights+KV scores ~5 points *lower* on ifeval than bfp8** despite being higher
precision. Write-up (uncommitted on the original box):
`falcon3_7b_accuracy_debug_issue.md`, with decode IR gists linked inside.
Not required for the hang hunt, but the same scripts serve both.

---

## 12. Quick start on the new machine

```bash
# 1. build tt-xla at this branch (editable), then:
cd <tt-xla> && source venv/activate

# 2. lm_eval (section 5)
uv venv --python 3.10 .venv_lm_eval && UV_EXTRA_INDEX_URL=... uv pip install ...

# 3. py-spy, for when it wedges
pip install py-spy

# 4. serve (wait for "[warmup] WARMUP COMPLETE")
./serve_falcon3_7b_forge.sh

# 5. drive evals in another shell, in a loop, NO client timeout
for i in $(seq 1 20); do
  echo "=== iter $i $(date) ==="
  ./run_falcon3_7b_evals.sh --output eval_results_iter_$i || break
done

# 6. when an iteration stops making progress -> section 7 runbook, py-spy FIRST
```
