# Falcon3-7B-Instruct: SDPA numerics investigation (Aug 2026)

Debug notes for [tt-inference-server#4752](https://github.com/tenstorrent/tt-inference-server/issues/4752)
(Falcon3-7B eval accuracy) and [tt-metal#51927](https://github.com/tenstorrent/tt-metal/issues/51927)
(SDPA-decode accuracy). Written as a handoff — a fresh session should be able to pick up from here.

spec_tests / VLLMParamConformanceTest findings are in `FALCON3_SPEC_TESTS_DEBUG.md`.

Raw artifacts (logs, eval reports, IR, scripts) live in `falcon3_sdpa_ccfg_2026-08-02/`, untracked.

---

## TL;DR

- Darko's chisel work identified SDPA-decode as the dominant Relative-L2 contributor in Falcon3.
- We reduced SDPA-decode rel_l2 **5.1x** (0.024995 -> 0.004872), verified on P150.
- **Eval scores did not improve.** All four builds sit within ~1 stderr of each other.
- ~96% of generated outputs *changed*, and ~18% of prompts flipped correct/incorrect — but the
  flips cancel. SDPA numerical error and eval accuracy are decoupled for this model.
- Two real bugs were found and fixed along the way (one tt-xla, one tt-metal).

---

## Bug 1 (tt-xla): fidelity knobs silently discarded at optimization_level >= 1

`TTNNPipelines.h` `resolveOptimizationLevelOptions()`:

```cpp
if (computeCfgMathFidelity.getNumOccurrences() == 0 && optimizationLevel > 0) {
    computeCfgMathFidelity = OptionalMathFidelity::Undefined;
}
if (computeCfgFp32DestAccEn.getNumOccurrences() == 0 && optimizationLevel > 0) {
    computeCfgFp32DestAccEn = std::optional<bool>(std::nullopt);
}
```

tt-xla set these by direct assignment (`module_builder.cc`). Assignment does **not** bump
`getNumOccurrences()` — only `cl` parsing does. We run at opt level 1, so both overrides were
reset before `TTNNSetComputeKernelConfig` ran.

Consequence: the `hifi2 + fp32` and `hifi4 + fp32` rows in issue #4752's table never applied those
knobs. They were effectively plain runs, which is why they matched the no-override rows.

**Fix** (commit `f93f094c4`): route them through `PassPipelineOptions::parseFromString` so the
occurrence counter increments.

Verified in the emitted TTNN IR: before, **zero** `ttnn.matmul` and **zero** SDPA ops carried
`compute_config`; after, all 4864 do.

`math_approx_mode` was additionally plumbed end-to-end (it had no path at all). It needs **four**
layers, all of which must be present or it fails:

```
launch script env
  -> vllm_runner.py additional_config
  -> TTConfig (integrations/vllm_plugin/vllm_tt/platform.py)   <-- easy to miss, raises TypeError
  -> PJRT compile_options (compile_options.h / .cc)
  -> module_builder.cc parseFromString
  -> tt-mlir TTNNSetComputeKernelConfig
```

## Bug 2 (tt-metal): SDPA always uses the approximate exp

`ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp`,
`sub_exp_block_bcast_cols_inplace` — shared by SDPA **prefill and decode**:

```cpp
exp_tile_init<true /* approx */, scale_fp32, InputClamping::None>();
exp_tile<true /* approx */, false /* scale_en */, InputClamping::None, iterations>(j, vector_mode_exp);
```

`sdpa_decode_program_factory.cpp:767` defines `EXP_APPROX_MODE`, but the compute kernel never reads
it — so the ttnn `exp_approx_mode` knob is inert (toggling it gives bit-identical output).

It cannot simply be forwarded: the accurate bf16 path folds the scale into `TTI_SFPMULI`, which
needs a compile-time immediate, so a runtime scale fails to link with
`error: impossible constraint in 'asm'`. The `is_fp32_dest_acc_en` branch uses ordinary SFPI math
and does accept a runtime scale.

**Patch** — take the accurate path only when `fp32_dest_acc_en` is set:

```cpp
constexpr bool exp_approx = !DST_ACCUM_MODE;
exp_tile_init<exp_approx, scale_fp32, InputClamping::None>();
...
constexpr uint16_t scale_bf16_v = static_cast<uint16_t>(scale_fp32 >> 16);
exp_tile<exp_approx, !exp_approx /* scale_en */, InputClamping::None, iterations>(
    j, vector_mode_exp, scale_bf16_v);
```

Branch: **`kmabee/sdpa_accurate_exp`** (`b6371bb6488`, on `f1f4ff75579`).
Patch file: `falcon3_sdpa_ccfg_2026-08-02/ttmetal_sdpa_accurate_exp.patch`.

**Important limitation:** because it is gated on `fp32_dest_acc_en`, the patch is **inert for any
caller using ttnn defaults** — including Darko's repro script as written, which passes no
`compute_kernel_config`. The gating is probably not the right long-term shape.

## The 0.9702 constant is NOT bug 2

The constant scale factor Darko reported in the single-unmasked-token case is `math_approx_mode`
(ttnn defaults it `true`, selecting `_reciprocal_compat_<APPROXIMATION_MODE ? 2 : 3>` — one Newton
iteration instead of two). **No code change needed**; it is reachable via `compute_kernel_config`.

With `approx=True`, HiFi2 and HiFi4 are bit-identical at 0.970164 — the approximation saturates
everything else, which is why fidelity looked irrelevant.

---

## Kernel results (P150, grid 11x10, Darko's repro shapes)

Single unmasked token (`cur_pos=0`; output must be a bit-exact copy of `V[0]`):

| fidelity | approx | fp32acc | patch | scale | rel_l2 |
|:--|:--|:--|:--|--:|--:|
| HiFi2 (default) | True | False | no | 0.970164 | 0.029920 |
| HiFi4 | False | False | no | 0.999906 | 0.002642 |
| HiFi4 | False | True | **yes** | **1.000000** | **0.000000** |

Realistic decode positions, rel_l2 vs fp32 reference from the same bf16 values:

| config | rel_l2 |
|:--|--:|
| default | 0.024995 |
| HiFi4/approx=False/fp32=True, no patch | 0.020278 |
| HiFi4/approx=False/fp32=True, **patch** | **0.004872** |

bf16 error budget (staged torch references rounding where the kernel rounds): scores->bf16
0.002404, +probs->bf16 0.002648, +bf16 recip 0.003512. So the kernel was ~7x its own bf16 budget
before, ~1.4x after.

## Eval results — the negative result

Config fixed: bf16 weights + bf16 KV, `math_fidelity=hifi4`, `fp32_dest_acc_en=true`, opt 1,
trace on, device sampling, greedy. Full sets (541 ifeval / 198 gpqa).

| build | ifeval | gpqa |
|:--|--:|--:|
| 1 baseline | 66.91 | 38.89 |
| 2 + tt-xla knob plumbing + tt-mlir `61010e770` (SDPA ccfg) | 67.28 | 39.90 |
| 3 + tt-metal accurate exp | 68.02 | 38.89 |
| 4 + `math_approx_mode=false` | 65.62 | 39.90 |
| target (0.95 x L4) | 69.0 | 41.3 |

lm-eval stderr is 2.01-2.04 at n=541; spread is 2.40, mean 66.96 vs baseline 66.91. **No build is
distinguishable from any other.**

### Outputs changed a lot; scores did not

Per-prompt diff of generated text vs baseline (full ifeval, 541 docs):

| build | text differs | flips | gained | lost | net |
|:--|--:|--:|--:|--:|--:|
| +ccfg | 519 (95.9%) | 96 | 49 | 47 | +2 |
| +accurate exp | 521 (96.3%) | 94 | 50 | 44 | +6 |
| +math_approx | 518 (95.7%) | 97 | 45 | 52 | -7 |

~96% of generations differ and ~18% of prompts flip pass/fail, but flips land in both directions
in near-equal numbers. The change is real and large; it is simply uncorrelated with correctness.

### Do not trust the downsampled evals

CI-nightly ifeval swings 2.6 points across builds whose full-set values move by ~1. On build 3 it
read **61.99** while the full set read **68.02** — opposite directions. Use full runs only.

---

## How to run things

### Serve Falcon3-7B on P150

```bash
cd /localdev/kmabee/tt-xla            # MUST cd here first, see gotcha below
source venv/activate
TT_INFERENCE_SERVER_ROOT=/localdev/kmabee/tt-inference-server \
TT_METAL_HOME=/localdev/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal \
MATH_FIDELITY=hifi4 FP32_DEST_ACC_EN=true MATH_APPROX_MODE=false \
  /localdev/kmabee/tt-inference-server/tt-media-server/launch_falcon3_bf16.sh |& tee server.log
```

Compile takes ~17-20 min at 32K ctx. Ready when the log shows `VLLM model load` / `/health` 200.
Add `CPU_SAMPLING=true` for host-side sampling. `launch_falcon3_baseline.sh` is the bfp8 variant.

### Run evals

```bash
cd /localdev/kmabee/tt-inference-server
env -u PYTHONPATH /localdev/kmabee/tt-xla/venv/bin/python run.py \
  --model Falcon3-7B-Instruct --tt-device p150 --engine forge \
  --impl forge-vllm-plugin --workflow evals --service-port 8019 \
  --dev-mode --skip-system-sw-validation --limit-samples-mode ci-nightly
```

For **full** sets, delete the two `EvalLimitMode.CI_NIGHTLY: 0.5,` lines from the Falcon3 block in
`evals/eval_config.py` (the `f90bba971` treatment) — `limit_samples_map.get()` returns `None` and
no `--limit` reaches lm-eval. `git checkout` the file afterwards.

Reports land in `workflow_logs/reports_output/evals/`.

### Run the SDPA kernel bench directly (no server)

```bash
cd .../third_party/tt-metal/src/tt-metal
TT_METAL_HOME=$PWD PYTHONPATH=$PWD/ttnn:$PWD /localdev/kmabee/tt-xla/venv/bin/python \
  /localdev/kmabee/tt-xla/falcon3_sdpa_ccfg_2026-08-02/sdpa_knob_sweep.py
```

`PYTHONPATH` must include `$PWD/ttnn`, not just `$PWD` — otherwise `import ttnn` picks up a
namespace package and `ttnn.open_device` is missing.

Kernels are **JIT-compiled** from `TT_METAL_HOME`, so tt-metal kernel edits need **no rebuild** —
just restart the server. The cache is content-hashed and invalidates itself.

### Rebuild tt-xla (only for C++/plugin changes)

```bash
cd /localdev/kmabee/tt-xla && source venv/activate && cmake --build build |& tee build.log
```

tt-mlir source patches survive this: `git checkout ${TT_MLIR_VERSION}` is a no-op when HEAD already
equals the pin, so `git cherry-pick -n` changes are preserved.

---

## Gotchas that cost time

- **`venv/activate` is `$(pwd)`-relative.** Sourcing it from anywhere other than
  `/localdev/kmabee/tt-xla` builds garbage `PYTHONPATH` entries **and creates a stray `venv/`
  directory in the current folder**. Bit us twice (once polluting the tt-metal tree).
- **`venv/activate` puts `tt-xla/tests` on `PYTHONPATH`**, whose `utils.py` shadows
  tt-inference-server's `utils` package — the v2 runner then dies on `import jax.lax`. Run all
  tt-inference-server commands with `env -u PYTHONPATH`.
- **Backgrounded commands reset cwd.** Multi-step `mkdir && cp` chains silently landed archives in
  `/localdev/kmabee/tt-xla/results/` instead of the artifact dir. Use absolute paths.
- **eval `rc=1` is not a failure** — it is the acceptance gate reporting a score below target.
  Read the report markdown, not the exit code.
- **The eval output dir accumulates** across runs; select the results/samples file by doc count
  (541 / 198), not by "latest".

---

## Open items

- Throughput cost of the tt-metal patch is **unmeasured**. `exp` is SDPA's inner loop and the
  header is shared with prefill. This decides whether the `fp32_dest_acc_en` gating is acceptable
  or it needs to be an explicit opt-in.
- Confirm Darko's repro is byte-identical pre/post patch at ttnn defaults (inferred from the
  gating, not yet measured).
- Per-category ifeval comparison against the L4 reference. #4733 has the headline numbers
  (10 runs, ifeval mean 72.79, gpqa 43.43 every run) but no per-sample output; settling it needs
  one GPU run with `--log_samples`. Both reference runs were full-set — inferable because every
  ifeval value is an exact k/541 and 43.43 = 86/198.
