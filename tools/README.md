# Benchmark residency probes (dev-only)

Reproduction tooling for the imagegen/video-gen benchmark warm-metric defect.
**Nothing here is imported by production code** — the probes subclass the shipped
pipelines and never modify them.

| file | purpose |
|---|---|
| `residency_probe.py` | the three signals: `[DRAM]` device residency, `[CMPL]` XLA compile counters, `[TIER]` tt-metal program-build tier read from the Inspector log |
| `generic_call_probe.py` | model-agnostic: runs `generate()` N times and reports per-call `uncached`/`ctime`/`progs`. Answers "is call 2 warm?" for any pipeline exposing `setup()`/`generate()` |
| `zimage_tier_probe.py` | Z-Image, per-component cold/warm inside one residency |
| `sdxl_tier_probe.py` | SDXL-Lightning, same, for the `.to("cpu")` eviction shape |
| `zimage_tier_report.py` | turns a Z-Image probe run into the evidence tables |

Also required for the `[DRAM]` signal: the `PJRT_Device_MemoryStats` implementation
in `pjrt_implementation/` on this branch (issue #470). Without it DRAM reads `n/a`.

## Two traps this tooling exists to avoid

1. `xm.get_memory_info()` with **no argument** resolves to the virtual SPMD device
   and *throws* inside `PjRtComputationClient::GetMemoryInfo`. A bare `except` makes
   that indistinguishable from "not implemented" — which is why every earlier log
   read `dram=n/a`. Pass an explicit addressable device.
2. `torch_xla/_dynamo/dynamo_bridge.py:650` calls `metrics.clear_counters()` and
   restores only `UncachedCompile` and `DynamoExtractCompiledGraph`. Deltas built on
   `CachedCompile`/`MarkStep`/`CompileTime` can go **negative**. Only `uncached` and
   `dynamo` are load-bearing; the rest are printed with a trailing `~`.

## Run

    export HF_HOME=<hf cache> TT_METAL_CACHE=<cache> PYTHONPATH=.:tests:tests/benchmark
    export TT_METAL_LOGS_PATH=<run dir>          # per-run Inspector dir
    python tools/generic_call_probe.py \
        --pipeline third_party.tt_forge_models.z_image.pytorch.pipeline \
        --cls ZImageTTPipeline --cfg ZImageConfig \
        --steps 2 --calls 3 --run-dir <run dir> --tag zimage

`--calls 3` is deliberate: two calls cannot classify every model. HunyuanVideo-1.5
rebuilds on call 2 and is warm on call 3, so a warmup+steady scheme mislabels it.
