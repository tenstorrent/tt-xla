# Task: Implement DeepSeek V4 attention support in the tt-xla vLLM plugin

## Role and effort

You are implementing DeepSeek V4 (DSV4) attention on the Tenstorrent (`tt`) platform inside
the `tt-xla` vLLM plugin. This is a hard, cross-repo compiler + inference task. Work at maximum
rigor: read source before writing code, verify every assumption against the actual repos, and
prefer concrete diffs and runnable tests over prose. Do not pattern-match to how MLA "usually"
works — DSV4 has a sparse indexer branch, a sliding-window branch, and a two-branch softmax
merge that ordinary MLA does not.

## What we actually care about: the attention path

**The single most important outcome is a working attention implementation for DSV4.** Focus
your effort there. If you hit problems in parts of the model that are *not* attention — most
likely the MoE / FusedMoE path — you may ignore, stub, or work around them rather than fixing
them, as long as doing so still lets you exercise and validate the attention mechanism. A DSV4
model that can't run its MoE but *can* run and correctly compute its attention layers is a
success for this task; the reverse is not.

A practical consequence: you will likely find it easier to test attention in isolation than to
stand up the whole model. Write tests that target the DSV4 attention mechanisms directly —
construct the attention modules (and their KV caches, metadata, and custom ops) standalone, feed
them controlled inputs, and check outputs — rather than relying on a full end-to-end model
forward that drags in MoE, sampling, and everything else. Full-model runs are welcome where they
work, but they are not the bar; isolated attention correctness is.

## Repos in play

- `tenstorrent/tt-xla` — (the repo you are currently in) This is the PJRT/`torch_xla` frontend.
  The plugin lives in `integrations/vllm_plugin/vllm_tt/`. Custom ops that lower to StableHLO live in
  `python_package/tt_torch/custom_ops.py`.
- `tenstorrent/tt-mlir` — (present under third_party/) the MLIR compiler that lowers the StableHLO graph
  (with `tt.*` custom calls) through TTIR to the TTNN dialect. Custom-call conversion patterns are in
  `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`; runtime executors are under
  `runtime/lib/ttnn/operations/`.
- `tenstorrent/tt-metal` — (present under third_party/) the C++/kernel implementations of TTNN ops.
  The attention kernels you care about are under `ttnn/cpp/ttnn/operations/transformer/`.
- `vllm-project/vllm` — (https://github.com/vllm-project/vllm.git) upstream. The reference DSV4 model is
  `vllm/model_executor/models/deepseek_v4/` (attention, compressor, indexer, SWA cache) and the
  sparse/paged-MLA attention backends under `vllm/v1/attention/backends/mla/`.
- `vllm-project/tpu-inference` — (https://github.com/vllm-project/tpu-inference.git) the closest existing
  out-of-tree vLLM platform. Its `deepseek_v4` experimental path (`tpu_inference/.../experimental/deepseek_v4/`,
  `runner/kv_cache_manager.py`) is a useful reference for how another non-GPU backend handles
  the same model, though its overlay / `uint8`-reinterpret cache strategy is TPU-specific and
  probably not what you want on TT.

Clone whatever you need. Note that the vllm package should already be installed in the venv.

## Context: a design doc exists, but it is a lead, not a contract

A prior analysis produced a design document: DSV4_TT_Attention_Design.md. It traced all four repos and proposed an approach.
**Read it for orientation and to save yourself rediscovery time, but you are explicitly not required to follow it.**
It was written from source reading, not from an implementation attempt, so parts of it are unverified
assumptions. If you find a cleaner, more correct, or lower-risk approach at any point, take that
approach instead and document why you diverged. Treat the doc's claims as hypotheses to confirm
against the actual code, not as ground truth.

The doc's key claims you should independently verify before relying on them:

1. That `TTPlatform.get_attn_backend_cls` in `platform.py` hard-rejects `use_sparse=True` before
   reaching the MLA branch (a blocker for DSV4).
2. That the single shared `TTMetadata` fanned out via
   `dict.fromkeys(self._attention_layer_names, attn_metadata)` in `model_runner.py` cannot
   represent DSV4's per-branch (SWA vs compressed) page tables and sequence lengths.
3. That `initialize_kv_cache` raises `NotImplementedError` for more than one KV cache group —
   DSV4 needs multiple groups.
4. That the required TTNN kernels largely already exist: `paged_flash_multi_latent_attention_decode`
   accepting `attention_sink` + `sliding_window_size` (with `slidingWindowSize` currently
   hardcoded to `std::nullopt` in the tt-mlir runtime executor), and `sparse_sdpa` ("Sparse MLA
   prefill, DeepSeek DSA") taking a top-k `indices` tensor with `0xFFFFFFFF` sentinel masking.
   Confirm these signatures against `tt-metal` `main` before building on them, and confirm
   whether a *sparse decode* variant exists (the doc claims it does not).

The pinned vLLM version the plugin uses is confirmed to ship the `deepseek_v4` model module and
the `MLAAttentionSpec.compress_ratio` / `storage_block_size` fields, so you can rely on those
being present. If any of the *other* claims above turn out to be wrong at the pinned versions,
that changes the plan — surface it rather than coding around a false premise.

## How the plugin's OOT machinery works (verify, don't assume)

The existing MLA path is your template. Study it in
`vllm_tt/attention_impls/attention_mla.py`, `vllm_tt/__init__.py` (backend registration via
`register_backend`, layer substitution via `register_oot_layers`,
`MultiHeadLatentAttentionWrapper.register_oot`), and `vllm_tt/platform.py`. Study how
`torch.ops.tt.*` ops are defined in `python_package/tt_torch/custom_ops.py` — note the uniform
three-path pattern (an `xla` path that emits `stablehlo_custom_call` with frontend attributes, a
`cpu` eager reference path, and a `register_fake`). The `cpu` reference path is important: it's
what makes op-by-op correctness testing possible without hardware, and you should ship one for
every new op.

For the compiler side, each `tt.<name>` custom call needs a matching `OpConversionPattern` in
tt-mlir's `StableHLOToTTIRPatterns.cpp` (keyed on the call target name), TTIR/TTNN op definitions
in the respective `.td` files, and a runtime executor. Clone the existing
`tt.paged_flash_mla_decode` pattern + executor as a starting point for anything new.

## Scope discipline and first milestone

Do **not** try to land full DSV4 in one shot. The model has three attention layer types
(`compress_ratio <= 1` SWA-only, `== 4` CSA / lightning-indexer top-k, `== 128` C128A /
contiguous-prefix), plus the compressor, the indexer, multi-group KV caches, per-branch metadata,
and a two-branch online-softmax merge. Trying to do all of it before anything runs will produce
an untestable pile of code.

**The goal of this first pass is to get end-to-end tests of a single DeepSeek V4 Flash attention
layer working on the `tt` platform — and specifically, multiple such tests, each isolating a
different part of the attention mechanism.** Rather than one monolithic test, write a suite of
focused single-layer E2E tests so that when something breaks you know *which* attention component
broke. Concretely, aim for tests that separately target the distinct DSV4 attention behaviors,
for example:

- The SWA-only layer (`compress_ratio <= 1`): the natural first target and simplest slice. It
  needs the new backend selection (unblocking `use_sparse`), a sliding-window cache group,
  attention-sink handling, and the SWA attention call — but it avoids the compressor, the
  indexer top-k, and the two-branch merge, which are the highest-risk pieces. Get this one green
  first; it proves the plumbing.
- The sliding-window mechanism itself: verify that the window boundary is respected (a query
  attends only to the last `sliding_window` tokens), independent of the rest.
- The attention-sink path: verify sink logits fold into the softmax denominator correctly.
- The compressed / contiguous-prefix branch (C128A, `compress_ratio == 128`): the
  `(pos+1)//ratio` prefix attention, without the indexer top-k.
- The lightning-indexer top-k branch (CSA, `compress_ratio == 4`): the sparse index selection
  and sparse attention.
- The two-branch softmax merge: SWA output combined with the compressed-branch output. This is
  the highest-risk numeric piece — test it in isolation with a hand-built two-branch case before
  trusting it inside a full layer.

You do not have to implement all of these in this pass, and you certainly don't have to get all
of them green. But structure the work so each mechanism has its own test, and land them
incrementally: get the SWA-only slice fully working end-to-end first, then extend outward one
mechanism at a time, each with its own test. If you judge a different decomposition to be better,
choose it and justify the choice.

For each test: construct the relevant DSV4 attention module(s) (with KV cache, metadata, and
custom ops) at small shapes, run a forward pass (prefill and, where feasible, one decode step)
through the plugin, and assert numerical parity against a reference — either the upstream vLLM
CPU reference for that layer/mechanism, or the `cpu`-device reference path of your own custom
ops. Use the `num_hidden_layers` override in `TTConfig` and per-layer `compress_ratio` forcing to
keep any full-model runs tiny and fast to compile, and prefer isolated-module tests over
full-model runs wherever the isolated test is sufficient to validate the mechanism.

Everything beyond these single-layer attention tests — full multi-layer models, the MoE path,
fp8 latent cache format, cache overlay for memory savings, chunked prefill via `sparse_sdpa`'s
block-cyclic path — is explicitly out of scope for this milestone. Leave clear TODOs and a short
"what's next" note, but do not build it yet.

Along the way: ship the `cpu` reference and `register_fake` for every new custom op, keep
non-DSV4 models working (don't regress the existing MLA / dense paths — the multi-group and
metadata changes must be gated so single-group models are untouched), and respect the platform's
constraints (DYNAMO_TRACE_ONCE means static shapes — every variable-length thing, including any
top-k index tensor you introduce, must be padded to a compiled bucket).

## Working style

Verify against source at every step; when the design doc and the code disagree, the code wins.
Prefer running the CPU reference path over reasoning about numerics. When you hit a genuine fork
(e.g. kernel-side merge vs. StableHLO merge; SWA-only vs. a different first slice), state the
options and your choice with reasoning rather than silently picking one. If a required tt-mlir or
tt-metal change is on the critical path and you can't build those repos in this environment,
implement the tt-xla Python side fully, write the tt-mlir / tt-metal changes as concrete diffs,
and clearly mark what needs to be compiled and tested on hardware.

The definition of done for this task: **single DeepSeek V4 Flash attention layers run end-to-end
through the tt-xla vLLM plugin on the `tt` platform, validated by a suite of focused tests that
each check the numerical correctness of a distinct attention mechanism (sliding window, sink,
compressed prefix, indexer top-k, and the two-branch merge) against a reference — with the
SWA-only slice green first. A working attention path is the priority; failures confined to
non-attention parts of the model such as the MoE may be ignored or stubbed.**
