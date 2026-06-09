### Testing notes for chunked prefill (#4986)

Some testing tips from debugging this — sharing in case they're useful, not meant to be prescriptive.

**Test the {chunking on/off} × {batch 1 / batch>1} matrix.** Correctness bugs tend to hide specifically in the *chunking × batch>1* corner — each axis can pass alone while the combination fails. Concretely, what we observed (Llama-3.2-3B, greedy):
- batch 1, multi-chunk → token-identical to single-shot ✅ (chunk attention logic is correct for one user)
- batch>1, single-shot (no chunking) → all users identical ✅
- batch>1, multi-chunk → users ≥1 diverge ❌ (still open)

So passing batch=1 and/or single-shot batch>1 is not sufficient; the multi-chunk batch>1 case needs its own coverage.

**Use identical prompts in a batch as a cheap oracle.** Greedy (`temperature=0`) decoding of N *identical* prompts must produce N *identical* outputs — each row is independent. Per-user divergence is an unambiguous correctness signal. Note the perf benchmarks only PCC-check User 0, so they pass while Users 1..N are wrong; inspect *all* users' outputs, not just the pass/fail.

**Single-chunk vs multi-chunk equivalence.** Same long prompt generated greedily with a large budget (no chunking) vs a small `max_num_batched_tokens` (forces multi-chunk) must give identical tokens. The batch=1 single-chunk run is a good ground-truth reference to compare every other config against.

**`fp32_dest_acc_en` is a useful precision-vs-logic discriminator.** If a divergence disappears with `fp32_dest_acc_en=True`, it's a precision / compute-kernel-config issue (likely tt-mlir, e.g. the gather-index class — see below). If it persists under fp32, it's a logic bug. This cleanly separates pre-existing infra precision issues from feature-logic issues.

**Prerequisite — the gather-index fix.** Without it, batch>1 prefill diverges for a *pre-existing* reason unrelated to chunking: tt-mlir lowers the per-user last-token gather to a float index matmul with bf16 dest accumulation, and flat indices ≥512 round to the wrong row (tt-xla #5116 / tt-mlir #8666 family). Test with that fix applied so you're exercising the feature, not the infra bug. (We confirmed single-shot batch>1 is correct once it's in.)

**Per-op activation diff is the fastest localizer.** Instrumenting the tt-mlir runtime op loop to compare each op's per-user output (for identical prompts) pinpoints exactly which op first introduces divergence — far quicker than bisecting from logits. That's how the gather-index issue above was found.

**Also worth covering:** chunk-boundary lengths (just above/below `k * chunk_budget`, and the 1-token-remainder case), and `VLLM_XLA_CHECK_RECOMPILATION=1` to catch shape-driven recompiles across chunk steps.

---

### Known batch>1 fixes / follow-up (from debugging)

Two batch>1 correctness bugs were found and root-caused in the chunked path:

1. **Non-block-aligned chunk** — `_block_aligned_chunk` could schedule a chunk
   that wasn't block-aligned when the per-step budget remainder was `< block_size`,
   leaving `num_computed_tokens` mid-block and corrupting the block-granular
   `fill_page_table` write. **Fixed** (defer the request instead).

2. **Mixed-stage prefill packing** — the scheduler packs the budget greedily, so a
   *fresh* request (`num_computed=0`) can share a step with a *continuation*
   (`num_computed>0`); the fresh user is then forced into the all-or-nothing
   `prefix_chunk_mode` and is corrupted. Confirmed by serializing prefill (no
   mixing) → output becomes correct.
   - **Planned fix (Option 1, near-term):** don't pack prefill requests at
     different `num_computed` stages into one step (all-fresh / all-same-stage
     still batch; staggered prefills serialize).
   - **⚠️ Perf follow-up (Option 2):** if serializing staggered prefills hurts
     throughput, make `prefix_chunk_mode` correct for a `computed=0` user mixed
     with `computed>0` users so mixed-stage steps stay batched. Revisit once
     chunked-prefill perf numbers exist. (Mechanism-pinning tooling: env-gated
     per-op row-diff in `runtime/lib/ttnn/program_executor.cpp`.)

Also note: batch>1 prefill requires the tt-mlir gather-index fp32 fix
(tt-xla #5116) to be present, or it diverges for a pre-existing reason
independent of chunking.
