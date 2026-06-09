# [vLLM] Batched prefill: identical prompts produce divergent greedy output (bf16 destination accumulation)

## Summary

Greedy (`temperature=0`) decoding of **N identical prompts in one batch** does not produce identical outputs across all rows. Each row in a batch is an independent computation, so argmax decoding must be identical for every row — divergence is a correctness bug.

The divergence has a clean, structural signature: the batch splits into **contiguous groups of rows** that each compute a different result (e.g. rows `0..k-1` emit one token, rows `k..N-1` emit another). It is **deterministic**, and it is **fully eliminated by enabling fp32 destination accumulation** (`fp32_dest_acc_en=True`). It is therefore a **bf16 destination-accumulation** problem in the batched prefill graph (tt-mlir compute-kernel-config family).

## Reproduction

Llama-3.2-3B-Instruct, 32 identical prompts, greedy, generating a single token (pure prefill):

```python
import vllm

PROMPT = "Here is an exhaustive list of the best practices for writing clean code:"
N = 32

llm = vllm.LLM(
    model="meta-llama/Llama-3.2-3B-Instruct",
    max_model_len=128,
    max_num_seqs=N,
    max_num_batched_tokens=N * 128,        # prefill all N prompts together
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    additional_config={"fp32_dest_acc_en": False},
)
sp = vllm.SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
outs = llm.generate([PROMPT] * N, sp)

ids = [o.outputs[0].token_ids[0] for o in outs]
for i, t in enumerate(ids):
    print(f"row {i:2d}: token {t}")
print("unique tokens:", len(set(ids)))   # bug: 2 (should be 1)
```

Observed (deterministic): rows 0–3 emit token `720`, rows 4–31 emit token `320`; `unique tokens = 2`. With `fp32_dest_acc_en=True` all 32 rows emit `720` (`unique tokens = 1`). Full captured output: https://gist.github.com/kmabeeTT/dd7524de473cd8158040fc860886cafd

Reproduced at tt-xla `8a54156b5` (`main`) with stock tt-mlir `c5f398432` (the version `main` pins), single device.

## Observations (to guide debugging)

1. **Contiguous-group split, boundary tracks the prefill shape.** The batch splits into a contiguous prefix `0..k-1` vs the rest `k..N-1`. Where the boundary `k` falls depends on the prefill shape (the program-config / core-grid the batched ops are given), **not** on a fixed user count:

   | additional_config | first divergent row |
   |:--|:--|
   | `{fp32_dest_acc_en: False}` | **4** (group 0–3 vs 4–31) |
   | `{fp32_dest_acc_en: False, min_context_len: 32}` | **16** (group 0–15 vs 16–31) |

   Forcing a 32-token minimum prefill context moves the boundary from 4 to 16. This is consistent with the cause being the **batch grouping of the runtime-selected program-config**, not a magic number. (Batches at or below the boundary are clean: with the boundary at 16, batch 4/8/16 all agree and batch 24 splits 16/8.)

2. **Deterministic.** Re-running an identical config gives byte-identical per-row output every time (same split, same token ids). Not a race.

3. **Layer-count independent.** Same behavior on a single-decoder-layer truncation (`additional_config={"num_hidden_layers": 1}`) and on the full 28-layer model; only the specific token ids differ.

4. **Not the sampler.** Forcing host-side sampling (`additional_config={"cpu_sampling": True}`) reproduces the split, so the wrong values are already in the **logits coming off the device**, not in on-device sampling/argmax.

5. **`fp32_dest_acc_en=True` eliminates it** — all rows agree. So the divergence is entirely a low-precision (bf16) destination-accumulation effect. (Note: enabling it on 8B-class models has been associated with a separate DRAM-OOM regression, so it is not a free global fix.)

6. **No single op reproduces it in isolation — even config-matched.** A standalone `ttnn` program running each production prefill op (the QKV / O / gate-up / down matmuls, dense causal SDPA, rms_norm) at the exact batched shapes with **N identical rows** produces **bit-identical** outputs across all rows — including when passed the *exact* compute config the lowered graph specifies (`math_fidelity=hifi4, fp32_dest_acc_en=False`). A row-local op (matmul / elementwise / norm) cannot, by construction, turn identical input rows into a grouped split. So the split is **introduced by graph composition**, not by any single kernel.

7. **Layouts are not the asymmetry.** All intermediate tensors in the lowered prefill graph are **interleaved DRAM** (no sharded layouts), and the matmul ops carry only `compute_config = hifi4` with **no explicit program_config** — the core grid is selected from the activation shape at runtime.

## Likely root cause

The combination of (5)+(6)+(7) points at the **in-graph program-config / core-grid that tt-metal selects for the batched prefill ops under the compiler-emitted `hifi4, fp32_dest_acc_en=False` config**: it partitions the batch into contiguous groups whose bf16 destination-accumulation reductions round differently, so one group's logits diverge enough to flip the greedy token. The boundary moving with the prefill shape (4 vs 16 above) is direct evidence of this grouping. fp32 destination accumulation has enough headroom that the groups agree.

This is the same family as the known compute-kernel-config issues:
- [tt-mlir #8666](https://github.com/tenstorrent/tt-mlir/issues/8666) — `fp32_dest_acc_en` not applied properly for `ttnn.matmul`.
- [tt-mlir #8657](https://github.com/tenstorrent/tt-mlir/issues/8657) — SDPA has no compute-kernel-config / fp32-accumulation path (`ttnn.scaled_dot_product_attention*` don't implement `TTNNComputeKernelConfigOpInterface`).

## Suggested next step

Localizing the exact op needs an **in-graph per-op activation dump** (tt-mlir runtime op callback / `dumpTensor`): run the batched prefill, dump every op's output, and find the first op whose output for a row past the boundary diverges from row 0 for identical inputs. That op's emitted program-config / core-grid is the one introducing the grouping asymmetry.

## Also visible in the existing benchmark

The same divergence surfaces in the existing `tests/benchmark/test_vllm_benchmarks.py -k llama-3.2-3b` run (batch 32, `fp32_dest_acc_en=False`), which prints multiple distinct completions for 32 identical prompts but still passes. That path generates 128 tokens, so it shows the divergence in a noisier form (decode mixed in); use the single-token snippet above to see the clean structural signature.

It is already live in nightly CI — e.g. the `perf vllm_llama_3_2_3b (n150-perf)` job at https://github.com/tenstorrent/tt-xla/actions/runs/27064401476/job/79883258025 (2026-06-06) printed 3 distinct completions for the 32 identical prompts and was still marked **success**.

## Why it isn't caught today

Benchmarks assert token counts and PCC of **User 0** only, so per-user divergence (Users 1..N) is invisible — a batch of 32 identical prompts can print 2 distinct completions and still pass.

## Cross-reference comment for tt-mlir #8666 / PR #8708 (draft — not yet posted)

Both [#8666](https://github.com/tenstorrent/tt-mlir/issues/8666) and its fix PR [#8708](https://github.com/tenstorrent/tt-mlir/pull/8708) are open as of writing; PR #8708 gates the `fp32_dest_acc_en` matmul fix on `inner dim > 50000` + bf16 output (`kLargeInnerDimThreshold = 50000`, aimed at training-backward / vocab-size matmuls). Suggested comment:

**Heads-up from the inference/serving side (tt-xla vLLM).** We see a batched-prefill correctness bug that looks like the same `fp32_dest_acc_en` accumulation family: greedy decoding of N *identical* prompts in one batch diverges — the batch splits into deterministic contiguous groups that emit different tokens — and **`fp32_dest_acc_en=True` fully eliminates it** for prefill. Standalone repro + analysis: https://github.com/tenstorrent/tt-xla/issues/5116.

Relevant to this PR: the fix is gated on `inner dim > 50000` + bf16 output, but Llama inference matmul inner dims are <= ~8192, so it won't fire for inference. Consider gating by dtype/output rather than a fixed inner-dim cutoff so inference matmuls are covered.

Also worth tracking: a matmul-only fix won't cover the fused SDPA op, which has no fp32-accumulation path at all ([#8657](https://github.com/tenstorrent/tt-mlir/issues/8657)).
