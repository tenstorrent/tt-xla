# Handoff: b1-prefill DP device mismatch (tt-xla)

Branch: `ssalice/devstral-qwen-5893` (see git log for latest)
Full detail: `CHUNKED_SDPA_4K_RESULTS.md` in the same branch.

## The bug in one paragraph

Under the b1 serial-prefill cap, prefill and decode select **different row-count
buckets**. DP device assignment is derived from the row index *within the
bucket*, so a request can write its KV on one device during prefill and have
decode look for it on another. Symptom: the first generated token is correct
(the prefill step produced it), then the row degenerates into token soup.

## Evidence (measured, not inferred)

`TTXLA_SLOT_TRACE=1` dumps each request's row -> DP-device assignment per pass.
Qwen3-0.6B, 2 layers, mesh (4,8) dp=4, batch 8:

| request | prefill dev (bucket) | decode dev (bucket) | agree |
|---|---|---|---|
| 0 | 0 (4) | 0 (8) | OK |
| 1 | **1** (4) | **0** (8) | **MISMATCH** |
| 2 | **2** (4) | **1** (8) | **MISMATCH** |
| 3 | **3** (4) | **1** (8) | **MISMATCH** |
| 4-7 | 2,2,3,3 (8) | 2,2,3,3 (8) | OK |

- `prefill_batch_threshold=16` -> **3 of 8** requests change device.
- `prefill_batch_threshold=0`  -> **0 of 8**.

The mismatched set `{1,2,3}` is **exactly** the set of corrupted rows observed
in two independent full-depth Devstral-2-123B runs at 4096 context. Row 0
survives because it is the only index where `r == r//2`.

## Mechanism

1. `ascend_scheduler.py:130` — when `num_pending <= prefill_batch_threshold`,
   `fresh_prefill_cap = min_num_seqs` (=1): one fresh prefill admitted per step.
2. `input_batch` therefore grows 1,2,3,... and early prefill passes pick the
   **small** bucket (`_select_target_num_reqs`, `model_runner.py:5214`):
   `target=4`, `rows_per_dev=1` -> row `r` lands on **dev r**.
3. Once `actual_num_reqs > small`, prefill flips to the **big** bucket:
   `target=8`, `rows_per_dev=2` -> **dev r//2**.
4. Decode *always* uses the decode bucket (also 8 here) -> **dev r//2**.
5. Requests prefilled inside the small-bucket window are stranded.

Buckets are `(small, big, decode) = (min_num_reqs, max_prefill_num_reqs,
num_reqs_max_model_len)` — `_bucket_num_reqs`, `model_runner.py:5181`.
Note `min_num_seqs` is rounded **up to a multiple of dp_size**
(`model_runner.py:500`), so `min_num_seqs=1` becomes 4 on dp=4, 8 on dp=8.
Inputs are genuinely sharded (1 row/device), **not** replicated.

## Causal confirmation (not just correlation)

Qwen3-0.6B at **full depth** (28 layers -> coherent output, unlike a truncated
big model), mesh (4,8) dp=4, batch 8, greedy, 64 tokens. Slot trace and text
captured in the *same* run, so the relocation set and the corruption set are
measured together rather than compared across configs:

| arm | requests that relocate | rows failing `assert_output_coherent` |
|---|---|---|
| `pbt=0`  | **none** | **none** |
| `pbt=16` | **{1,2,3}** | **{1,2,3}** |

Identical sets, same run. Example rows:

```
row 1  (RELOCATED)
  pbt=0 : ' very cold, and the temperature is 20 degrees. The temperature is...'
  pbt=16: " very''\n\n''\n\n''\n\n''\n\n''..."      <- correct first token, then collapse

row 5  (not relocated)
  pbt=0 : ' a type of food that I have never had before. I have never had...'
  pbt=16: ' a type of food that I have never had before. I have always been...'
```

This closes the loop the earlier full-depth Devstral runs left open: there the
corrupted set `{1,2,3}` matched a relocation set measured in a *separate*
2-layer run. Here both come from one run on a model whose baseline is coherent.

**Caveat on diffing arms byte-for-byte:** rows 5,6,7 also differ between arms
while staying perfectly coherent. Different bucket usage changes padding and
collective reduction order, so benign numeric divergence is expected. Use
coherence, not equality, to decide whether a row is corrupted.

## The invariant is already known to the codebase

`model_runner.py:1203` (the `condense()` guard) states it outright:

```python
# TODO(dp-aware-condense): skipped under DP because relocating a request
# breaks its slot->replica KV-cache affinity and corrupts decode. A
# DP-aware condense (within each replica's slot range) is the fix. #5896.
if removed_req_indices and not dp_active:
    self.input_batch.condense(removed_req_indices)
```

So "a request must keep its slot -> replica affinity or decode corrupts" is
established, previously observed behaviour with its own issue (#5896). What this
handoff documents is a **second, unhandled way to break the same invariant**:

| Mechanism | Relocates a request? | Status |
|---|---|---|
| `condense()` compaction | yes | known; disabled under DP (#5896) |
| **prefill/decode bucket change** | **yes** | **unhandled — measured here** |

Condense is *not* the cause here: it is disabled under DP, and in this repro no
request finished early so there were no holes to compact. Related prior work on
the same invariant: #5799, #5778.

## Reproduce in ~20 min

```bash
docker exec <container> bash -c '
  export TTXLA_SLOT_TRACE=1
  cd /path/to/tt-xla && source venv/activate
  python -m pytest -svv \
    "tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_slot_trace[mesh_shape0-8-16]"'
```
Then pair prefill (`tok>0`) vs decode device per request id from the
`SLOTTRACE` lines. `run_slot_trace.sh` runs both `pbt=16` and `pbt=0`.

Full-depth repro (~80 min): `test_dptp_devstral_mixed_4k[mesh_shape0-4096-8-16-None]`.

## Ruled out

- **Not the tt-mlir chunked-SDPA users factor (#9142).** Same binary produces
  correct output at `pbt=0`; zero `Result shape must match query shape` in any
  run. The users factor was separately validated: Qwen3-32B full depth batch 32
  @4096 returns on-topic content for a ~356-token prefix living only in the
  paged KV cache.
- **Not stale device state.** Reproduced byte-identically after a real
  host-side `tt-smi -glx_reset`.
- **Not padding rows clobbering block 0.** Padding rows zero their page_table
  and write to block 0, but vLLM reserves it
  (`BlockPool: self.null_block = self.free_block_queue.popleft()`).
- **Not layer-count dependent.** The mismatch is structural and shows at 2
  layers.

## Fix directions (not implemented)

1. Make device assignment independent of the row-count bucket, so a request's
   device is stable across prefill/decode.
2. Force prefill and decode to share a bucket when DP is active (e.g. drop the
   small prefill bucket under DP). Costs the b1 small-graph optimisation.
3. Keep the buckets but migrate/reshard a request's KV when its device changes.

Option 1 is the least invasive to the b1 feature's purpose.

## Traps for whoever picks this up

- **Do not judge output with a hand-rolled alpha ratio.** `str.isalpha()` is
  True for CJK, so multilingual token soup scores ~0.99 and reads as clean. This
  produced a false "hypothesis falsified" result mid-investigation. Use
  `assert_output_coherent` from `tests/integrations/vllm_plugin/conftest.py`
  (it checks non-Latin script ratio).
- **Truncated models emit soup regardless of any bug.** At 10 layers both Qwen
  and Devstral are pure garbage, so text-quality comparisons are worthless
  there. The slot trace works anyway because it is structural. This is also why
  the pre-existing `test_dptp_qwen` pins `num_hidden_layers=2` *and* has its
  coherence assert commented out.
- `pbt=0` is a **bisect lever, not a fix** — it disables b1 small-graph prefill,
  which exists for low-concurrency latency.
- `tt-smi` is **not in the container**; run resets from the host or they exit
  127 silently.
- `venv/activate` dereferences an unbound `LD_LIBRARY_PATH`, so any driver
  script with `set -u` dies instantly on sourcing it.

## Build used

| | |
|---|---|
| tt-mlir | `1bbc2eca5d` (branch `ssalice/chunked-sdpa-users-factor-ttxla-pin`, base `c91bfb2d45`) |
| tt-metal | `f1f4ff7557` |
| tt-xla | `ssalice/devstral-qwen-5893` (main `1864e78fa` + PR #5893 + #5799) |

`third_party/CMakeLists.txt` carries an **uncommitted** pin override to that
tt-mlir branch. tt-xla main has since moved to tt-mlir `724d2ff7b4`; both pin
the same tt-metal, so rebasing the fix onto main's pin is a tt-mlir recompile
only. The skew does not affect this bug (it is tt-xla-side and reproduces on a
2-layer 0.6B model).
