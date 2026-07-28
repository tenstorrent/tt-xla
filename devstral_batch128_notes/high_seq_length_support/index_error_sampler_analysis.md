# IndexError in `sample_tokens` on the on-device sampling path — root cause

**Symptom.** `test_dptp_devstral` with `cpu_sampling=False`, mesh `[4,8]` (DP=4×TP=8),
`max_num_seqs=128`, `temperature=0.8, top_p=0.95` crashes at
`integrations/vllm_plugin/vllm_tt/model_runner.py:2079`:

```python
token_id = valid_sampled_token_ids[i][0]   # IndexError: list index out of range
```

Forward runs, sampling runs, the crash is in the *post-sampling* bookkeeping. `cpu_sampling=True`
works.

---

## TL;DR root cause

The bug is **NOT** a DP-gather problem. It is a **hard-coded batch=32 cap in the on-device
(non-greedy) sampler** that *truncates* the batch whenever the decode batch exceeds 32.

- The `tt::sampling` kernel requires exactly batch=32
  (`sampler.py:34`, `_TTNN_SAMPLING_BATCH_SIZE = 32`).
- Both helper functions pad the batch **to** 32 with the expression
  `_TTNN_SAMPLING_BATCH_SIZE - batch`. When `batch > 32` this value is **negative**, and
  `torch.nn.functional.pad(..., (0,0,0, negative))` **removes rows** (a slice), silently
  shrinking the tensor to 32 rows.
- Decode always runs at `target_num_reqs = self.max_num_reqs` (`model_runner.py:1240`), which
  for this test is **128**. So the sampler receives `[128, vocab]`, truncates to `[32, vocab]`,
  and returns only **32** sampled token ids.
- Back in `sample_tokens`, `selected_token_ids.cpu()[:num_reqs]` (`model_runner.py:2010`) can
  therefore yield at most 32 rows. After `torch.cat`, `valid_sampled_token_ids` has ~32 entries
  while `self.input_batch.num_reqs` is up to 128, so `valid_sampled_token_ids[i]` for `i >= 32`
  is out of range → the IndexError at 2079.

`cpu_sampling=True` works because it takes a completely different sampler
(`sample_from_logits_cpu`, `model_runner.py:3540`) that has no batch cap and operates on the
full `[num_reqs, vocab]` logits after `logits.cpu()`.

The trigger is **batch > 32 concurrent decode requests + non-greedy sampling + cpu_sampling=False**.
DP/TP are *not* mechanically required — DP+TP is simply the config in which `max_num_seqs=128`
(so decode batch = 128 > 32). The `[4,8]` mesh also explains the "32 per device" wording in the
docstring's issue reference #4440 (128 / 4 DP replicas = 32), but the truncation itself is a
batch-size bug, independent of the mesh.

---

## 1. Where `selected_token_ids` comes from (cpu_sampling=False)

Loop in `sample_tokens` (`model_runner.py:1882-2016`). Per iteration `_prepare_inputs` returns
`(attn_metadata, logits_indices, num_reqs=actual_num_reqs, target_num_reqs, end_index)`
(returned at `model_runner.py:1526-1532`; note the 3rd element is `actual_num_reqs`, the 4th is
the padded batch `target_num_reqs`).

Decode step: `_model_decode` → `_model_decode_compiled` (`model_runner.py:2818-2857`):

```
hidden_states = self.model(...)                          # [target_num_reqs, seq, H]  (DP-sharded on dim0)
selected = self.select_hidden_states(hidden_states, logits_indices)   # [target_num_reqs, H]
logits   = self.compute_logits(selected)                 # [target_num_reqs, vocab]
selected_token_ids = self.sample_from_logits(logits, sampling_metadata)   # <-- the crux
```

- `select_hidden_states` (`3495`): `batch_indices = arange(logits_indices.shape[0])` = `arange(target_num_reqs)`,
  so the result has `target_num_reqs` rows. On the 2D mesh it applies
  `sharding_constraint_tensor(result, mesh, (None, None))` → **fully replicated** (`3498-3499`).
- `compute_logits` (`3509`): applies `sharding_constraint_tensor(logits, mesh, (None, None))`
  when `is_sharded_compute_logits` (true here — ParallelLMHead present) → **fully replicated**
  `[target_num_reqs, vocab]` (`3515-3516`).

So the logits handed to the sampler are the **full, replicated** `[128, vocab]` — the DP gather
already happens. The greedy branch of `sample_from_logits` (`3535`,
`torch.argmax(logits, -1, keepdim=True)`) would correctly return `[128, 1]`. The failure is
entirely inside the **non-greedy** branch (`3537`, `self.sampler(...)`), which is taken because
`temperature=0.8, top_p=0.95` ⇒ `all_greedy=False`.

### Decisive evidence that the sampler sees the full [128], not a per-replica [32] (settles #1/#2)

The task's leading hypothesis — "the on-device sampler returns the DP-sharded per-replica token
ids without gathering" — is **wrong**, and there is a single fact that settles it:
`_precompile_sample_from_logits` (`model_runner.py:3131`) compiles the sampler graph with
`dummy_logits = [max_num_reqs=128, vocab]`. The compiled graph literally takes `[128, vocab] →
[32, 1]` **at compile time, independent of runtime sharding**. Under the DP-gather hypothesis the
compiled graph would be `[32, vocab] → [32, 1]` and there would be no truncation. Because the
graph is `[128, vocab]` in and 32 out, the shortfall cannot be per-replica sharding — it is the
in-graph 32-row truncation described below. (Confirmed further by `compute_logits`/
`select_hidden_states` applying `(None, None)` replication constraints, so the logits are gathered
to full width before the sampler.)

### The truncation, precisely

`sample_from_logits` → `Sampler.forward` → `Sampler.sample` (`sampler.py:165`) → non-greedy path:

1. `chunked_topk_candidates(logits)` (`sampler.py:365-418`), with `batch = 128`:
   ```python
   logits = torch.nn.functional.pad(
       logits, (0, 0, 0, _TTNN_SAMPLING_BATCH_SIZE - batch), value=float("-inf"))
       #                        ^ 32 - 128 = -96  ==> removes 96 rows ==> [32, vocab]
   ...
   return all_values[:batch], all_indices[:batch]   # batch=128 but only 32 rows exist ==> 32 rows
   ```
2. `_ttnn_sampling_padded(...)` (`sampler.py:307-362`): now `batch = filtered_logits.shape[0] = 32`,
   so the `if batch < 32` pad is skipped, the kernel runs on 32 rows, `result[:32]` → 32 rows.
3. `SamplerOutput.sampled_token_ids = sampled.unsqueeze(-1)` → `[32, 1]`.

Result: `selected_token_ids` has **32** rows regardless of the 128 it was asked for.

> Note: this shape is baked in at warmup. `_precompile_sample_from_logits`
> (`model_runner.py:3119-3149`) compiles the sampler graph with
> `dummy_logits = [max_num_reqs=128, vocab]`, so the compiled graph itself emits a 32-row output
> and is reused verbatim at runtime.

---

## 2. Why cpu_sampling=True gives full rows and False does not

| | selected_token_ids producer | batch cap? | rows returned |
|---|---|---|---|
| `cpu_sampling=True`  | `_model_unfused` → `sample_from_logits_cpu` (`model_runner.py:3540`) | none | full `target_num_reqs` |
| `cpu_sampling=False` (greedy) | `sample_from_logits` argmax (`3535`) | none | full `target_num_reqs` |
| `cpu_sampling=False` (non-greedy) | `sample_from_logits` → `Sampler` → `tt::sampling` | **32 (truncates)** | `min(32, target_num_reqs)` |

`sample_from_logits_cpu` does `logits = logits.cpu()` (`3549`) and then a pure-torch
argmax/top-k/top-p/Gumbel over the full `[num_reqs, vocab]` tensor — no batch=32 constraint. That
is the only reason the CPU path survives batch>32. It is **not** doing an extra DP all-gather that
the device path omits; both paths receive already-replicated logits. The difference is solely the
32-row kernel constraint in the device sampler.

---

## 3. `num_reqs` vs `target_num_reqs` and the DP round-up

- `_prepare_inputs` sets decode `target_num_reqs = self.max_num_reqs` (`1240`) and returns
  `actual_num_reqs` as the `num_reqs` used for the `[:num_reqs]` slice (`1529`).
- `max_num_reqs` is rounded up to a multiple of `dp_size` (`model_runner.py:326-336`). For
  `max_num_seqs=128, dp_size=4` it is already 128, so the round-up is a no-op here — it is **not**
  the cause.
- Under normal operation `sum(actual_num_reqs)` over loop iterations equals
  `self.input_batch.num_reqs` (line 2036), so the concatenation would be the right length **if the
  device tensor were full width**. It is the sampler truncation that makes each iteration's
  `.cpu()[:num_reqs]` short (it can only slice out ≤32 rows), so `torch.cat` is short and the outer
  index overruns.
- Important nuance: the crash needs **actual** decode `num_reqs > 32`, not merely
  `max_num_reqs > 32`. If ≤32 requests are actually in flight, `[:num_reqs]` on the 32-row tensor
  still returns `num_reqs` rows and there is no error (the lost rows would be silent wrong-token
  corruption — cf. the "token soup" in #4440 — but no IndexError). The 128-prompt test drives >32
  concurrent decodes, so it crashes.

---

## 4. Root-cause hypothesis (crisp)

`_TTNN_SAMPLING_BATCH_SIZE = 32` is used as a *fixed* pad target in `chunked_topk_candidates`
(`sampler.py:384-388`) and `_ttnn_sampling_padded` (`sampler.py:351-359`). The pad amount
`32 - batch` is negative when `batch > 32`, so `torch.nn.functional.pad` **truncates** the
batch to 32. Decode runs at `target_num_reqs = max_num_reqs = 128`, so the non-greedy on-device
sampler returns only 32 token ids. `sample_tokens` then builds a `valid_sampled_token_ids` list
of ~32 entries and indexes it up to `self.input_batch.num_reqs` (up to 128) at
`model_runner.py:2079` → `IndexError`.

---

## 5. Minimal repro design

The bug needs: **non-greedy sampling + cpu_sampling=False + more than 32 concurrent decode
requests.** It does **not** need the 123B model, and strictly does **not** need DP or TP — a
single-chip run with `max_num_seqs > 32` reproduces the truncation. Use DP+TP only if you want to
stay faithful to the reported galaxy path.

### Fastest repro (single chip or small mesh), Qwen3-0.6B

```python
model_name = "Qwen/Qwen3-0.6B"
prompts = ["Continue in English: The weather today is"] * 64   # > 32 concurrent
sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=8)  # non-greedy
llm_args = {
    "model": model_name,
    "max_num_seqs": 64,          # decode batch padded to 64 > 32  -> truncation to 32
    "max_model_len": 128,
    "max_num_batched_tokens": 2048,
    "gpu_memory_utilization": 0.4,
    "additional_config": {
        "num_hidden_layers": 2,   # keep it tiny/fast
        "cpu_sampling": False,    # REQUIRED to hit the on-device sampler
        "optimization_level": 0,  # avoids the separate #4387 trace crash; not needed for this bug
        # single-chip: omit parallel flags entirely, OR for the DP+TP variant on an 8-chip llmbox:
        # "enable_data_parallel": True,
        # "enable_tensor_parallel": True,
        # "mesh_shape": [2, 4],
    },
}
```

- **Minimum to reproduce:** `max_num_seqs = 64` (any value > 32) and **≥ 33 prompts submitted so
  that >32 requests decode simultaneously**. With `max_num_seqs=64` and 64 prompts the first
  decode step has 64 active requests → 64 > 32 → IndexError.
- **DP size / TP size:** neither is required. The single-chip case above is a *predicted minimal
  isolation* (not yet run on hardware): with parallelism disabled, decode still runs at
  `target_num_reqs = max_num_reqs = 64`, so the same truncation fires. This prediction is the
  strongest *argument* that the bug is not DP-dependent — it removes DP and TP as variables. Run
  it single-chip first (cheapest, disproves DP-dependence); keep the `mesh_shape=[2,4]` DP+TP
  variant alongside as the config faithful to the reported galaxy run.
- **Decode-split caveat:** the crash needs the *decode iteration's* actual batch to exceed 32,
  i.e. the `_prepare_inputs` while-loop must not sub-split the decode batch below 32 (it can trim
  under SMEM pressure via `num_reqs_max_model_len`, `model_runner.py:1194-1211`). A 2-layer 0.6B
  model has no such pressure, so all >32 requests decode in one iteration and the bug fires;
  choosing a tiny model/short context keeps this guaranteed.
- **Adapting the existing test:** `test_data_tensor_parallel_generation_wider_batch`
  (`tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py:65`) is the
  closest scaffold, but it uses `max_num_seqs=4` and `cpu_sampling=True`. Bump `max_num_seqs` to
  e.g. 40, submit ≥40 prompts, set `temperature=0.8, top_p=0.95`, and set `cpu_sampling=False`.

---

## 6. Candidate fix(es)

### Real fix (in `sampler.py`): tile the batch into groups of 32

The `tt::sampling` kernel genuinely requires batch=32, so the sampler must process batch in tiles
of 32 and concatenate, instead of assuming batch ≤ 32.

**(a) `chunked_topk_candidates` (`sampler.py:365-418`)** — the topk itself works for any batch; only
the pad-to-32 is wrong. Pad *up to the next multiple of 32* (never down):

```python
import math
batch = logits.shape[0]
padded_batch = max(_TTNN_SAMPLING_BATCH_SIZE,
                   math.ceil(batch / _TTNN_SAMPLING_BATCH_SIZE) * _TTNN_SAMPLING_BATCH_SIZE)
logits = torch.nn.functional.pad(logits, (0, 0, 0, padded_batch - batch), value=float("-inf"))
...
return all_values[:batch], all_indices[:batch]   # slice back to true batch (now correct for batch>32)
```

**(b) `_ttnn_sampling_padded` (`sampler.py:307-362`)** — run the kernel per 32-row tile and concat:

```python
batch = filtered_logits.shape[0]
outputs = []
for tile_start in range(0, batch, _TTNN_SAMPLING_BATCH_SIZE):
    tile_end = min(tile_start + _TTNN_SAMPLING_BATCH_SIZE, batch)
    v = values[tile_start:tile_end]; idx = indices[tile_start:tile_end]
    k = k_tensor[tile_start:tile_end]; p = p_tensor[tile_start:tile_end]; t = temp_tensor[tile_start:tile_end]
    n = tile_end - tile_start
    if n < _TTNN_SAMPLING_BATCH_SIZE:
        pad = _TTNN_SAMPLING_BATCH_SIZE - n
        v = F.pad(v, (0,0,0,pad), value=float("-inf")); idx = F.pad(idx, (0,0,0,pad))
        k = F.pad(k, (0,pad), value=1); p = F.pad(p, (0,pad), value=1.0); t = F.pad(t, (0,pad), value=1.0)
    res = torch.ops.tt.sampling(v, idx, k, p, t)
    outputs.append(res[:n])
return torch.cat(outputs, dim=0).to(torch.int64)
```

Note the `[:batch]` guards at `sampler.py:328/338/344` (`top_k[:batch]` etc.) already tolerate
`batch>32`; only the pad/return logic needs the tiling. Because the sampler graph is compiled at
`max_num_reqs` (`3119-3149`), the tiled loop will be traced at the full 128 width once at warmup —
consistent shapes, no runtime recompile.

**Confirming evidence the fix is at the right layer:** once tiling makes the sampler return the
full 128 rows, it also incidentally removes a latent shape mismatch in the mixed greedy/random
path — `torch.where(temp < eps, greedy_sampled=[128], random_sampled=[32])` in `Sampler.sample`
(`sampler.py:216-220`). This test dodges that mismatch only because `all_random=True` short-circuits
before the `torch.where` (`sampler.py:214-215`); a mixed batch would otherwise broadcast-fail.
The tiling fix makes `greedy_sampled` and `random_sampled` both `[128]`, so the layer is
internally consistent afterward.

### Mask / stop-gap (not a real fix)

- Force `cpu_sampling=True` whenever `max_num_reqs > 32` with non-greedy params. Hides the bug,
  loses the fused-kernel perf, and is what the other tests already do — this is exactly the
  "cpu_sampling=True is REQUIRED" workaround in the `test_dptp_devstral` docstring
  (`test_...:349-353`).
- Assert `target_num_reqs <= 32` in the sampler. Turns silent truncation into a loud failure but
  does not enable batch>32.

The tiling fix (a)+(b) is the proper fix; the cpu_sampling fallback is a mask.

> Caveat: fixing the shape truncation may then expose the separate **#4440 "token-soup"**
> correctness issue (wrong tokens per replica on the 2D mesh). That is a distinct problem from this
> IndexError; the shape fix is a prerequisite for even testing it.

---

## 7. Instrumentation to confirm on a device run

**In `sampler.py`, top of `chunked_topk_candidates` and `_ttnn_sampling_padded`
(the `after pad` print is the single load-bearing confirmation — it directly shows the negative
pad truncating on-device, the one link in the chain not proven by static reading):**
```python
print(f"[SAMPLER] chunked_topk_candidates: logits.shape={logits.shape}, batch={logits.shape[0]}, "
      f"pad_amount={_TTNN_SAMPLING_BATCH_SIZE - logits.shape[0]}")
# after the pad:  <-- LOAD-BEARING: expect [32, vocab] when batch>32
print(f"[SAMPLER] after pad: logits.shape={logits.shape}  (<-- shrinks to 32 if batch>32)")
```
```python
print(f"[SAMPLER] _ttnn_sampling_padded: in batch={filtered_logits.shape[0]}")
# before return:
print(f"[SAMPLER] _ttnn_sampling_padded: out rows={result[:batch].shape[0]}")
```

**In `sample_from_logits` (`model_runner.py:3519`), before return:**
```python
print(f"[SAMPLE] sample_from_logits: logits={tuple(logits.shape)} -> out_tokens={tuple(out_tokens.shape)} "
      f"all_greedy={sampling_metadata.all_greedy}")
```

**In `sample_tokens` loop (`model_runner.py`), just after line 2010 and after 2018/2036:**
```python
print(f"[LOOP] iter start_index={start_index} end_index={end_index} "
      f"num_reqs(actual)={num_reqs} target_num_reqs={target_num_reqs} dp_size={self.dp_size} "
      f"selected_token_ids.cpu()[:num_reqs].shape={tuple(selected_token_ids.shape)}")
# after cat (2018):
print(f"[CAT] combined len={len(combined_selected_tokens)} "
      f"cat rows={selected_token_ids.shape[0]}")
# after 2036:
print(f"[POST] self.input_batch.num_reqs={num_reqs} "
      f"len(valid_sampled_token_ids-to-be)={selected_token_ids.shape[0]}")
```

**Expected signature of the bug:** with `max_num_seqs=128` and >32 concurrent decodes you will see
`chunked_topk_candidates` receive `batch=128`, `pad_amount=-96`, and `after pad: [32, vocab]`;
`sample_from_logits` returns `out_tokens=(32, 1)`; the `[CAT]`/`[POST]` line shows `cat rows=32`
while `self.input_batch.num_reqs` is up to 128 → the `[i][0]` overrun at line 2079.
