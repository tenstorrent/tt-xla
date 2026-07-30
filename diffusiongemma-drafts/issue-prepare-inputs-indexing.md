# [vLLM] _prepare_inputs reads input_batch with pass-relative indices, wrong on any pass after the first

Issue draft. Not filed.

## Summary

`TTModelRunner._prepare_inputs` handles one pass at a time over a slice of the
batch, `[start_index, end_index)`, and the caller loops until every request is
covered. It builds its per pass lists starting at `start_index`, so inside the
function the loop index counts from 0 for that pass. `input_batch` is indexed
globally across all requests. Mixing the two means every `input_batch` read using
a pass local index is only correct while `start_index == 0`.

On any later pass the runner reads a different request's state than the one it is
building inputs for. The page table is the worst of them: a request ends up
reading and writing another request's KV blocks.

## Why it has not been seen

Nothing reached a second pass in practice:

* a single request never leaves pass 0
* the two things that split a batch, the SMEM row cap and `max_prefill_num_reqs`,
  rarely trigger in tested configs

It surfaced while making speculative decode work (#5836). A speculative row must
sit on a single KV block boundary because the chunked SDPA prefix offset is one
shared value per pass, so batches now get split by boundary and multi pass became
the common case. The mixed boundary test kept producing wrong tokens for one
request until this was fixed.

## Affected reads

In the non sliding path, fixed in #5836 by introducing
`req_slice = slice(start_index, start_index + num_reqs)`:

* `positions` (`num_computed_tokens_cpu[i]`)
* the `input_ids` gather (`token_ids_cpu_tensor[i]`)
* `seq_lens`
* the `paged_fill_cache` roll offsets
* `chunk_start_idx`
* both page table copies from `block_table[g].get_cpu_tensor()`

The sliding window path added by #5786 has the same pattern and is not fixed. It
includes the guard that refuses unsupported sliding prefill:

```python
nsched = np.asarray(num_scheduled_tokens_per_req[:actual_num_reqs])   # pass local, fine
num_computed = np.asarray(self.input_batch.num_computed_tokens_cpu[:actual_num_reqs])  # global
bad = (nsched > 1) & ((start_block > 0) | (num_computed > 0))
```

On a later pass the two arrays describe different requests, so the guard can fail
to raise when it should. Left alone in #5836 to avoid editing a just merged
feature from an unrelated PR.

## Suggested follow up

* apply `req_slice` to the sliding path, including the guard
* add a unit test that drives `_prepare_inputs` with `start_index > 0` and asserts
  the positions, page table and seq_lens belong to the requests in that pass

The second is the useful part. Three of the bugs fixed in #5836 were findable
without hardware; there is currently no cheap test that forces a pass split.
