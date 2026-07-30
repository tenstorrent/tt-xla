### Ticket
Follow up for #5542.

### Problem description

Ngram speculative decode did not produce correct output on TT on either model
runner. It was never validated end to end: the existing coverage is CPU unit
tests of the rejection sampler and proposer with fake drafters, and nothing there
loads a model, so nothing ever ran attention on hardware.

Seven pre-existing bugs. Two came in with the 0.25.1 uplift and made it crash.
The rest made it silently emit wrong tokens.

### What's changed

Crashes:

* `NgramProposer.propose()` gained a leading `num_speculative_tokens` argument in
  0.25.1 and the call site was never updated. The existing test missed it because
  its fake drafter still had the old 3 argument signature.
* `rejection_sampler` cast dtype on device before copying to host, which tt-mlir's
  runtime memcpy rejects. Now transfers first and casts on the host.

Wrong tokens:

* A speculative row never reached the cached prefix SDPA path, so it attended only
  to its own row and ignored the KV cache. The gate required
  `prefill_chunk_budget < max_model_len`, which spec decode can never satisfy since
  it forces `max_num_batched_tokens >= max_model_len x max_num_seqs` while the
  platform derives that from `prefill_chunk_size`. Split so only the op's real
  requirement, block alignment, drives `chunk_start_idx`.
* `paged_fill_cache` takes no write position and starts at a block boundary, while
  the read offsets by `num_computed` exactly. Those agree only when `num_computed`
  is block aligned, which prefill chunks guarantee and a speculative row never is.
  Rows now extend left to their block boundary and within row indices shift with
  them.
* The accepted token writeback anchored on `num_computed + num_scheduled` instead
  of the committed count. Equivalent only when every draft is accepted; on a
  partial accept the tokens land in the wrong slots of `token_ids_cpu`, so the
  returned ids look right but the next step reads back corrupted context. Now
  anchors on `num_tokens_no_spec`, matching upstream.
* Per pass results were concatenated assuming a uniform width. A pass with drafts
  yields `[reqs, num_spec_tokens + 1]` and one without yields `[reqs, 1]`.

One pass may only carry rows on the same block boundary, since the prefix offset
is a single shared value, so `_prepare_inputs` trims to the leading run sharing
one and the multi pass loop takes the rest.

That exposed a bug unrelated to spec decode: `_prepare_inputs` builds its per pass
lists from `start_index` but read `input_batch` globally in several places, so any
pass after the first read the wrong requests' state. Latent because a single
request never leaves pass 0 and the row caps rarely split.

### Validation

A draft is only accepted when it matches what the target model would have sampled
anyway, so greedy plus spec decode must be token identical to plain greedy. That
is the new e2e test, and it needs no golden text because the non speculative run
is the reference.

On n300 with `facebook/opt-125m`: single request token identical, and a two
request batch straddling different KV block boundaries token identical per
request. Chunked prefill still passes, which is the shared multi pass path these
changes most affect. CPU tests pass, including new device free tests for the block
boundary arithmetic.

Each engine config in the e2e test runs in its own subprocess: two `vllm.LLM`
instances in one process leave the first EngineCore holding /dev/tenstorrent and
the second stalls.

### Checklist
- [x] New/Existing tests provide coverage for changes
