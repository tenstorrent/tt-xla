# Note to Sungjoon (sshonTT) re: #5786

Slack/comment draft. Not posted.

---

Hey, found something while working on spec decode that touches #5786.

`_prepare_inputs` builds its per pass lists starting from `start_index`, so inside
the function the loop index counts from 0 for that pass. But `input_batch` is
indexed globally across all requests. So any `input_batch[...][:actual_num_reqs]`
or `[i]` in there is only correct on the first pass, and reads the wrong requests'
state on later ones.

I hit this in the non sliding path (positions, input_ids gather, seq_lens, the
fill roll offsets, chunk_start_idx and both page table copies) and fixed it in
#5836 with a `req_slice = slice(start_index, start_index + num_reqs)`.

The sliding block in #5786 has the same pattern in a few places, including the
safety guard:

```python
nsched = np.asarray(num_scheduled_tokens_per_req[:actual_num_reqs])
num_computed = np.asarray(self.input_batch.num_computed_tokens_cpu[:actual_num_reqs])
filling = nsched > 1
bad = filling & ((start_block > 0) | (num_computed > 0))
if np.any(bad):
    raise NotImplementedError(...)
```

`num_scheduled_tokens_per_req` is already pass local so that part is fine, but
`num_computed_tokens_cpu[:actual_num_reqs]` is global. On a pass after the first
the guard compares the two against different requests, so it can fail to raise
when it should, or raise when it should not. A hybrid model with a batch big
enough to split across passes would be enough to hit it.

Not urgent for me: my changes never reach it, because the block is gated on
`any(self._group_is_sliding)` and a speculative row is refused by that guard on
the first pass anyway. Flagging it because the guard is load bearing for you and
the failure mode is silent.

Worth knowing the caps that split a batch into passes are the SMEM row cap and
`max_prefill_num_reqs`, so this is reachable without spec decode at all.

`req_slice` lands in #5836 if you want to reuse it.
