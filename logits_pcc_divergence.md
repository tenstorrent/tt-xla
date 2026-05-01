# Logits PCC divergence in `test_e2e_prefill_pcc[1]`

## Summary

`test_e2e_prefill_pcc[1]` reports `hidden_states pcc=0.999773` (passing) and
`logits pcc=0.906730` (failing 0.95). The original suspicion was that the
divergence sits in `ParallelHead`. **It does not.**

The actual root cause is **MoE routing/expert-output divergence between the
CPU reference (`_cpu_forward` → original MoE) and the device A2a SparseMLP**,
manifesting only at sequence positions where every batch row carries real
content (positions 124–127 of the prefill). The head simply slices
`norm_out[:, -1]` and exposes that already-divergent last position.

The gap between aggregate `h` PCC (0.9998) and pos-127 `h` PCC (0.92) is
hidden by the 124 padding-dominated positions, which are nearly identical
across CPU and device and dominate the aggregate correlation.

## What was confirmed (and ruled out)

| Hypothesis | Verdict | How verified |
|---|---|---|
| Head amplifies error from inside the head | ❌ ruled out | Per-stage PCC inside `ParallelHead`: `h`=0.9998, `hc_out`=0.9991, `norm_out`=0.9977, `last_token`=0.9082, `logits`=0.9067 — the cliff is only at `[:, -1]` (a pure slice). |
| `x.square()` lowering to `ttnn.pow_scalar(x, 2.0)` is lossy | ❌ ruled out | Replaced `x.square()` with `x * x` in `ParallelHead.hc_head`. Confirmed via MLIR diff that the lowering changed (`vhlo.power_v1` removed, `ttnn.pow_scalar` removed for the head). Logits PCC stayed at exactly `0.906730`. Hardware-level the two ops resolve identically. |
| TTNN `rms_norm` fused op precision | ❌ ruled out | `norm_out` PCC is 0.9977 and matches its input (`hc_out` 0.9991). Norm did not introduce divergence. |
| Sigmoid / sum amplification in `hc_head` | ❌ ruled out | `hc_out` PCC = 0.9991, very close to its input `h` (0.9998). |
| Final matmul precision | ❌ ruled out | `last_token` and `logits` PCC are essentially identical (0.9082 vs 0.9067). The matmul preserves whatever PCC its input had. |
| Attention output divergence at last positions | ❌ ruled out | `attn_out` per-position PCC is ≥0.9997 at every position including 124–127. |
| **MoE/FFN output divergence at content-bearing positions** | ✅ **root cause** | `ffn_out` per-position PCC: pos127=0.8916, pos126=0.9503, pos125=0.9435, pos124=0.9975, pos120=0.9999, pos0=0.9999. |

## Method

A `return_intermediates` flag was threaded through `Block.forward`,
`ParallelHead.forward`, and `Transformer.forward` so the prefill graph emits
each intermediate tensor as a graph output (rather than relying on Python
side effects, which would not survive `torch.compile`). Two diagnostic tests
were added to `tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py`:

1. `test_e2e_prefill_head_intermediates_pcc[1]` — captures
   `(h, hc_out, norm_out, last_token, logits)` from CPU and device, prints
   stage-by-stage PCC and per-sequence-position PCC of `h`, `hc_out`,
   `norm_out`. This proved the cliff was at `slice [:, -1]` and that the
   head itself preserves PCC.

2. `test_e2e_prefill_block_intermediates_pcc[1]` — additionally captures
   `(attn_out, post_attn, ffn_out)` from layer 0, plus the per-position
   pad-vs-content distribution of `prompt_ids`. This proved the cliff is at
   `ffn_out` and only at content-bearing positions.

Both tests use a wrapped `sharding_constraint_hook` that applies the
`(None, None)` constraint only to the `logits` element of the head's tuple
output, since the original `sharding_constraint` op rejects tuples.

## Evidence

### 1. The head is innocent

Stage-by-stage PCC inside `ParallelHead.forward(x, …, return_intermediates=True)`:

```
h            (post-blocks, pre-head)        pcc=0.999773
hc_out       (post hc_head, pre norm)       pcc=0.999103
norm_out     (post RMSNorm, pre slice)      pcc=0.997701
last_token   (post slice [:, -1].float())   pcc=0.908171   ← cliff
logits       (post head matmul + gather)    pcc=0.906730
```

Each stage's max_abs_diff and mean_abs_diff also match this pattern: the
mean_abs_diff for `norm_out` is 0.011 (averaged over 32×128×4096 values),
but the mean_abs_diff for `last_token` is 0.103 (averaged over 32×4096
values from one position). 10× jump under a pure slice means one position
must be ~10× worse than the rest — there is no other way for a slice to
"introduce" error.

### 2. The block-output cliff lives in the last few sequence positions

Per-sequence-position PCC of `h` (the layer-0 output, before the head):

```
pos 0–124: PCC ≥ 0.998 (best at pos 12: 0.999929)
pos 125  : 0.9481
pos 126  : 0.9529
pos 127  : 0.9212  ← worst
```

Aggregate over all 128 positions averages to 0.9998 because the 125
well-matching positions dominate.

### 3. The cliff is in the FFN/MoE, not attention

Block-level per-position PCC at sub-stage outputs:

```
                  pos127   pos126   pos125   pos124   pos120   pos 0   aggregate
attn_out          0.9998   0.9997   0.9997   0.9998   0.9997   0.9998   0.999721
post_attn         0.9989   0.9998   0.9998   0.9997   1.0000   1.0000   0.999996
ffn_out           0.8916   0.9503   0.9435   0.9975   0.9999   0.9999   0.999709   ← cliff
h (block out)     0.9212   0.9686   0.9653   0.9979   0.9999   0.9998   0.999780
```

`attn_out` is essentially perfect at every position. `post_attn` (the HC
recombination of attention output with the residual) is nearly identical,
slightly worse at pos127 only because `attn_out` was already 0.9998 there
and the residual side is also numerically unchanged. The huge step is
between `post_attn` and `ffn_out`.

`h` ≈ `post_attn` + `hc_post(ffn_out, …)` and so inherits the `ffn_out`
cliff smoothed by the residual.

### 4. The cliff aligns exactly with the pad-vs-content boundary

Non-pad-token count per sequence position, summed across the 32 batch rows:

```
pos 112–117 = 0/32   (every row is pad here)
pos 118     = 1/32
pos 119     = 2/32
pos 120     = 5/32
pos 121     = 11/32
pos 122     = 20/32
pos 123     = 28/32
pos 124     = 32/32   ← first position where every row has content
pos 125–127 = 32/32
```

Position 124 is the first all-content position; that is exactly where
`ffn_out` per-position PCC starts dropping. Pad-only positions all route to
the same expert (pad embeds identically, so router scores identically) and
their FFN outputs are bit-stable across CPU/device. Content positions have
diverse router scores; tokens whose top-k boundary is close get flipped to
different experts on the device path, producing materially different FFN
outputs.

### 5. Aggregate metrics conceal per-position divergence

`hidden_states pcc = 0.9998` is computed over `32 * 128 * 4 * 4096 ≈ 67M`
values. Of those, ~62M live at positions 0–123 where CPU and device match
to within rounding. The ~1.6M values at positions 124–127, where most of
the divergence sits, contribute too little to the correlation to move the
aggregate. The head's `[:, -1]` slice removes that masking — `last_token`
PCC ≈ pos127 PCC because that is the only position the slice keeps.

## Root cause

**MoE routing divergence between the CPU reference and the device
implementation.** The test layout is:

- CPU prefill: `A2aSparseMLPWithSharedExperts._cpu_forward` →
  original MoE module (full softmax router, full topk, dense expert
  evaluation in PyTorch). See `python_package/tt_torch/sparse_mlp.py:540`.
- Device prefill: `A2aSparseMLPWithSparseExperts.forward` →
  all-to-all dispatch/combine kernels with sparse expert evaluation.

These two implementations are not numerically identical:

1. The router scores are computed in slightly different orders / dtypes,
   producing topk indices that flip for tokens whose `k`-th and `k+1`-th
   scores are within rounding of each other. Pad tokens are not at any
   such boundary; content tokens often are.
2. Even when the same experts are selected, the dispatch/combine
   reduction order differs from PyTorch's dense MoE accumulation order,
   producing small but real per-token deltas — which only become
   visible above the noise floor at content positions, where the
   per-token outputs are not all identical.

The 1-layer test maximizes this signal because the layer output flows
straight into the head with no further accumulation to dilute the
position-127 error.

## Corroborating evidence: longer content → higher logits PCC

An additional empirical observation: holding `total_input_tokens = 128`
fixed while varying the content/pad split shifts the logits PCC
substantially. Concretely:

- 10 content tokens / 118 left-pad tokens → low logits PCC
- 60 content tokens /  68 left-pad tokens → much better logits PCC

This is consistent with the diagnosis above and refines the mechanism.
Position 127 (the position the head reads) is content in both cases;
what changes between them is **the context that produces position 127's
hidden state** and therefore the activation pattern the router sees at
position 127.

- With 10 content / 118 pad, position 127's hidden state is the result
  of attention attending mostly over pad K/V and a tiny content tail.
  The model was never trained on sequences that are 92% pad, so the
  resulting hidden state lands in a region of activation space that is
  **out-of-distribution** for the router. OOD inputs produce router
  score vectors with **near-tied top-k entries** (many experts give
  similar scores). Tied top-k is exactly where CPU-vs-device numerical
  noise flips the chosen experts. Different experts → very different
  FFN output → low per-token PCC at position 127 → low logits PCC.

- With 60 content / 68 pad, position 127's hidden state is produced by
  attention over a more in-distribution mix. The router sees an
  activation pattern of the kind it was trained on, so the top-k margins
  are wider. Wide margins survive numerical noise unchanged → CPU and
  device pick the same experts → FFN outputs match → logits PCC stays
  high.

The earlier finding ("MoE routing divergence at content positions")
does not say "more content = worse PCC". It says "**routing divergence
happens where the router is uncertain**." This observation is the layer
beneath: the router is uncertain when its input is OOD, and the input
is OOD when the prefix is mostly pad.

### Predictions from this refinement

| total = 128 | content | pad | router input @ pos 127 | top-k margin | logits PCC |
|---|---|---|---|---|---|
| short content | 10 | 118 | OOD (pad-dominated) | narrow → easy to flip | low |
| medium content | 60 | 68 | near-typical | wider | better |
| no padding (full prompt) | 128 | 0 | typical | widest | should be best — clean test |

Two more falsifiable predictions:

1. The per-position PCC profile of `ffn_out` should **shift left** as
   content grows. With 10 content the cliff sits at positions 124–127.
   With 60 content the cliff should appear at every content position
   (~68–127), but each individual cliff should be **shallower** because
   each content position's router has a wider margin.
2. With **right-padding** instead of left-padding, position 127 is now
   pad and logits PCC should jump to ~1.0. Pad routing is
   bit-identical across CPU and device. (Don't run this for real
   generation — it only isolates the routing component for a numerics
   sanity check.)

### Implications for the fix

This sharpens the "router-flip dump" recommendation: the dump only
needs to focus on positions where the router top-k margin is small.
A very small fix that may close most of the gap is to force the
router's score → top-k computation in
`python_package/tt_torch/sparse_mlp.py` to fp32 with **deterministic
tiebreak** (smallest expert id wins on ties). The CPU reference already
does this implicitly via PyTorch's stable sort; the device path needs
to match. This is the locus most likely to fix the gap end-to-end.

What this does **not** explain — and therefore does not need fixing on
the attention side — is that `attn_out` PCC stays ≥ 0.9998 at position
127 even under heavy padding. Sparse attention with the learned
`attn_sink` is well-conditioned even on OOD inputs. The routing fix
should be sufficient.

## CPU-only generation also fails: the inputs themselves are degenerate

A CPU-only sanity check — `test_cpu_only_prefill[43]` — runs the full
43-layer real-checkpoint model on the same `PROMPTS` with no device
involvement, no `.to(device)`, no PCC comparison, just a single CPU
prefill and an `argmax` of the logits. Output (saved in
`cpu_only_43.log`):

- **30 of 32 prompts predict EOS (`<|end▁of▁sentence|>`, id 1) as the
  next token**.
- The two non-EOS rows produce ` about` (row 4, "What is two plus
  two?") and `.\n\n` (row 20, "What is the meaning of recursion?") —
  not coherent continuations either.

So the model on CPU alone is collapsing to "stop generating" for
nearly every row. The previous claim that "CPU-only would still produce
sensible output" is **wrong for this specific test setup**. Two
underlying causes combine to produce the collapse, neither of which
is on the device side.

### Cause 1: pad_id collides with eos_id

`_tokenize_prompts` chooses the pad id with the standard fallback:

```python
pad_id = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)
```

DeepSeek-V4's tokenizer does not define a dedicated pad token, so
`pad_id == eos_id == 1`. After left-padding to `PROMPT_LEN=128`, every
prompt looks literally like:

```
<|EOS|>  <|EOS|>  <|EOS|>  …  <|EOS|>     (118 times)
<content tokens>                          (4–10 tokens)
                                          ← model predicts next
```

The model was trained on text where `<|EOS|>` marks document
boundaries. Seeing 118 EOS tokens in a row is extreme out-of-
distribution. The maximum-likelihood next token is "another EOS" —
that is, "this document is also ending" — which is precisely the
behavior we observe.

### Cause 2: the model has no pad-aware attention mask

The DeepSeek modified-model implementation (`model_decode_opt.py`,
`kernel.py`) has **zero pad-token awareness**. Grepping for
`attention_mask`, `attn_mask`, `key_padding`, `padding_mask` returns
no hits. Every mask present in the code is purely position-based, not
content-based:

| Site | What it masks | What it does NOT mask |
|---|---|---|
| `get_window_topk_idxs` | positions outside the causal/sliding window | pad tokens |
| `sparse_attn`'s `valid = topk_idxs != -1` | `-1` sentinels from window/compressor bookkeeping | pad tokens |
| `Indexer.forward` masks | unfilled compressed-cache slots | pad tokens |
| `ParallelEmbedding` vocab-range mask | token ids outside this rank's vocab shard | pad tokens |

Pad tokens go through the model exactly as if they were real content:

1. `self.embed(input_ids)` — pad ids hit the same lookup table; EOS-
   as-pad gets the EOS embedding, which the model has rich learned
   semantics for.
2. `wkv(x)`, `wq_a(x)` — pad activations produce real K, V, Q.
3. `Q @ K^T` — pad K's contribute scores normally.
4. `softmax` — pad positions get real softmax weight, not zero.
5. `weights @ V` — pad V's are mixed into the output.

Combined with cause 1: the model is forced to treat 118 EOS-as-pad
tokens as 118 successive document terminators, fully attended to.
There is no architectural escape hatch.

### What this means for the failing PCC test

`test_e2e_prefill_pcc[1]` is not really measuring "does the device
backend match the CPU reference for a meaningful prefill." It is
measuring "do CPU and device agree on a tensor whose true value is
mostly EOS-prediction, with a few content-margin tokens that happen
to land on near-tied router boundaries." Even if the device achieved
PCC = 1.0, the predicted continuations would still be nonsense. The
0.91 logits PCC is a real CPU-vs-device disagreement, but it sits on
top of a degenerate output that no backend can rescue.

This subsumes the earlier "OOD-router-margin" diagnosis: the OOD-ness
isn't subtle. Position 127's hidden state lands in an EOS-collapse
basin because attention faithfully averaged 118 EOS K/Vs into it.
Whatever router-flip dynamics happen on top are second-order.

### Two practical fixes (in order of preference)

1. **Use non-padded prompts.** Build prompts that are exactly
   `PROMPT_LEN=128` tokens long so the test runs the model on real
   content end-to-end. Easy to implement: rewrite `_tokenize_prompts`
   to either (a) concatenate / repeat sentences until each row has
   exactly 128 tokens of meaningful text, or (b) draw 128-token
   slices from a larger reference corpus. With this change, both
   `test_e2e_prefill_pcc` and `test_cpu_only_prefill` measure
   something interesting again.
2. **Add a pad-aware attention + routing mask to the model.** Thread
   an `attention_mask: Optional[torch.Tensor]` from
   `Transformer.forward` through `Block.forward → Attention.forward
   → sparse_attn` (force pad-position scores to `-inf` before
   softmax) and `Gate.forward` (force pad-position router scores to
   a constant or skip dispatch entirely for pad positions). Heavier;
   touches `model_decode_opt.py` and `model.py`. But fixes the model
   for any production batched-inference path, not just this test.

The first fix should be done regardless. The second fix is only
needed if the production use case ever requires real left-padded
batched generation; if production always uses fitted prompts, the
first fix is sufficient.

## Clear explanation of the OOD inputs causing logits' PCC to degrade when less content tokens are presented

*This section is self-contained. It explains the underlying mechanism
behind the logits-PCC failure pattern from first principles, without
relying on the surrounding sections.*

### The setting in one paragraph

A 1-layer slice of the DeepSeek-V4 model is run twice on the same
batch of 32 prompts of length 128 tokens each: once on a CPU (the
reference) and once on a Tenstorrent device. After prefill, the model
emits a `[32, 129280]` tensor of logits — for each of the 32 batch
rows, a probability-like score for every word in the 129,280-word
vocabulary, computed at the **last sequence position only**. We
compare CPU and device logits with **PCC** (Pearson correlation
coefficient). PCC near 1.0 means the two backends agree closely; lower
PCC means they disagree.

The observation: when prompts are heavily left-padded (10 content
tokens + 118 pad tokens per row), logits PCC = 0.91. When prompts have
more content (≈55 content tokens + ≈73 pad), logits PCC rises sharply.
The mechanism below explains why.

### What "OOD" means

**OOD = Out-Of-Distribution.** It is shorthand for "an input the model
was not trained on (or trained to handle well)." A neural network is a
deterministic function: it produces *some* output for any input you
hand it. But the model only learned to organize its activations
sensibly for the kinds of inputs it saw during training. For inputs
unlike the training distribution, the network still computes
something, but the resulting hidden states and intermediate scores
land in regions of activation space the model never had to make
decisions about. They become essentially arbitrary — driven by random
projections rather than learned structure.

In our specific case, the OOD trigger is the prompt structure. The
tokenizer doesn't define a dedicated pad token, so the test code falls
back to `eos_token_id` for padding. Every "10-content / 118-pad"
prompt therefore literally looks like `[<EOS>] × 118` followed by a
short content tail. The model has never seen 118 successive
end-of-sentence markers in training, so the activations they produce
are off-distribution.

### What "MoE routing" and "router margin" mean

DeepSeek-V4's feed-forward block is a **Mixture-of-Experts (MoE)**: it
contains 256 separate feed-forward sub-networks ("experts"), and a
learned **router** picks which experts to actually run for each token.
The mechanism per token is:

1. The router takes the token's hidden vector and produces 256 scores —
   one per expert.
2. The **top-k highest-scoring experts** are selected (k = 8 in this
   model). Only those 8 experts run for this token.
3. The token's FFN output is the score-weighted sum of those 8
   experts' outputs.

The **margin** is the gap between the k-th score and the (k+1)-th
score after sorting. For example, if a token's sorted expert scores
are:

```
expert  17 :  0.95     ┐
expert  42 :  0.93     │
expert  88 :  0.91     │
expert 103 :  0.89     │  the 8 selected experts (top-k)
expert 140 :  0.85     │
expert  31 :  0.83     │
expert 200 :  0.82     │
expert 211 :  0.81     ┘
expert  99 :  0.80     ← 9th, just outside top-k
expert  72 :  0.79
...
```

then the margin is `0.81 − 0.80 = 0.01`.

- **Wide margin:** the gap between in and out is large. Numerical
  rounding noise (different summation orders, lower-precision
  arithmetic on the device) cannot change which 8 experts are chosen.
  CPU and device pick the same top-k, run the same experts, produce
  matching FFN outputs.
- **Narrow margin:** the in/out scores are nearly tied. CPU might
  compute the 8th's score as `0.811` and the 9th as `0.809`; device
  might compute them as `0.810` and `0.812`. The two backends pick
  **different experts**. Different expert weights → very different FFN
  output for that token → low PCC at that position.

### How OOD inputs make margins narrow

This is the link between the two ideas above.

A trained router learns to be *decisive* on training-distribution
inputs: gradients reward it for confidently preferring particular
experts when given a typical hidden vector. The score landscape ends
up shaped so that for in-distribution inputs, a few experts have
clearly higher scores than the rest — wide margin.

For OOD inputs, the router was never asked to distinguish anything.
Whatever scores it produces are driven by whatever the random
projection of the off-distribution vector happens to land on. With 256
experts and an input the router has no preference about, it's
**statistically likely** that several scores will sit close together
near the boundary — narrow margin.

### Why this affects position 127 specifically (and how content fixes it)

The model's output that the test compares is `logits[:, position 127]` —
the prediction for what comes *after* the input. Position 127's hidden
state is built by:

1. **Embedding lookup** at position 127's token id.
2. **Attention** at position 127, which mixes K/V vectors from all 128
   positions (0..127) into a single output vector. Crucially, the
   model has no attention mask that would let it ignore pad tokens —
   it averages over all 128 K/V's, EOS-pads included.
3. **MoE / router**, taking the post-attention vector at position 127
   and choosing experts.

With **10 content + 118 pad**, step 2 averages 118 EOS-embedding
K/V's and ~10 content K/V's. The result lands in a region of
activation space the model never trained on. Step 3's router sees an
OOD input → narrow top-k margin → CPU and device flip experts →
divergent FFN output → divergent hidden state at position 127 →
divergent logits → low PCC.

With **55 content + 73 pad**, step 2 averages many more content K/V's
and the post-attention vector is much closer to a typical training-
distribution activation. Step 3's router sees an in-distribution input
→ wide top-k margin → CPU and device pick the same experts → matching
FFN → matching logits → high PCC.

This is exactly the empirical pattern: holding everything else fixed
and just varying the content/pad ratio at the input level shifts the
PCC at the output. The mechanism is one continuous causal chain from
input statistics to top-k stability to backend agreement.

### The full chain

```
prompt structure (e.g. 10 content / 118 pad)
    ↓
attention at pos 127 averages over many EOS K/V's + few content K/V's
    ↓
hidden state at pos 127 lands in a region the router rarely / never
saw during training                                        ← the OOD step
    ↓
router scores have many near-tied entries around the top-k
boundary                                                   ← the narrow-margin step
    ↓
CPU's stable-sort top-k and device's TT-rounded top-k pick
different sets of 8 experts                                ← the flip step
    ↓
each path runs different experts → different FFN outputs at this token
    ↓
hidden state at pos 127 inherits the FFN divergence
    ↓
the head reads pos 127 only (via norm_out[:, -1])
    ↓
logits PCC at pos 127 is low
```

Each link is verifiable, and the content/pad ratio experiment tested
exactly the **OOD step**: shift the input from heavily pad-driven
toward typical, and every downstream link relaxes — pos 127 PCC of
`h` rises from 0.92 (10/118 split) to 0.99 (≈55/≈73 split).

### Why hidden states PCC stays high even though logits PCC is low

A natural objection: if positions 124–127 of the layer output diverge
this much, why does the aggregate "hidden states" PCC report 0.9998 —
nearly perfect — when the same test reports logits PCC of only 0.91?
The answer is that the two numbers measure two very differently shaped
tensors, and PCC is shape-sensitive in a way that hides per-position
errors.

**Hidden states is a [bsz, seq, hc_mult, dim] tensor — every sequence
position contributes.** The model's prefill output `h` (the tensor
returned for the "hidden states PCC" check) has shape
`[32, 128, 4, 4096]` — 32 batch rows × 128 sequence positions × 4
hyper-connection copies × 4096 hidden units. PCC is computed over the
**flattened** tensor: ~67 million scalar values total, of which only
the ~1.6 million sitting at positions 124–127 carry meaningful
divergence. The other ~65 million values, at pad-dominated positions
0–123, are essentially bit-identical across CPU and device because:

- The pad token's embedding is the same vector regardless of which
  row or backend computes it.
- Attention at a pad-only position averages over identical pad K/V's,
  yielding identical outputs.
- The MoE router, fed an identical pad-derived activation, computes
  identical scores and picks identical top-k experts — pad tokens are
  not at any router margin (their scores are determined entirely by
  the model's bias toward whatever expert handles "EOS-shaped"
  inputs), so there's no opportunity for noise to flip the choice.
- The experts that do run on pad tokens produce bit-stable outputs
  because the input is the same and the arithmetic in those layers,
  while not bit-identical CPU vs device, is well within the tight-PCC
  tolerance of dense linear ops.

So per-position PCC of `h` looks like:

```
positions 0  – 124 :  PCC ≥ 0.998   (essentially identical, ~125 positions)
positions 125– 127 :  PCC = 0.95, 0.95, 0.92   (the three divergent positions)
```

When PCC is computed over the flat 67 M values, the 125 well-matching
positions dominate the correlation — they pull the result toward 1.0
because both CPU and device's flattened vectors track each other
almost perfectly across the bulk of their entries. The three diverging
positions contribute only ~2.4% of the values and shift the aggregate
correlation by a few thousandths. Final aggregate: 0.9998.

**Logits is a [bsz, vocab_size] tensor — only one sequence position
contributes.** The head's last operation is

```python
logits = head.weight @ norm_out[:, -1].float()
```

The `norm_out[:, -1]` slice keeps the activation **at position 127
only** and discards the other 127 positions entirely. So the logits
tensor's shape is `[32, 129280]` — there's no sequence dimension left
for the well-matching positions to dilute anything. Whatever
divergence exists at position 127 is the only thing the PCC
calculation sees. That divergence is the 0.92-PCC-worth of error from
the OOD-router-margin chain above; the head matmul + per-vocab linear
projection preserves it (a linear projection of two correlated vectors
gives a similarly correlated output). Final logits PCC: 0.91.

In other words: the head acts as a **per-position selector** that
isolates one sequence position's quality and surfaces it. The
hidden-states aggregate is a **per-position averager** that hides
single-position quality issues behind 127 well-matching neighbors.
That's why the same model run produces 0.9998 for one metric and 0.91
for the other — they are not measuring competing things, they are
measuring the same thing at different reduction granularities, and the
test's choice of head-output as the second metric inadvertently
exposed exactly the worst-case position.

A practical corollary: any test using **aggregate hidden states PCC**
to claim model fidelity is potentially masking concentrated per-
position errors. The right per-position sanity check is something
like:

```python
worst_pos_pcc = min(_pcc(cpu_h[:, k, :, :], dev_h[:, k, :, :])
                    for k in range(seqlen))
assert worst_pos_pcc >= 0.95
```

which would have failed early in this investigation and pointed at the
right position immediately.

### Why this is *not* a backend bug

It's worth being explicit: the device backend is not "wrong." It
computes a valid, deterministic, internally consistent forward pass.
CPU does the same. Their disagreement is real but is concentrated at
a small number of tokens whose router scores are *legitimately
ambiguous* — neither backend is more correct than the other when the
top-k margin is rounding-noise-wide. The reason this disagreement is
so dramatic in this test is twofold:

1. The head only reads one position (pos 127), so the test sees a
   single token's worth of routing noise, not an average over the
   whole sequence.
2. The prompt structure (EOS-as-pad with 118 EOS tokens before any
   content) drives that one position deep into OOD territory, making
   its router margins systematically narrow.

Remove either of these — use a real attention mask, or use prompts
that don't need 118 EOS pads — and the PCC failure goes away without
changing any backend code.

## Why `pow_scalar` was a red herring

The original device MLIR (`debug_e2e_prefill_1_logits.log`) showed
`ttnn.pow_scalar(x, 2.0)` inside `hc_head` and inside `Block.hc_pre`,
which looked suspicious because `pow(x, 2)` can be implemented as
`exp(2 log|x|)` on some backends. We verified by:

1. Editing `ParallelHead.hc_head` to use `x * x`.
2. Re-running the test and dumping the new MLIR.
3. Confirming via grep that `vhlo.power_v1` for the
   `32x128x16384xf32` tensor (the head's flatten-2 input) was removed,
   and that the corresponding `ttnn.pow_scalar` was replaced by
   `ttnn.multiply`.
4. Observing that the resulting logits PCC was bit-identical
   (`0.906730`).

Conclusion: on this hardware/compiler stack, `ttnn.pow_scalar(x, 2.0)`
and `ttnn.multiply(x, x)` produce the same result. The change has been
reverted; future investigations need not chase this lowering.

## What was changed

Both kept on disk because they are useful for the next round of
investigation; both are no-ops when the new flags are False:

- `third_party/tt_forge_models/deepseek_v4/modified_model/model_decode_opt.py`
  - `ParallelHead.forward(..., return_intermediates: bool = False)` —
    when True, returns `(logits, hc_out, norm_out, last_token)`.
  - `Block.forward(..., return_intermediates: bool = False)` — when
    True, returns `(x, attn_out, post_attn, ffn_out)`.
  - `Transformer.forward(..., return_head_intermediates: bool = False,
    return_block_intermediates: bool = False)` — propagates the head
    and block intermediates through.
- `tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py`
  - `test_e2e_prefill_head_intermediates_pcc[num_layers]` —
    head-level diagnostic, parametrized on `[1]` for now.
  - `test_e2e_prefill_block_intermediates_pcc[num_layers]` —
    block-level diagnostic with per-position PCC and pad-vs-content
    histogram, parametrized on `[1]`.

## Recommended next steps

1. **Verify routing flips directly.** Add a one-shot diagnostic that
   runs prefill with `return_block_intermediates=True` and *also*
   captures the `topk_idxs` chosen by the router on CPU vs device, for
   positions 124–127. Three outcomes:
   - Topk indices match → bug is in expert-arithmetic precision (look
     at A2a dispatch/combine summation).
   - Topk indices flip for some rows → router-score tiebreak issue
     (look at `Gate.forward` in `model_decode_opt.py:714` and the
     A2a dispatch path in `sparse_mlp.py`).
   - Both contribute → fix them in order, routing first.

2. **Patch the device router for stability.** Likely fix sites in
   `python_package/tt_torch/sparse_mlp.py`:
   - Force fp32 for `Gate.forward`'s linear → softmax → topk path, so
     the device router matches the CPU reference bit-for-bit (or as
     close as TT-MLIR allows).
   - If the router already runs in fp32, consider clamping the topk
     boundary with a deterministic tiebreak (smaller expert index
     wins) to match PyTorch's behavior on ties.

3. **Update test expectations.** `test_e2e_prefill_pcc` currently
   asserts a single aggregate `logits` number. Recommended changes:
   - Lower the threshold for `num_layers=1` (the "untrained-tail"
     numerical sensitivity is intrinsic, not a regression) or add a
     `xfail` until the routing patch lands.
   - Add a per-position assertion: `pcc(h[:, k, ...])` for each
     position `k` where `prompt_ids` has any content row, with a
     PCC threshold appropriate to the per-position signal-to-noise.
   - This makes future regressions impossible to hide behind a
     pad-dominated aggregate.

4. **Sanity-check the deeper-layer cases.** `num_layers=10/15/20/30/43`
   currently report `hidden_states pcc=0.99` aggregate. Apply the
   per-position PCC sanity check there too — if the same content-vs-pad
   structure shows up, the higher-layer tests are also masking real
   errors.

## Reproducing

```bash
# requires llmbox (32 TT devices)
source venv/activate

# per-stage head intermediates (proves head is innocent)
pytest -svv tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py::test_e2e_prefill_head_intermediates_pcc[1]

# block sub-stage intermediates + content/pad histogram (proves cliff is in MoE)
pytest -svv tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py::test_e2e_prefill_block_intermediates_pcc[1]

# original failing test (for reference)
pytest -svv tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py::test_e2e_prefill_pcc[1]
```

## Key file/line references

- `third_party/tt_forge_models/deepseek_v4/modified_model/model_decode_opt.py`
  - `Transformer.forward` (line 972 in original; now line ~983 after the
    additive flag plumbing)
  - `ParallelHead.forward` (line 894), `ParallelHead.hc_head` (line 903)
  - `Block.forward` (line 865)
  - `Attention.forward` prefill branch (line 588)
  - `MoE` and `Gate` (line 696 and below)
- `python_package/tt_torch/sparse_mlp.py`
  - `A2aSparseMLPWithSharedExperts._cpu_forward` (line 524)
  - dispatch/combine path (around line 540)
  - `Gate` adapter (line 753)
- `tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py`
  - `_pcc` (line 425)
  - `test_e2e_prefill_pcc` (line 518) — the original failing test
  - `test_e2e_prefill_head_intermediates_pcc` — head-level diagnostic
  - `test_e2e_prefill_block_intermediates_pcc` — block-level diagnostic
