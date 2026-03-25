---
name: gpt-oss-layer-parsing
description: Explain GPT-OSS transformer architecture and identify a full GPT-OSS layer or an attention-only sublayer from Tracy ops CSVs or TTNN perf traces. Use when the user asks about GPT-OSS layer boundaries, even vs odd layers, full-context vs sliding-window attention, MoE structure, or extracting one attention block for profiling.
---

# GPT-OSS Layer Parsing

Use this skill when analyzing `test_gpt_oss_20b_tp` traces and you need to map raw op sequences back to GPT-OSS model structure.

## Architecture Summary

One GPT-OSS transformer layer has this logical structure:

1. input norm
2. attention sub-layer
3. post-attention norm
4. MoE sub-layer
5. output hidden state

Two layer types alternate across the stack:
- even layers: full-context attention
- odd layers: sliding-window attention

The block structure is otherwise the same. The practical parsing rule is:
- do not split layers by parity-specific op names alone
- instead, detect the repeated transformer block shape
- treat parity as an annotation on the attention mask behavior inside that block

## Attention Sub-layer Shape

Inside one attention sub-layer, expect this semantic sequence:

1. linear projection to Q
2. linear projection to K
3. linear projection to V
4. positional encoding on Q and K
5. scaled dot-product attention
6. attention mask application
7. learned bias added into the softmax denominator path
8. softmax
9. weighted sum over V
10. output projection
11. residual add

Important model fact:
- K and V use fewer heads than Q

In traces this may show up as:
- smaller K/V matmuls than Q matmuls when shapes are visible
- fused attention kernels where mask, bias, softmax, and value accumulation do not appear as separate rows

## MoE Sub-layer Shape

After attention, expect:

1. post-attention norm
2. router projection to expert logits
3. top-k routing logic
4. expert MLP work for active experts only
5. weighted combine of expert outputs
6. residual add

The expert MLP pattern is:
- gate projection
- up projection
- gated activation
- down projection

In low-level traces, router, dispatch, combine, and expert execution may be separated by collectives or packing/unpacking ops.

## What To Extract

Choose one of these targets before parsing:

### Full layer

Extract from the input norm that starts a transformer block through the residual add that ends the MoE sub-layer.

### Attention-only sublayer

Extract from the input norm immediately preceding Q/K/V projections through the residual add after the attention output projection.

Do not include:
- the post-attention norm
- router ops
- expert MLP ops
- MoE combine or final residual after MoE

## Parsing Workflow

1. First isolate the `decode_2` window from the parent profiling workflow.
2. Work within one device-parallel repeated region at a time. Ignore prefill and `decode_1`.
3. Scan for repeated transformer blocks rather than absolute row numbers.
4. Mark candidate attention starts at an input-norm-like region followed closely by 3 projection matmuls.
5. Confirm the attention body by finding either:
   - an explicit attention sequence, or
   - a fused SDPA-style kernel between QKV projections and the output projection
6. End the attention sub-layer at the residual add after the attention output projection.
7. Confirm the full transformer layer by checking that a post-attention norm and MoE-style region follow.
8. End the full layer at the residual add after expert combine.

## Boundary Heuristics

Prefer structural cues over exact op names.

Strong start-of-attention cues:
- norm or reduce-like op
- 3 nearby projection matmuls
- optional collective ops between projections in tensor-parallel traces

Strong mid-attention cues:
- `sdpa` or another fused attention kernel
- a cluster of ops that semantically covers mask, bias, softmax, and V aggregation
- one output projection matmul after the attention core

Strong end-of-attention cues:
- residual add immediately after the output projection
- the next significant region looks like a norm or router projection

Strong MoE cues:
- one router-like projection
- top-k or dispatch/scatter behavior
- repeated expert MLP work
- combine/reduce behavior before a final residual add

## Even/Odd Layer Detection

Once layer boundaries are known, assign parity by block index inside the sliced run:

- first GPT-OSS layer in the run: even
- second GPT-OSS layer in the run: odd
- then continue alternating

Use parity for reporting:
- even layer -> full-context attention
- odd layer -> sliding-window attention

If explicit attention-mask evidence in the trace disagrees with positional parity, report the discrepancy instead of forcing the label.

## Middle-Layer Selection For 6-Layer Runs

When the profiling run uses 6 GPT-OSS layers, produce two analyses:
- middle even layer
- middle odd layer

Select by parity group within the repeated block sequence:
- even-parity blocks: choose the middle even block
- odd-parity blocks: choose the middle odd block

Do not choose the middle block of all 6 layers and then infer parity afterward.

## Practical Notes

- Fused kernels are normal. Map them to semantic stages instead of requiring one CSV row per architecture bullet.
- Collectives may appear inside attention or MoE because of tensor parallelism or expert routing. Treat them as part of the enclosing sub-layer unless they clearly separate repeated transformer blocks.
- If shapes are available, Q is often the largest attention projection, while K and V are smaller because they use fewer heads.
- If op names are noisy, boundary detection from repeated block structure is more reliable than symbol matching.

## Output Expectations

When using this skill, report:

- whether the extraction target was a full layer or attention-only sublayer
- the selected parity, if known
- the row range or boundary ops used to define the slice
- the evidence used to justify the boundaries
- any ambiguity from fusion, collectives, or missing shape metadata

## References

- `.cursor/skills/layer-profiling/SKILL.md`
