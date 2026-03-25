---
name: llama-layer-parsing
description: Explain Llama transformer architecture and identify a full Llama layer or an attention-only sublayer from Tracy ops CSVs or TTNN perf traces. Use when the user asks about Llama layer boundaries, RMSNorm and SwiGLU structure, grouped-query attention, or extracting one attention block for profiling.
---

# Llama Layer Parsing

Use this skill when analyzing `test_llama_3_1_70b_tp_galaxy` traces and you need to map raw op sequences back to Llama model structure.

## Architecture Summary

One Llama transformer layer has this logical structure:

1. input RMSNorm
2. attention sub-layer
3. residual add
4. post-attention RMSNorm
5. MLP sub-layer
6. residual add
7. output hidden state

All layers use the same block structure. There is no GPT-OSS-style even/odd alternation.

## Attention Sub-layer Shape

Inside one attention sub-layer, expect this semantic sequence:

1. linear projection to Q
2. linear projection to K
3. linear projection to V
4. positional encoding on Q and K
5. scaled dot-product attention
6. softmax over scores
7. weighted sum over V
8. output projection
9. residual add

Important model fact:
- Llama uses grouped-query attention, so K and V use fewer heads than Q

In traces this may show up as:
- smaller K/V matmuls than Q matmuls when shapes are visible
- a fused attention kernel such as `sdpa`
- collectives interleaved with projections or output projection in tensor-parallel traces

## MLP Sub-layer Shape

After the attention residual, expect:

1. post-attention RMSNorm
2. gate projection
3. up projection
4. elementwise activation on the gate branch
5. gated multiply
6. down projection
7. residual add

This is the standard gated MLP pattern often referred to as SwiGLU.

## What To Extract

Choose one of these targets before parsing:

### Full layer

Extract from the input RMSNorm that starts a transformer block through the residual add that ends the MLP sub-layer.

### Attention-only sublayer

Extract from the input RMSNorm immediately preceding Q/K/V projections through the residual add after the attention output projection.

Do not include:
- the post-attention RMSNorm
- gate/up/down MLP projections
- activation or gated multiply ops
- the final residual after the MLP

## Parsing Workflow

1. First isolate the `decode_2` window from the parent profiling workflow.
2. Work within one device-parallel repeated region at a time. Ignore prefill and `decode_1`.
3. Scan for repeated transformer blocks rather than absolute row numbers.
4. Mark candidate layer starts at an RMS-like region followed closely by 3 projection matmuls.
5. Confirm the attention body by finding either:
   - an explicit Q/K/V -> attention -> output projection sequence, or
   - a fused SDPA-style kernel between QKV projections and the output projection
6. End the attention sub-layer at the residual add after the attention output projection.
7. Confirm the full layer by checking that another RMS-like region and a gated MLP sequence follow.
8. End the full layer at the residual add after the down projection.

## Boundary Heuristics

Prefer structural cues over exact op names.

Strong start-of-layer cues:
- RMSNorm-like op or reduce-like op
- 3 nearby projection matmuls
- optional collectives near projections in tensor-parallel traces

Strong mid-attention cues:
- `sdpa` or another fused attention kernel
- one output projection matmul after the attention core
- a residual add immediately after the output projection

Strong MLP cues:
- a second RMS-like region after the attention residual
- 2 projection matmuls that feed a gated activation
- one down-projection matmul
- a final residual add

Strong end-of-layer cue:
- the next repeated block begins with another RMS-like region and Q/K/V projections

## Middle-Layer Selection

For the reduced 3-layer Llama profiling run, choose the middle repeated transformer block inside the selected `decode_2` slice.

Do not rely on a raw numeric layer ID alone. Prefer the structurally repeated block bounded by:
- a starting RMS-like or reduce-like region
- attention with 3 projections, attention core, and output projection
- a second RMS-like region
- gated MLP work
- a final residual add

## Practical Notes

- Fused kernels are normal. Map them to semantic stages instead of requiring one CSV row per architecture bullet.
- Collectives may appear inside the attention or MLP region because of tensor parallelism. Treat them as part of the enclosing sub-layer unless they clearly separate repeated transformer blocks.
- If shapes are available, Q is often the largest attention projection, while K and V are smaller because they use fewer heads.
- If op names are noisy, repeated block structure is more reliable than symbol matching.

## Output Expectations

When using this skill, report:

- whether the extraction target was a full layer or attention-only sublayer
- the row range or boundary ops used to define the slice
- the evidence used to justify the boundaries
- whether the selected block is the middle repeated Llama layer in `decode_2`
- any ambiguity from fusion, collectives, or missing shape metadata

## References

- `.cursor/skills/layer-profiling/SKILL.md`
