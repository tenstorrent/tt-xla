**Update: root-caused + candidate fix**

Dug in with a per-op activation dump (instrumented the tt-mlir runtime op loop to compare each op's per-user output for 32 identical prompts). New learnings:

- **The decoder backbone is bit-row-invariant** — all `[1024,…]` matmuls, SDPA, and rms_norm produce identical per-user output. So this is **not** a model-compute precision issue (correcting the original "matmul dest-accumulation in the model" framing).
- **First divergence is at the per-user last-token gather** feeding the LM head. The SHLO is a multi-dim `stablehlo.gather`; tt-mlir lowers it to `flat_index = indices @ strides` via a **float `ttnn.matmul` with bf16 destination accumulation** + embedding lookup.
- bf16 represents integers exactly only to 256 (step 2 to 512, **step 4 above 512**). So any user whose last-token **flat index ≥ 512** rounds to the wrong value → gathers the **wrong row** → wrong logits → wrong token. This exactly explains the boundary (first user hitting index 512: **user 16** at stride 32, **user 4** at stride 128), the determinism, and why global `fp32_dest_acc_en=True` masked it.

**Candidate fix (tt-mlir):** force `fp32_dest_acc_en` on matmuls whose result feeds an embedding's index operand (index arithmetic must be exact; these matmuls are tiny). Branch `kmabee/issue-5116-gather-index-fp32-dest-acc`, in `TTNNSetComputeKernelConfig`.

**Testing (Llama-3.2-3B, 32 identical prompts, greedy, global fp32 OFF):** all now consistent + correct —

| case | before | after |
|---|---|---|
| batch 32, 1-layer / full 28-layer | split | unique=1 |
| minimal config (boundary 4) | split | unique=1 |
| decode (16 / 32 tokens) | divergent | unique=1 |
| batch 1 | — | no regression |

Output matches the fp32 reference (correct, not just consistent). This is squarely the `#8666` family but specifically the **gather/embedding index** path.
