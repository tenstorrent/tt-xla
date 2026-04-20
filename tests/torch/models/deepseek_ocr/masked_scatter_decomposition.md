# Masked Scatter Decomposition in DeepSeek OCR

## Table of Contents

1. [What does DeepSeek OCR need masked_scatter for?](#1-what-does-deepseek-ocr-need-masked_scatter-for)
2. [What does masked_scatter_ do?](#2-what-does-masked_scatter_-do)
3. [Why can't we use masked_scatter_ directly on TT?](#3-why-cant-we-use-masked_scatter_-directly-on-tt)
4. [Old Decomposition — Flatten everything to 1D](#4-old-decomposition--flatten-everything-to-1d)
5. [New Decomposition — Row-level cumsum + 2D gather](#5-new-decomposition--row-level-cumsum--2d-gather)
6. [Memory comparison](#6-memory-comparison)
7. [TT device status](#7-tt-device-status)

---

## 1. What does DeepSeek OCR need masked_scatter for?

DeepSeek OCR is a vision-language model. It takes an image and a text prompt,
encodes them, and produces text output. Internally, the model:

1. **Tokenises the text** into token IDs and converts them to embeddings
   via `embed_tokens`. This gives a tensor of shape `[1, S, D]` where
   `S` is the sequence length (e.g. 913) and `D` is the hidden dimension
   (e.g. 1280).

2. **Encodes the image** through SAM + CLIP vision encoders, then projects
   the image features into the same `D`-dimensional space. This produces
   `source` — a tensor of shape `[N, D]` where `N` is the number of
   image token positions (e.g. 903).

3. **Scatters the image features into the text embeddings.** Some positions
   in the sequence are placeholders for image tokens. A boolean mask
   `images_seq_mask` of shape `[S]` marks which positions should be
   replaced with image features. The model needs to:
   - Keep text embeddings at positions where `mask = False`
   - Replace embeddings at positions where `mask = True` with the image
     features, **in order** (first True position gets `source[0]`, second
     True position gets `source[1]`, etc.)

This is exactly what `masked_scatter_` does.

---

## 2. What does masked_scatter_ do?

`masked_scatter_` is a PyTorch in-place operation:

```python
inputs_embeds[idx].masked_scatter_(mask.unsqueeze(-1), source)
```

It scans the mask left-to-right. For each `True` position, it pulls the
**next** value from `source` and writes it into the target tensor. `False`
positions are left untouched.

### Small example

```
inputs_embeds (target):  shape [5, 3]    (S=5 rows, D=3 cols)
    [[1.0, 1.0, 1.0],    # row 0  (text token)
     [2.0, 2.0, 2.0],    # row 1  (text token)
     [3.0, 3.0, 3.0],    # row 2  (image placeholder)
     [4.0, 4.0, 4.0],    # row 3  (text token)
     [5.0, 5.0, 5.0]]    # row 4  (image placeholder)

mask_1d:  [False, False, True, False, True]
          (2 image positions → N=2)

source:  shape [2, 3]
    [[9.1, 9.2, 9.3],    # image feature 0
     [8.1, 8.2, 8.3]]    # image feature 1
```

After `masked_scatter_`:

```
result:
    [[1.0, 1.0, 1.0],    # row 0  — False → kept
     [2.0, 2.0, 2.0],    # row 1  — False → kept
     [9.1, 9.2, 9.3],    # row 2  — True  → source[0]
     [4.0, 4.0, 4.0],    # row 3  — False → kept
     [8.1, 8.2, 8.3]]    # row 4  — True  → source[1]
```

The key behaviour: True positions consume source values **sequentially**.
The 1st True gets `source[0]`, the 2nd True gets `source[1]`, and so on.

---

## 3. Why can't we use masked_scatter_ directly on TT?

When PyTorch compiles a model for the TT device via XLA/StableHLO, certain
ops introduce **dynamic shapes** — tensor dimensions that depend on data
values, not just tensor shapes. `masked_scatter_` is one of them because
the number of True values in the mask determines how many source values to
consume, and XLA cannot know this at compile time.

This causes a crash on TT:
```
TT_FATAL in ttnn::repeat_interleave -> transpose_impl
Index is out of bounds for the rank
```

**Solution**: Replace `masked_scatter_` with a decomposition that uses only
**static-shape** ops — ops whose output shapes are fully determined by
their input shapes, with no data-dependent dimensions.

Related issues:
- [tt-xla#3316](https://github.com/tenstorrent/tt-xla/issues/3316): masked_scatter_ fails with dynamic shapes
- [tt-xla#3412](https://github.com/tenstorrent/tt-xla/issues/3412): old decomp OOM due to large cumsum

---

## 4. Old Decomposition — Flatten everything to 1D

The old decomposition replaces `masked_scatter_` with static-shape ops by
working entirely in **flattened 1D** space.

### The idea

Instead of scanning left-to-right and pulling from source (which is
dynamic), we:
1. Flatten the 2D target `[S, D]` and the broadcast mask `[S, D]` into
   1D vectors of length `S*D`.
2. Use `cumsum` on the flattened mask to compute, for every position, which
   source element it should get.
3. Gather the source elements using those indices.
4. Use `where` to pick source values at True positions and keep originals
   at False positions.

All of these ops have **static output shapes** — they always produce
tensors of the same shape as their inputs, regardless of how many True
values exist.

### Step-by-step with example

Using the same example: `S=5, D=3, N=2`.

#### Step 1: Broadcast mask from [S] to [S, D]

```python
mask = mask_1d.unsqueeze(-1)                    # [5, 1]
mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)  # both [5, 3]
```

```
mask_broad:
    [[F, F, F],
     [F, F, F],
     [T, T, T],
     [F, F, F],
     [T, T, T]]
```

Each row of the mask is repeated D times across columns.

#### Step 2: Flatten everything to 1D

```python
mask_flat   = mask_broad.reshape(-1)             # [15]
data_flat   = data.reshape(-1)                   # [15]
source_flat = source.reshape(-1)                 # [6]
```

```
mask_flat:   [F,F,F, F,F,F, T,T,T, F,F,F, T,T,T]
              row0    row1    row2    row3    row4
data_flat:   [1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5]
source_flat: [9.1, 9.2, 9.3, 8.1, 8.2, 8.3]
```

#### Step 3: Cast mask to int64

```python
mask_i = mask_flat.long()
# [0,0,0, 0,0,0, 1,1,1, 0,0,0, 1,1,1]
```

We need integers for cumsum (cumsum doesn't work on bool).

#### Step 4: Cumulative sum

```python
source_idx = torch.cumsum(mask_i, 0) - 1
```

`cumsum` counts how many True values we've seen so far at each position.
Subtracting 1 converts this to a 0-based index into source_flat.

```
mask_i:      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
cumsum:      [0, 0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6]
cumsum - 1:  [-1,-1,-1,-1,-1,-1, 0, 1, 2, 2, 2, 2, 3, 4, 5]
```

- Position 6 (first True) → index 0 → `source_flat[0]` = 9.1
- Position 7 (second True) → index 1 → `source_flat[1]` = 9.2
- Position 8 (third True) → index 2 → `source_flat[2]` = 9.3
- Position 12 (fourth True) → index 3 → `source_flat[3]` = 8.1
- etc.

#### Step 5: Clamp negative indices

```python
source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
```

Positions before the first True have index -1 (no source value to
consume). Clamping to 0 makes them point to `source_flat[0]`, but this is
harmless because `where` will discard these values anyway (the mask is
False there).

```
clamped:     [0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 4, 5]
```

#### Step 6: Gather source values

```python
gathered = source_flat[source_idx]
```

Simple indexing — for each of the 15 positions, pick the source value at
the computed index.

```
gathered:    [9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.1, 9.2, 9.3, 9.3, 9.3, 9.3, 8.1, 8.2, 8.3]
```

(False positions get garbage values from source — doesn't matter.)

#### Step 7: Conditional select with where

```python
result_flat = torch.where(mask_flat, gathered, data_flat)
```

For each position:
- If mask is True → take the gathered source value
- If mask is False → keep the original data value

```
result_flat: [1, 1, 1, 2, 2, 2, 9.1, 9.2, 9.3, 4, 4, 4, 8.1, 8.2, 8.3]
```

#### Step 8: Reshape back to [S, D]

```python
result = result_flat.view_as(inputs_embeds)
```

```
result:
    [[1.0, 1.0, 1.0],
     [2.0, 2.0, 2.0],
     [9.1, 9.2, 9.3],
     [4.0, 4.0, 4.0],
     [8.1, 8.2, 8.3]]
```

Identical to the original `masked_scatter_` output.

### The memory problem

The cumsum operates on a **flattened** tensor of size `S * D`. For
DeepSeek OCR:

```
S = 913, D = 1280
Flattened size = 913 × 1280 = 1,168,640 elements
Memory for int64 cumsum: 1,168,640 × 8 bytes = ~8.9 MB
Total intermediates (mask_flat, data_flat, source_idx, gathered, result_flat):
    ~27.9 MB
```

This is wasteful because the mask is **row-uniform** — every column within
a row has the same mask value (True or False). We're doing `D=1280` times
more work than necessary.

---

## 5. New Decomposition — Row-level cumsum + 2D gather

The key insight: since `mask_1d` is a **per-row** mask (all columns in a
row share the same True/False value), we can compute the source index
**once per row** instead of once per element.

### Step-by-step with example

Same example: `S=5, D=3, N=2`.

```
inputs_embeds:  [5, 3]
mask_1d:        [False, False, True, False, True]
source:         [2, 3]
```

#### Step 1: Cast mask to int64

```python
mask_i = mask_1d.long()
# [0, 0, 1, 0, 1]
```

Same as before, but this is just `[S]` = 5 elements, not `[S*D]` = 15.

#### Step 2: Cumulative sum on [S]

```python
source_idx = torch.cumsum(mask_i, 0) - 1
```

```
mask_i:      [0, 0, 1, 0, 1]
cumsum:      [0, 0, 1, 1, 2]
cumsum - 1:  [-1, -1, 0, 0, 1]
```

This tells us: row 2 should get `source[0]`, row 4 should get `source[1]`.

The cumsum input is only **[S] = 5 elements** vs [S*D] = 15 in the old
decomp. For DeepSeek OCR, that's **[913]** vs **[1,168,640]** — a
**1280x reduction**.

#### Step 3: Clamp negative indices

```python
source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
```

```
clamped:     [0, 0, 0, 0, 1]
```

Same purpose as before — makes negative indices safe for gather.

#### Step 4: Broadcast index to 2D

```python
source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds)
```

```
source_idx_2d:  [5, 3]
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [1, 1, 1]]
```

`unsqueeze(-1)` turns `[5]` into `[5, 1]`, then `expand_as` broadcasts
it to `[5, 3]`. Every column in a row has the same index — because we
want to gather the **entire row** from source.

Note: `expand_as` does NOT allocate new memory — it creates a **view**
with stride 0 along the expanded dimension. This is a zero-cost op.

#### Step 5: 2D Gather

```python
gathered_rows = torch.gather(source, 0, source_idx_2d)
```

`torch.gather(input, dim, index)` does:
```
output[i][j] = input[index[i][j]][j]    (when dim=0)
```

For each position `(i, j)` in the output, it looks up
`source[source_idx_2d[i][j]][j]`.

```
gathered_rows:  [5, 3]
    [[9.1, 9.2, 9.3],    # index=0 → source[0] = [9.1, 9.2, 9.3]
     [9.1, 9.2, 9.3],    # index=0 → source[0]
     [9.1, 9.2, 9.3],    # index=0 → source[0]
     [9.1, 9.2, 9.3],    # index=0 → source[0]
     [8.1, 8.2, 8.3]]    # index=1 → source[1] = [8.1, 8.2, 8.3]
```

Rows at False positions get garbage values (source[0] here), but that's
fine — `where` will discard them.

#### Step 6: Conditional select with where

```python
result = torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds)
```

`mask_1d.unsqueeze(-1)` broadcasts `[5]` → `[5, 1]` → `[5, 3]`:

```
mask_2d:
    [[F, F, F],
     [F, F, F],
     [T, T, T],
     [F, F, F],
     [T, T, T]]
```

For each element:
- True → take from `gathered_rows`
- False → keep from `inputs_embeds`

```
result:
    [[1.0, 1.0, 1.0],    # False → original
     [2.0, 2.0, 2.0],    # False → original
     [9.1, 9.2, 9.3],    # True  → source[0]
     [4.0, 4.0, 4.0],    # False → original
     [8.1, 8.2, 8.3]]    # True  → source[1]
```

Identical to both `masked_scatter_` and the old decomposition.

---

## 6. Memory comparison

| | Old Decomposition | New Decomposition | Reduction |
|---|---|---|---|
| **cumsum input size** | S × D = 1,168,640 | S = 913 | **1280x smaller** |
| **cumsum memory (int64)** | 8.9 MB | 0.007 MB | **1280x** |
| **Total intermediates** | ~27.9 MB | ~4.5 MB | **~84%** |
| **Flatten/reshape ops** | 5 (mask, data, source, gathered, result) | 0 | eliminated |
| **Gather dimension** | 1D indexing on [S\*D] | 2D gather on [S, D] | same work, better locality |

### Why is the new decomp so much smaller?

The old decomp flattens `[S, D]` → `[S*D]` and runs cumsum on every
element. But the mask is the same across all D columns in a row! So
the old decomp computes the same index 1280 times per row.

The new decomp computes the index **once per row** (`cumsum` on `[S]`),
then broadcasts it to all columns via `expand_as` (which is free — just
a stride trick, no memory allocation).

---

## 7. TT device status

### CPU correctness

Both decompositions are **bit-exact correct** on CPU. All 24 CPU sanity
tests pass, including direct comparison between old decomp, new decomp,
and `masked_scatter_` reference.

### TT device issues

| Test | PCC (CPU vs TT) | Status |
|---|---|---|
| `masked_scatter_` reference | Crashes | `TT_FATAL` in `repeat_interleave` |
| Old decomposition | Crashes | `TT_FATAL` in `repeat_interleave` |
| New decomposition | **0.42** | PCC drop |
| New decomp (compiler disabled) | **0.42** | Same PCC drop |

### Root cause: `torch.gather` on TT

Per-op isolation tests identified `torch.gather` as the sole culprit:

| Op | PCC | Status |
|---|---|---|
| `mask_1d.long()` | ~1.0 | PASS |
| `torch.cumsum(mask_i, 0)` | ~1.0 | PASS |
| `cumsum - 1` | ~1.0 | PASS |
| `torch.clamp(idx, 0, max)` | ~1.0 | PASS |
| `unsqueeze + expand` | ~1.0 | PASS |
| **`torch.gather(source, 0, idx_2d)`** | **0.08** | **FAIL** |
| `torch.where(mask, gathered, orig)` | ~1.0 | PASS |
| Op1-Op5 pipeline (no gather) | ~1.0 | PASS |
| Op1-Op6 pipeline (with gather) | **0.08** | FAIL |
| Full decomposition | **0.42** | FAIL |

The full decomposition PCC (0.42) is higher than gather alone (0.08)
because `where` preserves correct values at False positions (~1.1% of
elements, i.e. 10 out of 913 rows), partially masking the gather damage.

### Current workaround

The decomposition is decorated with `@torch.compiler.disable` so it runs
eagerly on CPU and is excluded from XLA tracing. The rest of the model
is still compiled for TT device.

```python
@torch.compiler.disable
def _masked_scatter_decomp(inputs_embeds_row, mask_1d, source):
    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)
    gathered_rows = torch.gather(source, 0, source_idx_2d)
    return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)
```

### Related issues

- [tt-xla#3316](https://github.com/tenstorrent/tt-xla/issues/3316): `masked_scatter_` fails with dynamic shapes on TT
- [tt-xla#3412](https://github.com/tenstorrent/tt-xla/issues/3412): Old decomp runs cumsum on [S*D] int64 → OOM on TT
