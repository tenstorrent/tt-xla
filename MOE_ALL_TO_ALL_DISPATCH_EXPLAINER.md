# MoE expert-parallel dispatch on a 2D mesh: `cluster_axis`, dispatch extent, and the expert_mapping

An explainer for the `all_to_all_dispatch_metadata` op and the two device-count
quantities that are easy to conflate. Written as background for
`MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md`.

## MoE in one paragraph

A normal transformer layer sends every token through one big feed-forward network (FFN).
A **Mixture-of-Experts (MoE)** layer replaces that one FFN with many smaller ones called
**experts** (GPT-OSS-120B has 128 per layer). A tiny **router** looks at each token and
picks its top-k experts (say k=4 of 128). Only those experts run for that token. You get
the quality of a huge model but only pay compute for k experts per token — the model is
"sparsely activated."

The catch: you now store *all* 128 experts' weights. That's far too much for one chip, so
you spread the experts across devices. That's **expert parallelism (EP)**.

## Why we need an "all-to-all dispatch"

Say experts are split across 8 devices, 16 experts each. A token sitting on device 3 might
have picked an expert that lives on device 5. So before you can run the experts, you have
to **ship each token to the device(s) that hold its chosen experts**, run the expert math
there, then ship the results back. That shipping is:

- **dispatch** — scatter tokens to the devices holding their experts (this is
  `all_to_all_dispatch_metadata`),
- expert compute — each device runs its local experts on the tokens it received,
- **combine** — gather results back to each token's original device.

`all_to_all_dispatch_metadata` is the *dispatch* stage. It also emits routing "metadata"
(bookkeeping of which token went where) so the later combine can undo the shuffle — hence
the name.

## The device mesh and `cluster_axis`

The 32 devices are arranged as a 2D grid — a **mesh** — of 4 rows × 8 columns. We use the
two axes for two different kinds of parallelism:

- **columns (8):** expert parallelism — the 128 experts are split across the 8 columns
  (16 per column).
- **rows (4):** data parallelism — each row is a *full independent replica* of all 128
  experts, and the four rows just process different quarters of the batch.

`cluster_axis` names the axis used for EP/dispatch. Here `cluster_axis = 1` (columns). The
consequence that matters: **a token is only ever dispatched among the 8 devices in its own
row.** Row 0's tokens never travel to row 1 — row 1 is a separate copy doing its own batch
slice.

## The two quantities

A deliberately tiny mesh makes the numbers easy: **2 rows × 4 columns = 8 devices**,
**8 experts**, `cluster_axis = 1` (columns). So 8 experts ÷ 4 columns = **2 experts per
device**:

```
            col0     col1     col2     col3
row0    [E0,E1]  [E2,E3]  [E4,E5]  [E6,E7]
row1    [E0,E1]  [E2,E3]  [E4,E5]  [E6,E7]      <- row1 is an identical replica of row0
```

Global device index = `row*num_cols + col`, so the 8 devices are numbered 0..7:

```
            col0  col1  col2  col3
row0          0     1     2     3
row1          4     5     6     7
```

### Quantity 1 — dispatch extent (= 4 here; = 8 on the real 4x8 mesh)

**"How many devices does a token get scattered across during dispatch?"**

A token lives in some row and needs to reach whichever *column* holds its expert. It only
moves **within its row**, and there are 4 columns, so a token is dispatched among **4
devices**. That's `dispatch_devices = num_cols = 4` (for `cluster_axis=1`).

This is a **routing** number. The op uses it to:

- size the dispatched output buffer (each token is replicated toward its 4 possible
  column-destinations),
- decide the network neighbors — who each device can send to (the 4 devices along its
  row).

It is *not* 8 (the whole mesh), because dispatch never crosses rows.

### Quantity 2 — mapping row count (= 8 here; = 32 on the real 4x8 mesh)

The **`expert_mapping`** is a lookup table with shape `[devices, experts]`. Think of it as:
*"for each device, which experts does it hold?"* Row `d` is device `d`'s view.

In our toy, expert `e` lives in column `e // 2`, and every device agrees on that, so
**every row is identical**:

```
expert_mapping[d] = [ owner(E0), owner(E1), ..., owner(E7) ]
                  = [   0,   0,    1,   1,    2,   2,    3,   3 ]   for every d
                     (owner = the column that holds that expert, 0..3)
```

Here's the crucial part: **the op runs one copy of its kernel on every physical device,
and each device reads *its own row* of this table to learn the layout. To find "its own
row," the kernel uses its *global* mesh index** (0..7):

```cpp
noc_async_read_page(linearized_mesh_coord, ...)   // linearized_mesh_coord = row*num_cols + col
```

Device 6 (row 1, col 2) reads **row 6**. Device 7 reads row 7. So the table must have **one
row per physical device = 8 rows**, even though there are only 4 *distinct* rows of content
(rows 0 and 4 are identical, 1 and 5 identical, etc.). The duplicate rows exist purely so
that global index 6 or 7 can index in-bounds.

This is an **indexing** number: one row per device, addressed by global device id.

### Why the two are easy to confuse

On a **1D** mesh — 1 row × 8 columns (8 devices) — the dispatch extent (8 columns) and the
mapping row count (8 devices) are **the same number**. The original tt-mlir code was
written for that "1×N dispatch ring" and used one value for both. It even said so in a
comment.

The moment you go **2D** (add the 4 replica rows), they diverge:

- dispatch extent stays **8** (columns — tokens still only move within a row),
- mapping rows jump to **32** (every physical device must be able to look itself up by
  global index).

### The bug, in these terms

tt-mlir built the `expert_mapping` with **`dispatch_devices` rows** (8 on the real mesh)
when it needed **`total_mesh_devices` rows** (32). So device global-index 8 (row 1, col 0)
tried to read row 8 of an 8-row table. tt-metal's validate caught it first:

> `Expert mapping tensor first dimension must equal number of devices (32), got 8`

(In the toy it would read "expected 8, got 4.")

The **values** were already right (every row is `owner(e) = e // experts_per_device`, and
all rows identical), so the fix is just to make the table tall enough — one row per mesh
device — which is what the fix does: keep the expert count driven by the cluster axis
(`experts_per_device × cluster_devices`), but set the row count to the total mesh device
count. See `MOE_DECODE_EXPERT_MAPPING_2D_MESH_BUGREPORT.md`.
