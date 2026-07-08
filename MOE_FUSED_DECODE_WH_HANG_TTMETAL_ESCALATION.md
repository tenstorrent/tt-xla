# Escalation: GPT-OSS fused-MoE decode hangs at a `cluster_axis=0` all_gather (Wormhole galaxy, ≥2 layers)

**To:** tt-metal fabric / CCL team
**From:** tt-xla (branch `hshah/moe-compute-path`), 2026-07-08
**Board:** Wormhole 32-chip galaxy (TG / 6U), `(4,8)` mesh, `cluster_axis=0` = size-4 dispatch ring, `FABRIC_1D_RING`
**Severity:** blocks the fused-MoE decode path beyond 1 layer
**What we need:** on-device instrumentation (Watcher / tt-lens) of the stalled collective — we cannot get it here (Watcher is broken in our container, §6).

---

## 1. Summary

The GPT-OSS-20B fused-MoE decode graph **hangs at a plain `ttnn.all_gather` on `cluster_axis=0`** (the LM-head logit
gather, `8×1×201088 → 32×1×201088`), which is the first fabric collective *after* both MoE layers' collectives have
completed. It is **depth-dependent**: 1 layer passes (PCC 0.999), 2 layers hang.

**We have spent significant effort isolating this and can now tell you what it is NOT** (§4), which should sharply focus
your investigation:

- It is **NOT the fused-MoE collectives.** A pure-`ttnn` reproduction of the *same* `all_to_all_dispatch_metadata` +
  `moe_compute` collectives followed by a `cluster_axis=0` `all_gather` — at the **exact** `(4,8)` dispatch-4 topology,
  run as a **single trace program** (no inter-op boundary), with **fresh per-layer semaphores**, and even with a full
  **201088-wide** victim all_gather — **never hangs**, at 1, 2, or 16 blocks. (Harness attached: §7.)
- It is **NOT a cross-axis interaction.** Removing the attention `cluster_axis=1` all-gathers (graph-verified) does
  **not** fix it — the 2-layer decode still hangs at the same `cluster_axis=0` all_gather.
- An earlier hypothesis of ours — a credit leak in the per-op `tt_fabric_mux` departure-gated teardown — is **refuted**
  by the negative control above (the same mux + collectives don't hang in isolation).

⟹ The hang requires the **full compiled decode graph's state**, not the fused collectives alone. The most likely
remaining causes are **systemic**: L1/DRAM/core pressure or resource/allocation state of the whole compiled program
when this large `cluster_axis=0` all_gather runs, after 2 layers of attention + MoE + const-eval'd weights + KV cache.
Pinning it down needs to see *where on the device* the all_gather worker is stuck — which we can't (§6).

---

## 2. Minimal reproducer (tt-xla)

```bash
# tt-xla, branch hshah/moe-compute-path
pytest -svv --num-layers 2 tests/benchmark/test_llms.py::test_gpt_oss_20b_moe_fused_galaxy_pcc
```

- `--num-layers 1` → **PASS** (prefill PCC 0.999905, decode PCC 0.999237; device == CPU reference).
- `--num-layers 2` (or full 24) → **HANG** (~10 min in), at the LM-head logit `all_gather` (see §3).

Config: `(4,8)` mesh; `"batch"`=axis0 (size 4, EP / `cluster_axis=0` dispatch ring), `"model"`=axis1 (size 8, attention
TP); `FABRIC_1D_RING`; experts EP-4 on axis0; attention TP-8 on axis1; lm_head replicated; weights `bfloat4_b`. Each
decode layer = `{attention (axis-1 all_gathers) → MoE (axis-0 dispatch + moe_compute)}`; tail = `rms_norm → matmul
(→201088) → all_gather(cluster_axis=0)`.

---

## 3. Hang signature and exact localization (trace-proven)

We instrumented the tt-mlir runtime op-dispatch loop (per-op `ENTER`/`EXIT` to stderr) plus the two CCL handlers. At the
2-layer hang:

```
[MOE_TRACE] all_to_all_dispatch #1 ENTER / EXIT      layer 0 dispatch  — completes
[MOE_TRACE] moe_compute        #1 ENTER / EXIT       layer 0 combine   — completes
[MOE_TRACE] all_to_all_dispatch #2 ENTER / EXIT      layer 1 dispatch  — completes
[MOE_TRACE] moe_compute        #2 ENTER / EXIT       layer 1 combine   — completes
...
[OP_TRACE] #1424 ENTER RMSNormOp   / EXIT
[OP_TRACE] #1427 ENTER MatmulOp    / EXIT            LM head (-> 8x201088)
[OP_TRACE] #1430 ENTER ReshapeOp   / EXIT
[OP_TRACE] #1431 ENTER AllGatherOp                   cluster_axis=0 logit gather  [NO EXIT — HANGS HERE]
```

Every MoE collective and every op up to the LM-head matmul **returns**; the process then busy-polls forever inside the
`all_gather`. Physical signature (measured via `/proc` + hwmon; ptrace is blocked in our container, §6): one host thread
pegged `R` with `voluntary_ctxt_switches` **frozen** for minutes; all 32 chips uniformly **~35 W, dead-steady** (parked
on a fabric wait; idle/released is ~23 W). A clean **`SIGTERM`** unwinds it and releases the mesh.

---

## 4. What we ruled out (this is the valuable part)

**(a) The fused-MoE collectives are NOT the cause — pure-ttnn negative control.** We built a standalone ttnn test
(`tmp/moe_credit_leak_repro.py`, §7) that drives the **real** `ttnn.experimental.moe_compute` (internal combine) +
`all_to_all_dispatch_metadata` via `TTMoEDecode`, then a `cluster_axis=0` `all_gather`, with **no** synchronize between
blocks. It does **not** hang under any of:

| Variant | Result |
|---|---|
| eager, 1 / 2 / 16 blocks | PASS |
| single **trace** program (all ops back-to-back, no boundary — like the compiled graph) | PASS |
| **fresh per-layer semaphores** (like tt-xla's prelude-hoisted per-collective semaphores) | PASS |
| exact **`(4,8)` dispatch-4** topology | PASS |
| **201088-wide** victim all_gather (matching the real LM-head gather) | PASS |
| all of the above **combined** | PASS |

So `{dispatch, moe_compute} × N → cluster_axis=0 all_gather`, in isolation, is fine — even at the exact topology, as one
back-to-back trace, with fresh semaphores and the full victim width.

**(b) Cross-axis interaction is NOT the cause.** We replicated attention (dropped TP-8 on `"model"`/axis1 — both the
attention weights and the KV-cache head sharding), graph-verified that the two `cluster_axis=1` attention all-gathers
were removed, and re-ran 2-layer on device: **still hangs** at the `cluster_axis=0` all_gather. So the axis-1 attention
CCLs coexisting with the axis-0 MoE CCLs is not the trigger.

**(c) Not the per-op mux teardown / EDM credit leak.** Our earlier hypothesis (the `tt_fabric_mux` graceful-termination
drain is gated on departure — `all_channels_drained`, `tt_fabric_mux.cpp:230-242` / `forward_data` frees slots on
`send_payload_flush_non_blocking` at `:117-124` "not handling acks" — closing the EDM connection before forwarded
packets are completion-credited into the never-reset single-digit sender-channel pool) is **refuted** by (a): the same
mux + collectives don't hang in isolation. If there is a leak, it is not sufficient on its own.

**(d) Not the LM-head sharding hook.** Gating the test's `sharding_constraint_hook` on lm_head was a graph no-op — the
all_gather comes from the output sharding propagation, not that hook.

---

## 5. What remains (full-graph-only — needs your instrumentation)

The hang reproduces **only** in the full compiled decode graph, at a large `cluster_axis=0` `all_gather` that runs after
2 layers of attention + MoE. The remaining candidates, none cleanly ablatable from our side:

- **Systemic resource/allocation state**: L1/DRAM/core pressure from the whole program (2× attention + 2× MoE with its
  persistent buffers + mux cores + const-eval'd `bfloat4_b` weights + paged KV cache), such that the `all_gather`'s
  workers/EDM connection can't get a core / L1 region / buffer they need and block. The fast harness has abundant free
  resources; the full graph does not.
- **Data-dependent chain**: the all_gather's input is produced by the LM-head matmul (`2880→201088`); some interaction
  of that big matmul's allocation/completion with the following gather.
- **tt-mlir lowering specifics**: mux core ranges, `num_links`, buffer/memory configs, or const-eval buffer placement in
  the full program differing from the fast harness's `TTMoEDecode` defaults.

**What we need:** Watcher / tt-lens on the actual 2-layer hang to show **which core / waypoint / semaphore the
`all_gather` workers (or their EDM connection) are stuck on**, and whether it's a resource/allocation stall vs a fabric
credit/connection stall. That single data point would resolve it; we can't get it (§6).

---

## 6. Separate blocker: Watcher is broken in this build (please also fix / advise)

We could not use `TT_METAL_WATCHER` to instrument the hang. On this build:
1. It hard-`TT_THROW`s on **fast-dispatch core (0,0)** reading `watcher.enable == 0`
   (`tt_metal/impl/debug/watcher_device_reader.cpp:588`) — an unconditional throw not guarded by any
   `TT_METAL_WATCHER_DISABLE_*` flag.
2. Patching that throw to warn-and-skip lets Watcher continue, but it then **segfaults in the ETH-core
   retraining-count dump** (`DumpEthLinkStatus`) during setup, before reaching the decode.
3. Each crash is an unclean abort **during fabric bring-up, which strands an ETH core** → the mesh needs a reset. (No
   `tt-smi` on this host and the container has `CapEff=0`, so recovery requires an operator reset each time.)

So Watcher needs fixing (fast-dispatch `watcher.enable` handling + the ETH-dump segfault), or you'll need to repro in a
Watcher-working environment.

---

## 7. Pointers / artifacts

- **Negative-control harness (proven does NOT reproduce):** `tmp/moe_credit_leak_repro.py`. Standalone ttnn; drives the
  real `moe_compute`+`dispatch`+`all_gather`. Env knobs: `NUM_BLOCKS`, `MESH=4,8|8,4`, `TRACE=1`, `FRESH_SEMS=1`,
  `WIDE=201088`. Run from the tt-xla venv (`python tmp/moe_credit_leak_repro.py`). Use it as your baseline: if your fix
  makes the full graph pass, this should still pass; and if you can make *this* hang, you've found the missing
  ingredient.
- **Diagnostic env gates left in the tt-xla test (dormant):** `TTXLA_NO_LMHEAD_GATHER` (llm_pcc.py; no-op, see §4d),
  `TTXLA_NO_ATTN_TP` (test_llms.py; replicates attention — used for §4b). Runtime instrumentation: `MOE_TRACE` (always
  on) + `TTXLA_OP_TRACE` per-op ENTER/EXIT in `program_executor.cpp`.
- Full investigation log: `MOE_FUSED_DECODE_WH_PROGRESS_HANDOFF.md`.
- 2-layer decode IR: `moe_2layer_decode_ttnn_graph.mlir` (the hang op is `%159`, the `cluster_axis=0` `all_gather` at
  line 961).
- Companion: `MOE_FUSED_DECODE_BH_GALAXY_TTMETAL_ESCALATION.md` (the earlier Blackhole hang).
